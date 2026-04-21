from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import math

from .io import read_lines, read_sheet
from .utils.fp import mk
from .utils.img import load_analyze, zscore_brain


@dataclass(frozen=True)
class Cfg:
    slices: int = 24  # per volume (evenly spaced)
    pick: str = "topnz"  # topnz|lin
    aug: bool = True
    pool: str = "max"  # mean|max|lse
    axis: int = 2
    ch: int = 1


def _y(df: pd.DataFrame) -> np.ndarray:
    cdr = pd.to_numeric(df["CDR"], errors="coerce")
    if np.any(np.isnan(cdr.to_numpy())):
        raise ValueError("missing CDR labels; filter rows with CDR before calling _y()")
    return (cdr > 0).astype(int).to_numpy()


def _pick_slices(a: np.ndarray, k: int, mode: str) -> np.ndarray:
    z = int(a.shape[-1])
    if z <= 0:
        return np.array([], dtype=int)
    k = int(min(max(1, k), z))

    if mode == "lin":
        return np.linspace(0, z - 1, num=k).astype(int)

    # default: choose slices with most non-zero voxels (brain content)
    nz = np.count_nonzero(a != 0, axis=(0, 1))  # (Z,)
    ix = np.argsort(-nz)[:k]
    ix = np.sort(ix)
    return ix.astype(int)


def _aug(x: torch.Tensor) -> torch.Tensor:
    # x: K,1,H,W (float32)
    # light, label-preserving intensity/noise aug
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-1])  # left-right
    if torch.rand(()) < 0.2:
        x = torch.flip(x, dims=[-2])  # up-down (small prob)
    if torch.rand(()) < 0.9:
        s = 0.90 + 0.20 * torch.rand(())
        x = x * s
    if torch.rand(()) < 0.9:
        b = (-0.10 + 0.20 * torch.rand(()))
        x = x + b
    if torch.rand(()) < 0.5:
        n = 0.01 * torch.rand(())
        x = x + n * torch.randn_like(x)
    return x


class Vol2D(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        p = Path(r["t88_mask"])
        a = zscore_brain(load_analyze(p))
        # slice along requested axis (0/1/2), always treat slice dimension as last
        if self.cfg.axis == 0:
            a2 = np.moveaxis(a, 0, -1)  # (Y,Z,X)
        elif self.cfg.axis == 1:
            a2 = np.moveaxis(a, 1, -1)  # (X,Z,Y)
        else:
            a2 = a  # (X,Y,Z)

        ix = _pick_slices(a2, self.cfg.slices, self.cfg.pick)

        # build K slices, each is either 1ch or 3ch (2.5D) using neighbors
        sl = []
        for j in ix.tolist():
            if self.cfg.ch == 3:
                js = [max(0, j - 1), j, min(a2.shape[-1] - 1, j + 1)]
                s = np.stack([a2[..., t] for t in js], axis=0)  # 3,H,W
            else:
                s = a2[..., j][None, ...]  # 1,H,W
            sl.append(s.astype(np.float32))
        xs = np.stack(sl, axis=0)  # K,C,H,W
        x = torch.from_numpy(xs)
        if self.cfg.aug:
            x = _aug(x)
        y = int(r["y"])
        return x, torch.tensor(y, dtype=torch.long), str(r["id"])


class TinyNet(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.h = nn.Linear(64, 1)

    def enc(self, x: torch.Tensor) -> torch.Tensor:
        # x: B*K,1,H,W
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.c3(x))
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # B*K,64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc(x)
        return self.h(e).squeeze(1)


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_c)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        if self.down is not None:
            r = self.down(r)
        x = F.relu(x + r)
        return x


class ResNet18(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.c0 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.b0 = nn.BatchNorm2d(64)
        self.p0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l1 = self._layer(64, 64, n=2, stride=1)
        self.l2 = self._layer(64, 128, n=2, stride=2)
        self.l3 = self._layer(128, 256, n=2, stride=2)
        self.l4 = self._layer(256, 512, n=2, stride=2)
        self.h = nn.Linear(512, 1)

    def _layer(self, in_c: int, out_c: int, n: int, stride: int) -> nn.Sequential:
        xs = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(n - 1):
            xs.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*xs)

    def enc(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.b0(self.c0(x)))
        x = self.p0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # B,512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc(x)
        return self.h(e).squeeze(1)


def _collate(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(ids)


def _step_pool(net: nn.Module, xb: torch.Tensor, pool: str) -> torch.Tensor:
    b, k = xb.shape[:2]
    x = xb.view(b * k, *xb.shape[2:])
    if pool == "lse":
        logit = net(x).view(b, k)
        return torch.logsumexp(logit, dim=1) - math.log(max(1, k))

    e = net.enc(x).view(b, k, -1)
    if pool == "mean":
        g = e.mean(1)
    else:
        g = e.max(1).values
    return net.h(g).squeeze(1)


def _emb_pool(net: nn.Module, xb: torch.Tensor, pool: str) -> torch.Tensor:
    b, k = xb.shape[:2]
    x = xb.view(b * k, *xb.shape[2:])
    e = net.enc(x).view(b, k, -1)
    if pool == "mean":
        return e.mean(1)
    return e.max(1).values


def _eval(net: nn.Module, dl: DataLoader, dev: torch.device, pool: str) -> tuple[float, float, dict[str, float]]:
    net.eval()
    ps = []
    ys = []
    ids = []
    with torch.no_grad():
        for xb, yb, idb in dl:
            xb = xb.to(dev)
            logit = _step_pool(net, xb, pool=pool)
            p = torch.sigmoid(logit).cpu().numpy()
            ps.append(p)
            ys.append(yb.numpy())
            ids += idb
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    bac = float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    return auc, bac, {i: float(pp) for i, pp in zip(ids, p)}


def _best_thr(y: np.ndarray, p: np.ndarray) -> float:
    # pick threshold that maximizes balanced accuracy on validation
    if len(p) == 0:
        return 0.5
    # avoid overfitting to tiny float jitter: coarse grid only
    xs = np.linspace(0.1, 0.9, 17)
    best = (0.5, balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    for t in xs:
        bac = balanced_accuracy_score(y, (p >= t).astype(int))
        if bac > best[1]:
            best = (float(t), float(bac))
    return float(best[0])


def run_cnn2d(
    index: Path,
    sheet: Path,
    splits: Path,
    out: Path,
    seed: int,
    epochs: int,
    bs: int,
    lr: float,
    slices: int,
    pick: str,
    pool: str,
    aug: bool,
    axis: int,
    ch: int,
    arch: str,
) -> None:
    con = Console()
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    df = idx.merge(sh, on="id", how="inner")
    df = df[df["id"].str.endswith("_MR1")].copy()
    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df[~df["CDR"].isna()].copy()
    df["y"] = (df["CDR"] > 0).astype(int)
    df = df[df["t88_mask"].astype(str).str.len() > 0].copy()

    tr = read_lines(splits / "train.txt")
    va = read_lines(splits / "val.txt")
    te = read_lines(splits / "test.txt")

    dtr = df[df["id"].isin(tr)].copy()
    dva = df[df["id"].isin(va)].copy()
    dte = df[df["id"].isin(te)].copy()

    cfg = Cfg(
        slices=int(slices),
        pick=str(pick),
        pool=str(pool),
        aug=bool(aug),
        axis=int(axis),
        ch=int(ch),
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == "resnet18":
        net = ResNet18(in_ch=cfg.ch).to(dev)
    else:
        net = TinyNet(in_ch=cfg.ch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    dl_tr = DataLoader(Vol2D(dtr, cfg), batch_size=bs, shuffle=True, num_workers=0, collate_fn=_collate)
    # disable augmentation for eval
    dl_va = DataLoader(
        Vol2D(dva, Cfg(slices=cfg.slices, pick=cfg.pick, aug=False, pool=cfg.pool, axis=cfg.axis, ch=cfg.ch)),
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    dl_te = DataLoader(
        Vol2D(dte, Cfg(slices=cfg.slices, pick=cfg.pick, aug=False, pool=cfg.pool, axis=cfg.axis, ch=cfg.ch)),
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    ytr = dtr["y"].to_numpy(dtype=int)
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    pos_w = (n_neg / max(1, n_pos)) if (n_pos > 0 and n_neg > 0) else 1.0
    pos_w_t = torch.tensor([float(pos_w)], device=dev)

    best = {"auc": -1.0, "state": None}
    best_thr = 0.5
    hist = []
    # training header
    t0 = Table(title="cnn2d train", show_lines=False)
    t0.add_column("key")
    t0.add_column("value")
    t0.add_row("dev", str(dev))
    t0.add_row("epochs", str(epochs))
    t0.add_row("bs", str(bs))
    t0.add_row("lr", str(lr))
    t0.add_row("slices", str(cfg.slices))
    t0.add_row("pick", str(cfg.pick))
    t0.add_row("pool", str(cfg.pool))
    t0.add_row("axis", str(cfg.axis))
    t0.add_row("ch", str(cfg.ch))
    t0.add_row("arch", str(arch))
    t0.add_row("pos_weight", f"{pos_w:.3f} (neg={n_neg}, pos={n_pos})")
    t0.add_row("n_tr / n_va / n_te", f"{len(dtr)} / {len(dva)} / {len(dte)}")
    con.print(t0)

    best_epoch = -1
    for ep in tqdm(range(epochs), desc="epoch", total=epochs):
        net.train()
        tr_loss = []
        for xb, yb, _ in tqdm(dl_tr, desc="train", leave=False):
            xb = xb.to(dev)
            yb = yb.to(dev).float()
            logit = _step_pool(net, xb, pool=cfg.pool)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss.append(float(loss.detach().cpu().item()))

        auc, bac05, va_p = _eval(net, dl_va, dev, pool=cfg.pool)
        va_df = pd.DataFrame([{"id": k, "p": float(v)} for k, v in va_p.items()])
        va_df = va_df.merge(dva[["id", "y"]], on="id", how="inner")
        v_y = va_df["y"].to_numpy(dtype=int)
        v_p = va_df["p"].to_numpy(dtype=float)
        if len(v_p) and float(np.std(v_p)) < 1e-3:
            thr = 0.5
            bac = float(bac05)
        else:
            thr = _best_thr(v_y, v_p)
            bac = float(balanced_accuracy_score(v_y, (v_p >= thr).astype(int))) if len(v_y) else float("nan")
        va_ps = np.array(list(va_p.values()), dtype=float) if va_p else np.array([], dtype=float)
        hist.append(
            {
                "ep": int(ep + 1),
                "tr_loss": float(np.mean(tr_loss)) if tr_loss else float("nan"),
                "va_auc": float(auc),
                "va_bal_acc05": float(bac05),
                "va_bal_acc": float(bac),
                "va_thr": float(thr),
                "va_p_std": float(np.std(va_ps)) if len(va_ps) else float("nan"),
            }
        )
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            best_thr = float(thr)
            best_epoch = int(ep + 1)

        # show epoch summary in tqdm
        tqdm.write(
            f"ep {ep+1:03d}/{epochs} tr_loss={hist[-1]['tr_loss']:.4f} "
            f"va_auc={auc:.3f} va_bac={hist[-1]['va_bal_acc']:.3f} thr={thr:.3f} va_p_std={hist[-1]['va_p_std']:.4g}"
        )

    if best["state"] is not None:
        net.load_state_dict(best["state"])

    te_auc, te_bac05, te_p = _eval(net, dl_te, dev, pool=cfg.pool)
    te_df = pd.DataFrame([{"id": k, "p": float(v)} for k, v in te_p.items()])
    te_df = te_df.merge(dte[["id", "y"]], on="id", how="inner")
    te_y = te_df["y"].to_numpy(dtype=int)
    te_pv = te_df["p"].to_numpy(dtype=float)
    te_bac = float(balanced_accuracy_score(te_y, (te_pv >= best_thr).astype(int))) if len(te_y) else float("nan")

    run_dir = mk(out / "run")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"auc": te_auc, "bal_acc": te_bac, "bal_acc05": te_bac05, "thr": best_thr, "dev": str(dev)},
            indent=2,
        )
        + "\n"
    )
    (run_dir / "train.json").write_text(
        json.dumps(
            {
                "epochs": int(epochs),
                "bs": int(bs),
                "lr": float(lr),
                "pick": cfg.pick,
                "pool": cfg.pool,
                "slices": int(cfg.slices),
                "axis": int(cfg.axis),
                "ch": int(cfg.ch),
                "arch": str(arch),
                "pos_weight": float(pos_w),
                "thr": float(best_thr),
                "best_epoch": int(best_epoch),
                "n_tr": int(len(dtr)),
                "n_va": int(len(dva)),
                "n_te": int(len(dte)),
            },
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    torch.save(net.state_dict(), run_dir / "model.pt")
    (run_dir / "pred.json").write_text(json.dumps(te_p, indent=2) + "\n")

    # plots + errors
    te_df = dte[["id", "subj", "Age", "M/F", "CDR", "MMSE"]].copy()
    te_df["p"] = te_df["id"].map(te_p).astype(float)
    te_df["y"] = _y(dte)
    te_df["yhat"] = (te_df["p"] >= best_thr).astype(int)
    te_df["ok"] = (te_df["y"] == te_df["yhat"]).astype(int)
    te_df.sort_values(["ok", "p"], ascending=[True, False]).to_csv(run_dir / "errors.csv", index=False)

    y = te_df["y"].to_numpy()
    p = te_df["p"].to_numpy()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, p)
    plt.tight_layout()
    plt.savefig(run_dir / "roc.png", dpi=200)
    plt.close()

    # prediction distribution (by class)
    plt.figure(figsize=(6.0, 3.6))
    p0 = te_df[te_df["y"] == 0]["p"].to_numpy()
    p1 = te_df[te_df["y"] == 1]["p"].to_numpy()
    plt.hist(p0, bins=20, alpha=0.7, label="y=0")
    plt.hist(p1, bins=20, alpha=0.7, label="y=1")
    plt.axvline(best_thr, color="#111", linestyle="--", linewidth=1)
    plt.xlabel("predicted probability")
    plt.ylabel("count")
    plt.title("Prediction distribution (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "pred_dist.png", dpi=200)
    plt.close()

    ps = {
        "p_min": float(np.min(p)) if len(p) else float("nan"),
        "p_mean": float(np.mean(p)) if len(p) else float("nan"),
        "p_max": float(np.max(p)) if len(p) else float("nan"),
        "p_std": float(np.std(p)) if len(p) else float("nan"),
    }
    (run_dir / "pred_stats.json").write_text(json.dumps(ps, indent=2) + "\n")

    # final console summary
    t1 = Table(title="cnn2d results (test)", show_lines=False)
    t1.add_column("metric")
    t1.add_column("value", justify="right")
    t1.add_row("best_epoch", str(best_epoch))
    t1.add_row("thr (val)", f"{best_thr:.4f}")
    t1.add_row("auc", f"{te_auc:.4f}")
    t1.add_row("bal_acc@0.5", f"{te_bac05:.4f}")
    t1.add_row("bal_acc@thr", f"{te_bac:.4f}")
    t1.add_row("p_std (test)", f"{ps['p_std']:.6f}")
    con.print(t1)

    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, (p >= best_thr).astype(int), normalize="true")
    plt.tight_layout()
    plt.savefig(run_dir / "cm.png", dpi=200)
    plt.close()
