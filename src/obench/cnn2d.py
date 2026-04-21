from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .io import read_lines, read_sheet
from .utils.fp import mk
from .utils.img import load_analyze, zscore_brain


@dataclass(frozen=True)
class Cfg:
    slices: int = 24  # per volume (evenly spaced)


def _y(df: pd.DataFrame) -> np.ndarray:
    cdr = pd.to_numeric(df["CDR"], errors="coerce")
    if np.any(np.isnan(cdr.to_numpy())):
        raise ValueError("missing CDR labels; filter rows with CDR before calling _y()")
    return (cdr > 0).astype(int).to_numpy()


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
        # pick evenly spaced slices along last axis
        z = a.shape[-1]
        k = min(self.cfg.slices, z)
        ix = np.linspace(0, z - 1, num=k).astype(int)
        xs = a[:, :, ix]  # H,W,k
        xs = np.moveaxis(xs, -1, 0)  # k,H,W
        xs = xs[:, None, :, :]  # k,1,H,W
        y = int(r["y"])
        return torch.from_numpy(xs), torch.tensor(y, dtype=torch.long), str(r["id"])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.h = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B*K,1,H,W
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.c3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.h(x).squeeze(1)


def _collate(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(ids)


def _step(net: Net, xb: torch.Tensor) -> torch.Tensor:
    # xb: B,K,1,H,W -> flatten to B*K
    b, k = xb.shape[:2]
    x = xb.view(b * k, *xb.shape[2:])
    logit = net(x).view(b, k)
    return logit.mean(1)  # mean over slices


def _eval(net: Net, dl: DataLoader, dev: torch.device) -> tuple[float, float, dict[str, float]]:
    net.eval()
    ps = []
    ys = []
    ids = []
    with torch.no_grad():
        for xb, yb, idb in dl:
            xb = xb.to(dev)
            logit = _step(net, xb)
            p = torch.sigmoid(logit).cpu().numpy()
            ps.append(p)
            ys.append(yb.numpy())
            ids += idb
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    bac = float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    return auc, bac, {i: float(pp) for i, pp in zip(ids, p)}


def run_cnn2d(index: Path, sheet: Path, splits: Path, out: Path, seed: int, epochs: int, bs: int, lr: float) -> None:
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

    cfg = Cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    dl_tr = DataLoader(Vol2D(dtr, cfg), batch_size=bs, shuffle=True, num_workers=0, collate_fn=_collate)
    dl_va = DataLoader(Vol2D(dva, cfg), batch_size=bs, shuffle=False, num_workers=0, collate_fn=_collate)
    dl_te = DataLoader(Vol2D(dte, cfg), batch_size=bs, shuffle=False, num_workers=0, collate_fn=_collate)

    best = {"auc": -1.0, "state": None}
    for _ in range(epochs):
        net.train()
        for xb, yb, _ in tqdm(dl_tr, desc="train", leave=False):
            xb = xb.to(dev)
            yb = yb.to(dev).float()
            logit = _step(net, xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        auc, bac, _ = _eval(net, dl_va, dev)
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    if best["state"] is not None:
        net.load_state_dict(best["state"])

    te_auc, te_bac, te_p = _eval(net, dl_te, dev)

    run_dir = mk(out / "run")
    (run_dir / "metrics.json").write_text(json.dumps({"auc": te_auc, "bal_acc": te_bac, "dev": str(dev)}, indent=2) + "\n")
    torch.save(net.state_dict(), run_dir / "model.pt")
    (run_dir / "pred.json").write_text(json.dumps(te_p, indent=2) + "\n")

    # plots + errors
    te_df = dte[["id", "subj", "Age", "M/F", "CDR", "MMSE"]].copy()
    te_df["p"] = te_df["id"].map(te_p).astype(float)
    te_df["y"] = _y(dte)
    te_df["yhat"] = (te_df["p"] >= 0.5).astype(int)
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

    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, (p >= 0.5).astype(int), normalize="true")
    plt.tight_layout()
    plt.savefig(run_dir / "cm.png", dpi=200)
    plt.close()
