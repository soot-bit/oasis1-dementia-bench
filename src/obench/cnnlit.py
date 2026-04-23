from __future__ import annotations

import json
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, balanced_accuracy_score, roc_auc_score
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from .cnn2d import AttentionPool, Cfg, ResNet18, TinyNet, Vol2D, _best_thr, _collate, _eval, _met, _step_pool
from .io import read_lines, read_sheet
from .utils.fp import mk

con = Console()


class LitCnn(pl.LightningModule):
    def __init__(self, arch: str, ch: int, pool: str, lr: float, pos_w: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = ResNet18(in_ch=ch) if arch == "resnet18" else TinyNet(in_ch=ch)
        self.pool = pool
        self.attn = None
        if pool == "attn":
            dim = 512 if arch == "resnet18" else 64
            self.attn = AttentionPool(dim=dim)
        self.lr = float(lr)
        self.register_buffer("pos_w", torch.tensor([float(pos_w)]))
        self.v_y: list[np.ndarray] = []
        self.v_p: list[np.ndarray] = []
        self.tr_loss: list[float] = []
        self.hist: list[dict[str, float]] = []

    def on_train_epoch_start(self) -> None:
        self.tr_loss = []

    def training_step(self, batch, batch_idx):
        xb, yb, _ = batch
        logit = _step_pool(self.net, xb, pool=self.pool, attn=self.attn)
        loss = F.binary_cross_entropy_with_logits(logit, yb.float(), pos_weight=self.pos_w)
        self.tr_loss.append(float(loss.detach().cpu()))
        self.log("tr_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(yb))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.v_y = []
        self.v_p = []

    def validation_step(self, batch, batch_idx):
        xb, yb, _ = batch
        logit = _step_pool(self.net, xb, pool=self.pool, attn=self.attn)
        p = torch.sigmoid(logit).detach().cpu().numpy()
        self.v_p.append(p)
        self.v_y.append(yb.detach().cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        p = np.concatenate(self.v_p) if self.v_p else np.array([], dtype=float)
        y = np.concatenate(self.v_y) if self.v_y else np.array([], dtype=int)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        bac = float(balanced_accuracy_score(y, (p >= 0.5).astype(int))) if len(y) else float("nan")
        p_std = float(np.std(p)) if len(p) else float("nan")
        loss = float(np.mean(self.tr_loss)) if self.tr_loss else float("nan")
        self.hist.append(
            {
                "ep": int(self.current_epoch + 1),
                "tr_loss": loss,
                "va_auc": auc,
                "va_bal_acc05": bac,
                "va_p_std": p_std,
            }
        )
        self.log("val_auc", auc, prog_bar=True, batch_size=len(y))
        self.log("val_bal_acc05", bac, prog_bar=True, batch_size=len(y))
        self.log("val_p_std", p_std, prog_bar=True, batch_size=len(y))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def _data(index: Path, sheet: Path, splits: Path):
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
    return df[df["id"].isin(tr)].copy(), df[df["id"].isin(va)].copy(), df[df["id"].isin(te)].copy()


def _save(out: Path, lit: LitCnn, dte: pd.DataFrame, dl_te: DataLoader, dev: torch.device, thr: float, hist: list[dict[str, float]], train: dict[str, object], seed: int) -> dict[str, object]:
    lit.net.to(dev)
    if lit.attn is not None:
        lit.attn.to(dev)
    auc, bac05, pred = _eval(lit.net, dl_te, dev, pool=lit.pool, attn=lit.attn)

    te = dte[["id", "subj", "Age", "M/F", "CDR", "MMSE"]].copy()
    te["p"] = te["id"].map(pred).astype(float)
    te["y"] = (pd.to_numeric(dte["CDR"], errors="coerce") > 0).astype(int).to_numpy()
    te["yhat"] = (te["p"] >= thr).astype(int)
    te["ok"] = (te["y"] == te["yhat"]).astype(int)

    y = te["y"].to_numpy(dtype=int)
    p = te["p"].to_numpy(dtype=float)
    met = _met(y=y, p=p, thr=thr, seed=seed)
    met["thr"] = float(thr)
    met["dev"] = str(dev)

    run = mk(out / "run")
    (run / "metrics.json").write_text(json.dumps(met, indent=2) + "\n")
    (run / "train.json").write_text(json.dumps(train, indent=2) + "\n")
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    torch.save(lit.net.state_dict(), run / "model.pt")
    if lit.attn is not None:
        torch.save(lit.attn.state_dict(), run / "attn.pt")
    (run / "pred.json").write_text(json.dumps(pred, indent=2) + "\n")
    te.sort_values(["ok", "p"], ascending=[True, False]).to_csv(run / "errors.csv", index=False)

    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, p)
    plt.tight_layout()
    plt.savefig(run / "roc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.0, 3.6))
    plt.hist(te[te["y"] == 0]["p"].to_numpy(), bins=20, alpha=0.7, label="y=0")
    plt.hist(te[te["y"] == 1]["p"].to_numpy(), bins=20, alpha=0.7, label="y=1")
    plt.axvline(thr, color="#111", linestyle="--", linewidth=1)
    plt.xlabel("predicted probability")
    plt.ylabel("count")
    plt.title("Prediction distribution (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / "pred_dist.png", dpi=200)
    plt.close()

    ps = {
        "p_min": float(np.min(p)) if len(p) else float("nan"),
        "p_mean": float(np.mean(p)) if len(p) else float("nan"),
        "p_max": float(np.max(p)) if len(p) else float("nan"),
        "p_std": float(np.std(p)) if len(p) else float("nan"),
    }
    (run / "pred_stats.json").write_text(json.dumps(ps, indent=2) + "\n")

    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, (p >= thr).astype(int), normalize="true")
    plt.tight_layout()
    plt.savefig(run / "cm.png", dpi=200)
    plt.close()

    return met


def run_cnnlit(
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
    patience: int = 8,
    workers: int = 0,
    precision: str = "auto",
) -> None:
    pl.seed_everything(seed, workers=True)
    dtr, dva, dte = _data(index=index, sheet=sheet, splits=splits)
    cfg = Cfg(slices=int(slices), pick=str(pick), aug=bool(aug), pool=str(pool), axis=int(axis), ch=int(ch))
    ev = Cfg(slices=cfg.slices, pick=cfg.pick, aug=False, pool=cfg.pool, axis=cfg.axis, ch=cfg.ch)

    n_workers = int(max(0, workers))
    pin = bool(torch.cuda.is_available())
    dl_kw = {
        "batch_size": int(bs),
        "num_workers": n_workers,
        "collate_fn": _collate,
        "pin_memory": pin,
        "persistent_workers": n_workers > 0,
    }
    dl_tr = DataLoader(Vol2D(dtr, cfg), shuffle=True, **dl_kw)
    dl_va = DataLoader(Vol2D(dva, ev), shuffle=False, **dl_kw)
    dl_te = DataLoader(Vol2D(dte, ev), shuffle=False, **dl_kw)

    ytr = dtr["y"].to_numpy(dtype=int)
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    pos_w = (n_neg / max(1, n_pos)) if (n_pos > 0 and n_neg > 0) else 1.0

    t0 = Table(title="cnnlit train", show_lines=False)
    t0.add_column("key")
    t0.add_column("value")
    t0.add_row("trainer", "lightning")
    t0.add_row("epochs", str(int(epochs)))
    t0.add_row("bs", str(int(bs)))
    t0.add_row("lr", f"{float(lr):g}")
    t0.add_row("arch", str(arch))
    t0.add_row("axis / ch", f"{int(axis)} / {int(ch)}")
    t0.add_row("slices / pick", f"{int(slices)} / {pick}")
    t0.add_row("pool", str(pool))
    t0.add_row("patience", str(int(patience)))
    t0.add_row("workers", str(n_workers))
    t0.add_row("pos_weight", f"{pos_w:.3f} (neg={n_neg}, pos={n_pos})")
    t0.add_row("n_tr / n_va / n_te", f"{len(dtr)} / {len(dva)} / {len(dte)}")
    con.print(t0)

    lit = LitCnn(arch=arch, ch=int(ch), pool=str(pool), lr=float(lr), pos_w=float(pos_w))
    ckpt = ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1, save_weights_only=True)
    stop = EarlyStopping(monitor="val_auc", mode="max", patience=int(patience))
    prec = "16-mixed" if precision == "auto" and torch.cuda.is_available() else precision
    if prec == "auto":
        prec = "32-true"
    trainer = pl.Trainer(
        max_epochs=int(epochs),
        accelerator="auto",
        devices=1,
        precision=prec,
        logger=False,
        enable_checkpointing=True,
        callbacks=[ckpt, stop],
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        default_root_dir=str(out / "lit"),
    )
    trainer.fit(lit, train_dataloaders=dl_tr, val_dataloaders=dl_va)

    if ckpt.best_model_path:
        state = torch.load(ckpt.best_model_path, map_location="cpu")
        lit.load_state_dict(state["state_dict"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.net.to(dev)
    if lit.attn is not None:
        lit.attn.to(dev)
    _, _, va_pred = _eval(lit.net, dl_va, dev, pool=str(pool), attn=lit.attn)
    va = pd.DataFrame([{"id": k, "p": v} for k, v in va_pred.items()]).merge(dva[["id", "y"]], on="id")
    thr = _best_thr(va["y"].to_numpy(dtype=int), va["p"].to_numpy(dtype=float))
    train = {
        "trainer": "lightning",
        "epochs": int(epochs),
        "bs": int(bs),
        "lr": float(lr),
        "pick": str(pick),
        "pool": str(pool),
        "slices": int(slices),
        "axis": int(axis),
        "ch": int(ch),
        "arch": str(arch),
        "patience": int(patience),
        "workers": int(n_workers),
        "precision": str(prec),
        "pos_weight": float(pos_w),
        "thr": float(thr),
        "best_model_path": str(ckpt.best_model_path),
        "n_tr": int(len(dtr)),
        "n_va": int(len(dva)),
        "n_te": int(len(dte)),
    }
    met = _save(out=out, lit=lit, dte=dte, dl_te=dl_te, dev=dev, thr=thr, hist=lit.hist, train=train, seed=seed)

    t1 = Table(title="cnnlit results (test)", show_lines=False)
    t1.add_column("metric")
    t1.add_column("value", justify="right")
    t1.add_row("thr (val)", f"{float(thr):.4f}")
    t1.add_row("auc", f"{float(met['auc']):.4f}")
    t1.add_row("bal_acc@0.5", f"{float(met['bal_acc05']):.4f}")
    t1.add_row("bal_acc@thr", f"{float(met['bal_acc']):.4f}")
    if "brier" in met:
        t1.add_row("brier", f"{float(met['brier']):.4f}")
    con.print(t1)
