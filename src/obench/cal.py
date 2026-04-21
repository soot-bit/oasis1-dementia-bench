from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .io import read_sheet
from .utils.fp import mk


@dataclass(frozen=True)
class CalRes:
    n: int
    auc: float
    auc_flip: float
    bal_acc: float
    brier: float
    ece: float
    p_min: float
    p_mean: float
    p_max: float
    p_std: float


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    # Expected Calibration Error (binary), equally spaced bins in [0,1]
    y = y.astype(int)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = float(y[m].mean())
        conf = float(p[m].mean())
        out += (m.sum() / n) * abs(acc - conf)
    return float(out)


def _read_pred(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text())
        return pd.DataFrame([{"id": k, "p": float(v)} for k, v in d.items()])
    df = pd.read_csv(p)
    if "id" not in df.columns:
        raise ValueError(f"missing 'id' col in {p}")
    if "p" not in df.columns:
        raise ValueError(f"missing 'p' col in {p}")
    return df[["id", "p"]].copy()


def _labels(sheet: Path) -> pd.DataFrame:
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    sh["CDR"] = pd.to_numeric(sh.get("CDR"), errors="coerce")
    sh = sh[~sh["CDR"].isna()].copy()
    sh["y"] = (sh["CDR"] > 0).astype(int)
    return sh[["id", "y", "CDR"]].copy()


def run_cal(pred: Path, sheet: Path, out: Path, bins: int = 10) -> None:
    pr = _read_pred(pred)
    lab = _labels(sheet)
    df = pr.merge(lab, on="id", how="inner")

    y = df["y"].to_numpy(dtype=int)
    p = np.clip(df["p"].to_numpy(dtype=float), 0.0, 1.0)

    if len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, p))
        auc_flip = float(roc_auc_score(y, 1.0 - p))
    else:
        auc = float("nan")
        auc_flip = float("nan")
    bac = float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    brier = float(np.mean((p - y) ** 2))
    ece = _ece(y, p, bins=bins)

    res = CalRes(
        n=int(len(df)),
        auc=auc,
        auc_flip=auc_flip,
        bal_acc=bac,
        brier=brier,
        ece=ece,
        p_min=float(np.min(p)) if len(p) else float("nan"),
        p_mean=float(np.mean(p)) if len(p) else float("nan"),
        p_max=float(np.max(p)) if len(p) else float("nan"),
        p_std=float(np.std(p)) if len(p) else float("nan"),
    )

    p_out = mk(out)
    (p_out / "metrics.json").write_text(json.dumps(res.__dict__, indent=2) + "\n")
    df.assign(ent=_entropy(p)).to_csv(p_out / "pred_with_labels.csv", index=False)

    # reliability diagram
    edges = np.linspace(0.0, 1.0, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    accs = []
    confs = []
    ws = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            accs.append(np.nan)
            confs.append(float(mids[i]))
            ws.append(0)
            continue
        accs.append(float(y[m].mean()))
        confs.append(float(p[m].mean()))
        ws.append(int(m.sum()))

    plt.figure(figsize=(4.5, 4.0))
    plt.plot([0, 1], [0, 1], "--", color="#666", linewidth=1)
    plt.scatter(confs, accs, s=[max(20, 8 * w) for w in ws], alpha=0.9)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("mean predicted probability")
    plt.ylabel("empirical accuracy")
    plt.title(f"Reliability (ECE={ece:.3f})")
    plt.tight_layout()
    plt.savefig(p_out / "reliability.png", dpi=200)
    plt.close()

    # confidence histogram
    plt.figure(figsize=(5.5, 3.4))
    plt.hist(p, bins=20, alpha=0.9, color="#2a6fdb")
    plt.xlabel("predicted probability")
    plt.ylabel("count")
    plt.title("Confidence histogram")
    plt.tight_layout()
    plt.savefig(p_out / "conf_hist.png", dpi=200)
    plt.close()

    # entropy histogram
    ent = _entropy(p)
    plt.figure(figsize=(5.5, 3.4))
    plt.hist(ent, bins=20, alpha=0.9, color="#7a3db8")
    plt.xlabel("predictive entropy (nats)")
    plt.ylabel("count")
    plt.title("Uncertainty (entropy)")
    plt.tight_layout()
    plt.savefig(p_out / "entropy_hist.png", dpi=200)
    plt.close()
