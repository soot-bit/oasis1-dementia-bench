from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import read_lines, read_sheet
from .utils.fp import mk


@dataclass(frozen=True)
class FuseCfg:
    num: tuple[str, ...] = ("Age", "Educ", "SES", "eTIV", "nWBV", "ASF")
    cat: tuple[str, ...] = ("M/F", "Hand")


def _pipe(cfg: FuseCfg, model: str) -> Pipeline:
    num = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    cat = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num, list(cfg.num)),
            ("cat", cat, list(cfg.cat)),
        ],
        remainder="passthrough",  # keeps embedding columns
    )

    if model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
        )
    else:
        clf = LogisticRegression(max_iter=4000, class_weight="balanced")

    return Pipeline([("pre", pre), ("clf", clf)])


def _subset(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return df[df["id"].isin(ids)].copy()


def run_fuse(index: Path, sheet: Path, emb: Path, splits: Path, out: Path, model: str = "mlp") -> None:
    cfg = FuseCfg()

    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    sh["CDR"] = pd.to_numeric(sh.get("CDR"), errors="coerce")
    sh = sh[~sh["CDR"].isna()].copy()
    sh["y"] = (sh["CDR"] > 0).astype(int)

    e = pd.read_csv(emb)
    df = idx.merge(sh, on="id", how="inner").merge(e, on=["id", "y"], how="inner")
    df = df[df["id"].astype(str).str.endswith("_MR1")].copy()

    tr = read_lines(splits / "train.txt")
    va = read_lines(splits / "val.txt")
    te = read_lines(splits / "test.txt")

    dtr = _subset(df, tr)
    dva = _subset(df, va)
    dte = _subset(df, te)

    emb_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    feats = list(cfg.num + cfg.cat) + emb_cols

    p = _pipe(cfg, model=model)
    p.fit(dtr[feats], dtr["y"].to_numpy())

    te_p = p.predict_proba(dte[feats])[:, 1]
    te_y = dte["y"].to_numpy()

    auc = float(roc_auc_score(te_y, te_p)) if len(np.unique(te_y)) > 1 else float("nan")
    bac = float(balanced_accuracy_score(te_y, (te_p >= 0.5).astype(int)))

    run_dir = mk(out / "run")
    (run_dir / "metrics.json").write_text(json.dumps({"auc": auc, "bal_acc": bac, "model": model}, indent=2) + "\n")

    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(te_y, te_p)
    plt.tight_layout()
    plt.savefig(run_dir / "roc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(te_y, (te_p >= 0.5).astype(int), normalize="true")
    plt.tight_layout()
    plt.savefig(run_dir / "cm.png", dpi=200)
    plt.close()

    err = dte[["id", "subj", "Age", "M/F", "CDR", "MMSE", "eTIV", "nWBV", "ASF", "y"]].copy()
    err["p"] = te_p
    err["yhat"] = (te_p >= 0.5).astype(int)
    err["ok"] = (err["y"] == err["yhat"]).astype(int)
    err.sort_values(["ok", "p"], ascending=[True, False]).to_csv(run_dir / "errors.csv", index=False)
