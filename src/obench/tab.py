from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import read_lines, read_sheet
from .utils.fp import mk


@dataclass(frozen=True)
class TabCfg:
    # keep explicit + small
    num: tuple[str, ...] = ("Age", "Educ", "SES", "eTIV", "nWBV", "ASF")
    cat: tuple[str, ...] = ("M/F", "Hand")


def _y(df: pd.DataFrame) -> np.ndarray:
    cdr = pd.to_numeric(df["CDR"], errors="coerce")
    if np.any(np.isnan(cdr.to_numpy())):
        raise ValueError("missing CDR labels; filter rows with CDR before calling _y()")
    return (cdr > 0).astype(int).to_numpy()


def _prep(df: pd.DataFrame, cfg: TabCfg) -> pd.DataFrame:
    x = df.copy()
    for c in cfg.num:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in cfg.cat:
        if c in x.columns:
            x[c] = x[c].astype(str).replace({"nan": np.nan})
    return x


def _pipe(cfg: TabCfg) -> Pipeline:
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
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def _subset(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return df[df["id"].isin(ids)].copy()


def run_tab(index: Path, sheet: Path, splits: Path, out: Path) -> None:
    cfg = TabCfg()

    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    df = idx.merge(sh, on="id", how="inner")
    df = df[df["id"].str.endswith("_MR1")].copy()
    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df[~df["CDR"].isna()].copy()
    df = _prep(df, cfg)

    tr = read_lines(splits / "train.txt")
    va = read_lines(splits / "val.txt")
    te = read_lines(splits / "test.txt")

    dtr = _subset(df, tr)
    dva = _subset(df, va)
    dte = _subset(df, te)

    p = _pipe(cfg)
    p.fit(dtr[list(cfg.num + cfg.cat)], _y(dtr))

    def pred(d: pd.DataFrame) -> np.ndarray:
        return p.predict_proba(d[list(cfg.num + cfg.cat)])[:, 1]

    te_p = pred(dte)
    te_y = _y(dte)

    auc = float(roc_auc_score(te_y, te_p))
    bac = float(balanced_accuracy_score(te_y, (te_p >= 0.5).astype(int)))

    run_dir = mk(out / "run")
    (run_dir / "metrics.json").write_text(json.dumps({"auc": auc, "bal_acc": bac}, indent=2) + "\n")

    # plots
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

    # error table
    err = dte[["id", "subj", "Age", "M/F", "CDR", "MMSE", "eTIV", "nWBV", "ASF"]].copy()
    err["p"] = te_p
    err["y"] = te_y
    err["yhat"] = (te_p >= 0.5).astype(int)
    err["ok"] = (err["y"] == err["yhat"]).astype(int)
    err.sort_values(["ok", "p"], ascending=[True, False]).to_csv(run_dir / "errors.csv", index=False)

    # quick EDA plot (label vs age)
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=err, x="y", y="Age")
    plt.tight_layout()
    plt.savefig(run_dir / "age_by_y.png", dpi=200)
    plt.close()
