from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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


def _pre(cfg: TabCfg, scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    num = Pipeline(steps=num_steps)

    cat = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num, list(cfg.num)),
            ("cat", cat, list(cfg.cat)),
        ],
        remainder="drop",
    )


def _pipe(cfg: TabCfg, name: str) -> Pipeline:
    if name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=7,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
        return Pipeline([("pre", _pre(cfg, scale=False)), ("clf", clf)])
    if name == "gb":
        clf = GradientBoostingClassifier(random_state=7)
        return Pipeline([("pre", _pre(cfg, scale=False)), ("clf", clf)])

    # default: log-reg
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("pre", _pre(cfg, scale=True)), ("clf", clf)])


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

    feats = list(cfg.num + cfg.cat)
    ytr = _y(dtr)
    yte = _y(dte)

    run_dir = mk(out / "run")
    models = [("logreg", "logistic regression"), ("rf", "random forest"), ("gb", "gradient boosting")]
    rows = []

    def eval_one(name: str) -> tuple[np.ndarray, float, float]:
        p = _pipe(cfg, name=name)
        p.fit(dtr[feats], ytr)
        te_p = p.predict_proba(dte[feats])[:, 1]
        auc = float(roc_auc_score(yte, te_p)) if len(np.unique(yte)) > 1 else float("nan")
        bac = float(balanced_accuracy_score(yte, (te_p >= 0.5).astype(int)))
        return te_p, auc, bac

    for name, disp in models:
        te_p, auc, bac = eval_one(name)
        rows.append({"model": name, "name": disp, "auc": auc, "bal_acc": bac, "n_test": int(len(dte))})

        # keep the original "run/" outputs as the logreg artefacts for downstream tooling
        out_dir = run_dir if name == "logreg" else mk(run_dir / name)
        (out_dir / "metrics.json").write_text(json.dumps({"auc": auc, "bal_acc": bac}, indent=2) + "\n")

        plt.figure(figsize=(5, 4))
        RocCurveDisplay.from_predictions(yte, te_p)
        plt.tight_layout()
        plt.savefig(out_dir / "roc.png", dpi=200)
        plt.close()

        plt.figure(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(yte, (te_p >= 0.5).astype(int), normalize="true")
        plt.tight_layout()
        plt.savefig(out_dir / "cm.png", dpi=200)
        plt.close()

        err = dte[["id", "subj", "Age", "M/F", "CDR", "MMSE", "eTIV", "nWBV", "ASF"]].copy()
        err["p"] = te_p
        err["y"] = yte
        err["yhat"] = (te_p >= 0.5).astype(int)
        err["ok"] = (err["y"] == err["yhat"]).astype(int)
        err.sort_values(["ok", "p"], ascending=[True, False]).to_csv(out_dir / "errors.csv", index=False)

    summ = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summ.to_csv(run_dir / "summary.csv", index=False)

    # quick EDA plot (label vs age)
    if "Age" in dte.columns:
        plt.figure(figsize=(5, 4))
        sns.boxplot(data=dte.assign(y=yte), x="y", y="Age")
        plt.tight_layout()
        plt.savefig(run_dir / "age_by_y.png", dpi=200)
        plt.close()

    # comparison plot (AUC)
    plt.figure(figsize=(6.0, 3.2))
    xs = summ.sort_values("auc", ascending=True)
    plt.barh(xs["name"], xs["auc"], color="#2a6fdb", alpha=0.9)
    plt.xlim(0, 1)
    plt.xlabel("ROC-AUC")
    plt.title("Tabular baselines (test)")
    plt.tight_layout()
    plt.savefig(run_dir / "compare_auc.png", dpi=200)
    plt.close()
