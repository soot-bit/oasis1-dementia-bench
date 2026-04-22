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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import read_lines, read_sheet
from .utils.fp import mk


@dataclass(frozen=True)
class TabCfg:
    # keep explicit + small
    num: tuple[str, ...] = ("Age", "Educ", "SES", "eTIV", "nWBV", "ASF")
    cat: tuple[str, ...] = ("M/F", "Hand")


MODELS: tuple[tuple[str, str], ...] = (
    ("logreg", "logistic regression"),
    ("rf", "random forest"),
    ("gb", "gradient boosting"),
)


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


def _load(index: Path, sheet: Path, cfg: TabCfg) -> pd.DataFrame:
    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    df = idx.merge(sh, on="id", how="inner")
    df = df[df["id"].str.endswith("_MR1")].copy()
    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df[~df["CDR"].isna()].copy()
    return _prep(df, cfg)


def _fit_eval(name: str, cfg: TabCfg, feats: list[str], dtr: pd.DataFrame, dte: pd.DataFrame) -> tuple[Pipeline, np.ndarray, float, float]:
    ytr = _y(dtr)
    yte = _y(dte)
    pipe = _pipe(cfg, name=name)
    pipe.fit(dtr[feats], ytr)
    te_p = pipe.predict_proba(dte[feats])[:, 1]
    auc = float(roc_auc_score(yte, te_p)) if len(np.unique(yte)) > 1 else float("nan")
    bac = float(balanced_accuracy_score(yte, (te_p >= 0.5).astype(int)))
    return pipe, te_p, auc, bac


def _feat_names(pipe: Pipeline, cfg: TabCfg) -> list[str]:
    pre = pipe.named_steps["pre"]
    feat = list(cfg.num)
    oh = pre.named_transformers_["cat"].named_steps["oh"]
    feat.extend(oh.get_feature_names_out(list(cfg.cat)).tolist())
    return feat


def _coef_rows(pipe: Pipeline, cfg: TabCfg, fold: int) -> list[dict[str, object]]:
    clf = pipe.named_steps["clf"]
    feat = _feat_names(pipe, cfg)
    coef = clf.coef_.ravel().tolist()
    return [{"fold": fold, "feature": f, "coef": float(c), "abs_coef": float(abs(c))} for f, c in zip(feat, coef, strict=True)]


def _imp_rows(pipe: Pipeline, cfg: TabCfg, fold: int, model: str) -> list[dict[str, object]]:
    clf = pipe.named_steps["clf"]
    feat = _feat_names(pipe, cfg)
    imp = clf.feature_importances_.tolist()
    return [{"model": model, "fold": fold, "feature": f, "importance": float(v)} for f, v in zip(feat, imp, strict=True)]


def run_tab(index: Path, sheet: Path, splits: Path, out: Path) -> None:
    cfg = TabCfg()
    df = _load(index=index, sheet=sheet, cfg=cfg)

    tr = read_lines(splits / "train.txt")
    va = read_lines(splits / "val.txt")
    te = read_lines(splits / "test.txt")

    dtr = _subset(df, tr)
    dva = _subset(df, va)
    dte = _subset(df, te)

    feats = list(cfg.num + cfg.cat)
    yte = _y(dte)

    run_dir = mk(out / "run")
    rows = []

    for name, disp in MODELS:
        _, te_p, auc, bac = _fit_eval(name=name, cfg=cfg, feats=feats, dtr=dtr, dte=dte)
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


def run_tabcv(index: Path, sheet: Path, out: Path, folds: int = 5, seed: int = 7) -> None:
    cfg = TabCfg()
    df = _load(index=index, sheet=sheet, cfg=cfg)
    feats = list(cfg.num + cfg.cat)
    y = _y(df)

    if len(df) < folds:
        raise ValueError(f"need at least {folds} labelled MR1 rows for {folds}-fold CV")

    run_dir = mk(out / "run")
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    imp_rows: list[dict[str, object]] = []

    for fold, (tr_ix, te_ix) in enumerate(split.split(df["subj"], y), start=1):
        dtr = df.iloc[tr_ix].copy()
        dte = df.iloc[te_ix].copy()
        yte = _y(dte)

        for name, disp in MODELS:
            pipe, te_p, auc, bac = _fit_eval(name=name, cfg=cfg, feats=feats, dtr=dtr, dte=dte)
            brier = float(np.mean((te_p - yte) ** 2))
            fold_rows.append(
                {
                    "model": name,
                    "name": disp,
                    "fold": fold,
                    "n_train": int(len(dtr)),
                    "n_test": int(len(dte)),
                    "auc": auc,
                    "bal_acc": bac,
                    "brier": brier,
                }
            )
            if name == "logreg":
                coef_rows.extend(_coef_rows(pipe=pipe, cfg=cfg, fold=fold))
            else:
                imp_rows.extend(_imp_rows(pipe=pipe, cfg=cfg, fold=fold, model=name))

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"])
    fold_df.to_csv(run_dir / "fold_metrics.csv", index=False)

    summ = (
        fold_df.groupby(["model", "name"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            bal_acc_mean=("bal_acc", "mean"),
            bal_acc_std=("bal_acc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values("auc_mean", ascending=False)
    )
    summ.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summ.to_dict(orient="records"), indent=2) + "\n")

    coef_df = pd.DataFrame(coef_rows)
    if not coef_df.empty:
        coef_df.to_csv(run_dir / "logreg_coef_folds.csv", index=False)
        coef_summ = (
            coef_df.groupby("feature", as_index=False)
            .agg(
                coef_mean=("coef", "mean"),
                coef_std=("coef", "std"),
                abs_coef_mean=("abs_coef", "mean"),
                abs_coef_std=("abs_coef", "std"),
            )
            .sort_values("abs_coef_mean", ascending=False)
        )
        coef_summ.to_csv(run_dir / "logreg_coef_summary.csv", index=False)

    imp_df = pd.DataFrame(imp_rows)
    if not imp_df.empty:
        imp_df.to_csv(run_dir / "tree_importance_folds.csv", index=False)
        imp_summ = (
            imp_df.groupby(["model", "feature"], as_index=False)
            .agg(
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
            )
            .sort_values(["model", "importance_mean"], ascending=[True, False])
        )
        imp_summ.to_csv(run_dir / "tree_importance_summary.csv", index=False)
