from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io import read_sheet
from .utils.fp import mk


def run_eda(index: Path, sheet: Path, out: Path) -> None:
    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    df = idx.merge(sh, on="id", how="inner")
    df = df[df["id"].str.endswith("_MR1")].copy()
    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    lab = df[~df["CDR"].isna()].copy()
    lab["y"] = (lab["CDR"] > 0).astype(int)

    p = mk(out)

    summary = {
        "n": int(len(df)),
        "n_subj": int(df["subj"].nunique()) if "subj" in df.columns else int(df["id"].nunique()),
        "n_lab": int(len(lab)),
        "y_counts": lab["y"].value_counts(dropna=False).to_dict(),
        "cdr_counts": lab["CDR"].value_counts(dropna=False).to_dict(),
        "missing": df.isna().mean().sort_values(ascending=False).head(30).to_dict(),
    }
    (p / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # label counts
    plt.figure(figsize=(4, 3))
    sns.countplot(data=lab, x="y")
    plt.tight_layout()
    plt.savefig(p / "y_counts.png", dpi=200)
    plt.close()

    # age distribution by label (if present)
    if "Age" in lab.columns:
        plt.figure(figsize=(5, 4))
        sns.histplot(data=lab, x="Age", hue="y", bins=20, stat="density", common_norm=False, element="step")
        plt.tight_layout()
        plt.savefig(p / "age_hist.png", dpi=200)
        plt.close()

    # morphometric correlation
    cols = [c for c in ["Age", "eTIV", "nWBV", "ASF", "MMSE"] if c in lab.columns]
    for c in cols:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    if len(cols) >= 2:
        c = lab[cols].corr(numeric_only=True)
        plt.figure(figsize=(0.7 * len(cols) + 2, 0.7 * len(cols) + 2))
        sns.heatmap(c, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig(p / "corr.png", dpi=200)
        plt.close()
