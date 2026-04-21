from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils.fp import mk


def run_err_tab(errors: Path, out: Path) -> None:
    df = pd.read_csv(errors)
    p = mk(out)

    # basic slices
    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
    df["CDR"] = pd.to_numeric(df.get("CDR"), errors="coerce")

    # confusion tags
    df["tag"] = "ok"
    m_fp = (df["y"] == 0) & (df["yhat"] == 1)
    m_fn = (df["y"] == 1) & (df["yhat"] == 0)
    df.loc[m_fp, "tag"] = "fp"
    df.loc[m_fn, "tag"] = "fn"

    summ = {
        "n": int(len(df)),
        "tags": df["tag"].value_counts().to_dict(),
        "age_mean_by_tag": df.groupby("tag")["Age"].mean(numeric_only=True).to_dict(),
        "mmse_mean_by_tag": df.groupby("tag")["MMSE"].mean(numeric_only=True).to_dict(),
    }
    (p / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")

    # FP/FN by age
    if df["Age"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.stripplot(data=df, x="tag", y="Age", jitter=0.25, alpha=0.8, order=["fp", "fn", "ok"])
        plt.tight_layout()
        plt.savefig(p / "age_by_tag.png", dpi=200)
        plt.close()

    # FP/FN by MMSE
    if df["MMSE"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.stripplot(data=df, x="tag", y="MMSE", jitter=0.25, alpha=0.8, order=["fp", "fn", "ok"])
        plt.tight_layout()
        plt.savefig(p / "mmse_by_tag.png", dpi=200)
        plt.close()

    # Show a compact markdown page (data-safe: no raw MR images)
    md = [
        "# Error analysis (tabular baseline)",
        "",
        "This page summarizes false positives / false negatives from `errors.csv` produced by `obench tab`.",
        "",
        "## Summary",
        f"- n: {summ['n']}",
        f"- tags: {summ['tags']}",
        "",
        "## Plots",
        "- `age_by_tag.png`: age distribution for FP/FN/OK",
        "- `mmse_by_tag.png`: MMSE distribution for FP/FN/OK (if MMSE available)",
        "",
    ]
    (p / "README.md").write_text("\n".join(md))

