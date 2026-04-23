from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import read_sheet
from .utils.img import fix_path
from .utils.fp import mk


def run_manifest(index: Path, sheet: Path, out: Path, mr1_only: bool, label_only: bool) -> None:
    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})

    df = idx.merge(sh, on="id", how="left")
    if mr1_only:
        df = df[df["id"].astype(str).str.endswith("_MR1")].copy()

    df["CDR"] = pd.to_numeric(df.get("CDR"), errors="coerce")
    df["y"] = (df["CDR"] > 0).astype("Int64")
    if label_only:
        df = df[~df["CDR"].isna()].copy()

    path_cols = ["img", "t88_mask", "t88_gfc", "subj111", "fseg", "raw", "proc", "seg", "xml", "txt"]
    for col in [c for c in path_cols if c in df.columns]:
        df[col] = df[col].fillna("").astype(str)
        df[col] = df[col].map(lambda x: str(fix_path(Path(x))) if x else "")

    # canonical image path used by image baselines
    df["img"] = df["t88_mask"].astype(str)

    keep = [
        "id",
        "subj",
        "y",
        "CDR",
        "Age",
        "M/F",
        "Hand",
        "Educ",
        "SES",
        "MMSE",
        "eTIV",
        "nWBV",
        "ASF",
        "img",
        "t88_mask",
        "t88_gfc",
        "subj111",
        "fseg",
        "raw",
        "proc",
        "seg",
        "xml",
        "txt",
    ]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()

    mk(out.parent)
    df.to_csv(out, index=False)
