from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_sheet
from .utils.fp import mk


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


def _label(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["CDR"] = pd.to_numeric(x["CDR"], errors="coerce")
    x = x[~x["CDR"].isna()].copy()
    x["y"] = (x["CDR"] > 0).astype(int)
    return x


def _mr1_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["id"].str.endswith("_MR1")].copy()


def _subj_frame(df: pd.DataFrame) -> pd.DataFrame:
    # one row per subject (OAS1_xxxx)
    g = df.groupby("subj", as_index=False)
    out = g.agg({"y": "max"})
    return out


def _split(subj: pd.DataFrame, seed: int, test: float, val: float) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    a = subj.copy()

    # stratified shuffle on subjects
    ids0 = a[a["y"] == 0]["subj"].to_list()
    ids1 = a[a["y"] == 1]["subj"].to_list()
    rng.shuffle(ids0)
    rng.shuffle(ids1)

    def take(xs: list[str], frac: float) -> tuple[list[str], list[str]]:
        if not xs:
            return [], []
        n = int(round(len(xs) * frac))
        n = max(1, min(n, len(xs) - 1)) if len(xs) >= 2 else 1
        return xs[:n], xs[n:]

    test0, rem0 = take(ids0, test)
    test1, rem1 = take(ids1, test)
    test_subj = test0 + test1

    # val is fraction of remaining
    val_frac = val / max(1e-9, (1.0 - test))
    val0, tr0 = take(rem0, val_frac)
    val1, tr1 = take(rem1, val_frac)
    val_subj = val0 + val1
    tr_subj = tr0 + tr1

    rng.shuffle(test_subj)
    rng.shuffle(val_subj)
    rng.shuffle(tr_subj)
    return tr_subj, val_subj, test_subj


def run_split(index: Path, sheet: Path, out: Path, seed: int, test: float, val: float, mr1_only: bool) -> None:
    idx = pd.read_csv(index)
    x = read_sheet(sheet)

    # join on session ID
    x = x.rename(columns={"ID": "id"})
    df = idx.merge(x, on="id", how="inner")
    df = _label(df)
    if mr1_only:
        df = _mr1_only(df)

    subj = _subj_frame(df)
    tr_subj, va_subj, te_subj = _split(subj=subj, seed=seed, test=test, val=val)

    def sess_ids(subjs: list[str]) -> list[str]:
        return sorted(df[df["subj"].isin(subjs)]["id"].unique().tolist())

    s = Split(train=sess_ids(tr_subj), val=sess_ids(va_subj), test=sess_ids(te_subj))

    mk(out)
    (out / "train.txt").write_text("\n".join(s.train) + "\n")
    (out / "val.txt").write_text("\n".join(s.val) + "\n")
    (out / "test.txt").write_text("\n".join(s.test) + "\n")
