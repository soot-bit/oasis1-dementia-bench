from __future__ import annotations

from pathlib import Path

import pandas as pd

from obench.index import run_index


def _mk_sess(root: Path, sid: str) -> None:
    p = root / sid
    (p / "RAW").mkdir(parents=True)
    (p / "PROCESSED" / "MPRAGE" / "T88_111").mkdir(parents=True)
    (p / "PROCESSED" / "MPRAGE" / "SUBJ_111").mkdir(parents=True)
    (p / "FSL_SEG").mkdir(parents=True)
    (p / f"{sid}.xml").write_text("<x/>")
    (p / f"{sid}.txt").write_text("x")
    (p / "PROCESSED" / "MPRAGE" / "T88_111" / f"{sid}_x_t88_masked_gfc.img").write_text("x")


def test_index_dedup(tmp_path: Path) -> None:
    disc1 = tmp_path / "disc1"
    disc2 = tmp_path / "disc2"
    disc1.mkdir()
    disc2.mkdir()

    _mk_sess(disc1, "OAS1_0001_MR1")
    _mk_sess(disc2, "OAS1_0001_MR1")  # duplicate id across roots
    _mk_sess(disc2, "OAS1_0002_MR1")

    out = tmp_path / "index.csv"
    run_index([disc1, disc2], out)

    df = pd.read_csv(out)
    assert sorted(df["id"].tolist()) == ["OAS1_0001_MR1", "OAS1_0002_MR1"]

