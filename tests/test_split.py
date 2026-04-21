from __future__ import annotations

from pathlib import Path

import pandas as pd

from obench.split import run_split


def test_split_mr1_only_and_label_filter(tmp_path: Path) -> None:
    idx = pd.DataFrame(
        [
            {"id": "OAS1_0001_MR1", "subj": "OAS1_0001"},
            {"id": "OAS1_0001_MR2", "subj": "OAS1_0001"},
            {"id": "OAS1_0002_MR1", "subj": "OAS1_0002"},
        ]
    )
    idx_p = tmp_path / "index.csv"
    idx.to_csv(idx_p, index=False)

    sh = pd.DataFrame(
        [
            {"ID": "OAS1_0001_MR1", "CDR": 0.0},
            {"ID": "OAS1_0001_MR2", "CDR": 0.5},
            {"ID": "OAS1_0002_MR1", "CDR": None},  # should be filtered out
        ]
    )
    sh_p = tmp_path / "sheet.xlsx"
    sh.to_excel(sh_p, index=False)

    out = tmp_path / "splits"
    run_split(index=idx_p, sheet=sh_p, out=out, seed=7, test=0.5, val=0.5, mr1_only=True)

    tr = (out / "train.txt").read_text().splitlines()
    va = (out / "val.txt").read_text().splitlines()
    te = (out / "test.txt").read_text().splitlines()
    all_ids = set([*tr, *va, *te])

    assert all(i.endswith("_MR1") for i in all_ids)
    assert "OAS1_0002_MR1" not in all_ids  # missing label filtered

