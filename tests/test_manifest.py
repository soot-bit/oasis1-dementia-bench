from __future__ import annotations

from pathlib import Path

import pandas as pd

from obench.manifest import run_manifest


def test_manifest_has_label_and_img(tmp_path: Path) -> None:
    idx = pd.DataFrame(
        [
            {"id": "OAS1_0001_MR1", "subj": "OAS1_0001", "t88_mask": "x.img"},
            {"id": "OAS1_0002_MR1", "subj": "OAS1_0002", "t88_mask": "y.img"},
        ]
    )
    idx_p = tmp_path / "index.csv"
    idx.to_csv(idx_p, index=False)

    sh = pd.DataFrame(
        [
            {"ID": "OAS1_0001_MR1", "CDR": 0.0},
            {"ID": "OAS1_0002_MR1", "CDR": 0.5},
        ]
    )
    sh_p = tmp_path / "sheet.xlsx"
    sh.to_excel(sh_p, index=False)

    out = tmp_path / "manifest.csv"
    run_manifest(index=idx_p, sheet=sh_p, out=out, mr1_only=True, label_only=True)

    df = pd.read_csv(out)
    assert "y" in df.columns
    assert "img" in df.columns
    assert set(df["y"].tolist()) == {0, 1}

