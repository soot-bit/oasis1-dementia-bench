from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_sheet(p: Path) -> pd.DataFrame:
    df = pd.read_excel(p, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_lines(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]

