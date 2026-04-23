from __future__ import annotations

import numpy as np
from pathlib import Path

from obench.utils.img import fix_path
from obench.utils.img import zscore_brain


def test_zscore_brain_keeps_zeros() -> None:
    a = np.zeros((4, 4, 4), dtype=np.float32)
    a[1:3, 1:3, 1:3] = 2.0
    z = zscore_brain(a)
    assert np.all(z[a == 0] == 0)


def test_fix_path_swaps_legacy_prefix(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target = root / "data" / "oasis1" / "disc1" / "x.img"
    target.parent.mkdir(parents=True)
    target.write_text("x")
    old = root / "data" / "interim" / "oasis1" / "disc1" / "x.img"
    got = fix_path(old)
    assert got == target
