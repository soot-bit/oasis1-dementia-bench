from __future__ import annotations

import numpy as np

from obench.utils.img import zscore_brain


def test_zscore_brain_keeps_zeros() -> None:
    a = np.zeros((4, 4, 4), dtype=np.float32)
    a[1:3, 1:3, 1:3] = 2.0
    z = zscore_brain(a)
    assert np.all(z[a == 0] == 0)

