from __future__ import annotations

import numpy as np

from obench.cal import _ece


def test_ece_zero_for_perfect_calibration() -> None:
    # perfectly calibrated: p matches y exactly at 0/1
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    assert _ece(y, p, bins=4) == 0.0


def test_ece_positive_when_wrong_confident() -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    assert _ece(y, p, bins=4) > 0.9

