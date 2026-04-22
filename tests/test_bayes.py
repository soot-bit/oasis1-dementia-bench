from __future__ import annotations

import numpy as np

from obench.bayes import auc_bb_ci
from obench.bayes import cls_ci


def test_cls_ci_perfect_sep() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    out = cls_ci(y=y, p=p, draws=4000, seed=7)
    assert out["counts"] == {"tp": 2, "fn": 0, "tn": 2, "fp": 0}
    assert out["bal_acc"]["point"] == 1.0
    assert 0.0 <= out["bal_acc"]["lo"] <= out["bal_acc"]["mid"] <= out["bal_acc"]["hi"] <= 1.0
    assert out["sens"]["mid"] > 0.7
    assert out["spec"]["mid"] > 0.7


def test_auc_bb_ci_range() -> None:
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    out = auc_bb_ci(y=y, p=p, draws=2000, seed=7)
    assert out["point"] == 1.0
    assert 0.0 <= out["lo"] <= out["mid"] <= out["hi"] <= 1.0
    assert out["mid"] > 0.8
