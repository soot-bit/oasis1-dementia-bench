from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_analyze(img: Path) -> np.ndarray:
    if img.suffix != ".img":
        raise ValueError(f"expected .img, got {img}")
    x = nib.load(str(img))
    a = np.asanyarray(x.dataobj).astype(np.float32)
    a = np.squeeze(a)
    return a


def zscore_brain(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = a != 0
    if not np.any(m):
        return a
    v = a[m]
    mu = float(v.mean())
    sd = float(v.std())
    return (a - mu) / (sd + eps)
