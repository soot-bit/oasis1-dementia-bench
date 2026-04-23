from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def fix_path(p: Path) -> Path:
    if p.exists():
        return p
    s = p.as_posix()
    old = "/data/interim/oasis1/"
    new = "/data/oasis1/"
    if old in s:
        alt = Path(s.replace(old, new, 1))
        if alt.exists():
            return alt
    old_rel = "data/interim/oasis1/"
    new_rel = "data/oasis1/"
    if s.startswith(old_rel):
        alt = Path(new_rel + s[len(old_rel):])
        if alt.exists():
            return alt
    return p


def fix_img(img: Path) -> Path:
    return fix_path(img)


def load_analyze(img: Path) -> np.ndarray:
    img = fix_img(img)
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
    out = np.zeros_like(a, dtype=np.float32)
    out[m] = (a[m] - mu) / (sd + eps)
    return out
