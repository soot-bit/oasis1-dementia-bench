from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

