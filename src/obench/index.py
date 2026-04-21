from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils.fp import mk


@dataclass(frozen=True)
class Sess:
    id: str
    subj: str
    root: Path
    raw: Path
    proc: Path
    seg: Path
    xml: Path
    txt: Path
    t88_mask: Path | None
    t88_gfc: Path | None
    subj111: Path | None
    fseg: Path | None


def _is_sess_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("OAS1_") and "_MR" in p.name


def _iter_sess_roots(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)

    xs = [p for p in sorted(root.iterdir()) if _is_sess_dir(p)]
    if xs:
        return xs

    # allow passing a parent folder containing disc folders
    discs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("disc")]
    out: list[Path] = []
    for d in discs:
        out.extend([p for p in sorted(d.iterdir()) if _is_sess_dir(p)])
    return out


def _find_one(d: Path, pat: str) -> Path | None:
    xs = sorted(d.glob(pat))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    return xs[0]


def _sess(p: Path) -> Sess:
    sid = p.name
    subj = sid.split("_MR")[0]
    raw = p / "RAW"
    proc = p / "PROCESSED"
    seg = p / "FSL_SEG"

    xml = p / f"{sid}.xml"
    txt = p / f"{sid}.txt"

    t88 = proc / "MPRAGE" / "T88_111"
    s111 = proc / "MPRAGE" / "SUBJ_111"

    t88_mask = _find_one(t88, f"{sid}_*_t88_masked_gfc.img")
    t88_gfc = _find_one(t88, f"{sid}_*_t88_gfc.img")
    subj111 = _find_one(s111, f"{sid}_*_sbj_111.img")
    fseg = _find_one(seg, f"{sid}_*_t88_masked_gfc_fseg.img")

    return Sess(
        id=sid,
        subj=subj,
        root=p,
        raw=raw if raw.exists() else raw,
        proc=proc if proc.exists() else proc,
        seg=seg if seg.exists() else seg,
        xml=xml,
        txt=txt,
        t88_mask=t88_mask,
        t88_gfc=t88_gfc,
        subj111=subj111,
        fseg=fseg,
    )


def run_index(roots: list[Path], out: Path) -> None:
    sess: list[Sess] = []
    for r in roots:
        for p in _iter_sess_roots(r):
            sess.append(_sess(p))

    df = pd.DataFrame(
        [
            {
                "id": s.id,
                "subj": s.subj,
                "root": str(s.root),
                "raw": str(s.raw),
                "proc": str(s.proc),
                "seg": str(s.seg),
                "xml": str(s.xml),
                "txt": str(s.txt),
                "t88_mask": str(s.t88_mask) if s.t88_mask else "",
                "t88_gfc": str(s.t88_gfc) if s.t88_gfc else "",
                "subj111": str(s.subj111) if s.subj111 else "",
                "fseg": str(s.fseg) if s.fseg else "",
            }
            for s in sess
        ]
    )

    if not df.empty:
        df = df.drop_duplicates(subset=["id"], keep="first").sort_values("id")

    mk(out.parent)
    df.to_csv(out, index=False)
