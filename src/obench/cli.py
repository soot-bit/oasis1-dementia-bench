from __future__ import annotations

import argparse
from pathlib import Path

from .index import run_index
from .split import run_split
from .tab import run_tab
from .cnn2d import run_cnn2d
from .eda import run_eda
from .manifest import run_manifest
from .err import run_err_tab


def _p(p: str) -> Path:
    return Path(p).expanduser().resolve()


def main() -> None:
    ap = argparse.ArgumentParser(prog="obench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_i = sub.add_parser("index", help="index extracted OASIS sessions")
    ap_i.add_argument(
        "--root",
        required=True,
        action="append",
        type=_p,
        help="disc folder (repeatable) or parent folder containing discs",
    )
    ap_i.add_argument("--out", required=True, type=_p, help="output CSV path")

    ap_s = sub.add_parser("split", help="make subject-level splits")
    ap_s.add_argument("--index", required=True, type=_p)
    ap_s.add_argument("--sheet", required=True, type=_p, help="oasis_cross-sectional*.xlsx")
    ap_s.add_argument("--out", required=True, type=_p, help="output dir (writes train/val/test.txt)")
    ap_s.add_argument("--seed", type=int, default=7)
    ap_s.add_argument("--test", type=float, default=0.2)
    ap_s.add_argument("--val", type=float, default=0.2)
    ap_s.add_argument("--all-mr", action="store_true", help="include MR2/MR3... (default: MR1 only)")

    ap_t = sub.add_parser("tab", help="tabular baselines")
    ap_t.add_argument("--index", required=True, type=_p)
    ap_t.add_argument("--sheet", required=True, type=_p)
    ap_t.add_argument("--splits", required=True, type=_p)
    ap_t.add_argument("--out", required=True, type=_p)

    ap_c = sub.add_parser("cnn2d", help="2D CNN baseline from processed MRI (run in separate shell)")
    ap_c.add_argument("--index", required=True, type=_p)
    ap_c.add_argument("--sheet", required=True, type=_p)
    ap_c.add_argument("--splits", required=True, type=_p)
    ap_c.add_argument("--out", required=True, type=_p)
    ap_c.add_argument("--seed", type=int, default=7)
    ap_c.add_argument("--epochs", type=int, default=30)
    ap_c.add_argument("--bs", type=int, default=8)
    ap_c.add_argument("--lr", type=float, default=3e-4)

    ap_e = sub.add_parser("eda", help="basic EDA from spreadsheet + index")
    ap_e.add_argument("--index", required=True, type=_p)
    ap_e.add_argument("--sheet", required=True, type=_p)
    ap_e.add_argument("--out", required=True, type=_p)

    ap_m = sub.add_parser("manifest", help="build a merged manifest (index + labels + paths)")
    ap_m.add_argument("--index", required=True, type=_p)
    ap_m.add_argument("--sheet", required=True, type=_p)
    ap_m.add_argument("--out", required=True, type=_p, help="output CSV path")
    ap_m.add_argument("--mr1-only", action="store_true", default=True)
    ap_m.add_argument("--label-only", action="store_true", default=True)

    ap_x = sub.add_parser("errtab", help="tabular error analysis (from tab run outputs)")
    ap_x.add_argument("--errors", required=True, type=_p, help="errors.csv from `obench tab`")
    ap_x.add_argument("--out", required=True, type=_p, help="output dir")

    a = ap.parse_args()

    if a.cmd == "index":
        run_index(roots=a.root, out=a.out)
        return
    if a.cmd == "split":
        run_split(index=a.index, sheet=a.sheet, out=a.out, seed=a.seed, test=a.test, val=a.val, mr1_only=(not a.all_mr))
        return
    if a.cmd == "tab":
        run_tab(index=a.index, sheet=a.sheet, splits=a.splits, out=a.out)
        return
    if a.cmd == "cnn2d":
        run_cnn2d(index=a.index, sheet=a.sheet, splits=a.splits, out=a.out, seed=a.seed, epochs=a.epochs, bs=a.bs, lr=a.lr)
        return
    if a.cmd == "eda":
        run_eda(index=a.index, sheet=a.sheet, out=a.out)
        return
    if a.cmd == "manifest":
        run_manifest(index=a.index, sheet=a.sheet, out=a.out, mr1_only=a.mr1_only, label_only=a.label_only)
        return
    if a.cmd == "errtab":
        run_err_tab(errors=a.errors, out=a.out)
        return

    raise SystemExit(f"unknown cmd: {a.cmd}")
