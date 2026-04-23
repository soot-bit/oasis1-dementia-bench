from __future__ import annotations

import argparse
from pathlib import Path

from .index import run_index
from .split import run_split
from .tab import run_tab
from .tab import run_tabcv
from .cnn2d import run_cnn2d
from .eda import run_eda
from .manifest import run_manifest
from .err import run_err_tab
from .cal import run_cal
from .emb2d import run_emb2d
from .fuse import run_fuse
from .bench import run_benchcnn


def _p(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _add_cnn_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--index", required=True, type=_p)
    ap.add_argument("--sheet", required=True, type=_p)
    ap.add_argument("--splits", required=True, type=_p)
    ap.add_argument("--out", required=True, type=_p)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--slices", type=int, default=24)
    ap.add_argument("--pick", choices=["topnz", "lin", "mid"], default="topnz")
    ap.add_argument("--pool", choices=["max", "mean", "lse", "attn"], default="max")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--axis", type=int, choices=[0, 1, 2], default=2, help="slice axis (2=axial-ish default)")
    ap.add_argument("--ch", type=int, choices=[1, 3], default=1, help="2.5D channels per slice")
    ap.add_argument("--arch", choices=["tiny", "resnet18"], default="tiny")


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

    ap_tc = sub.add_parser("tabcv", help="subject-level cross-validation for classical tabular baselines")
    ap_tc.add_argument("--index", required=True, type=_p)
    ap_tc.add_argument("--sheet", required=True, type=_p)
    ap_tc.add_argument("--out", required=True, type=_p)
    ap_tc.add_argument("--folds", type=int, default=5)
    ap_tc.add_argument("--seed", type=int, default=7)

    ap_c = sub.add_parser("cnn2d", help="2D CNN baseline from processed MRI (manual trainer)")
    _add_cnn_args(ap_c)

    ap_l = sub.add_parser("cnnlit", help="2D CNN baseline with PyTorch Lightning")
    _add_cnn_args(ap_l)
    ap_l.add_argument("--patience", type=int, default=8, help="early-stopping patience on validation AUC")
    ap_l.add_argument("--workers", type=int, default=0, help="data-loader workers")
    ap_l.add_argument("--precision", default="auto", help="Lightning precision: auto, 32-true, 16-mixed, bf16-mixed")

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

    ap_xt = sub.add_parser("xaitab", help="tabular explainability: permutation, coefficients, local contributions")
    ap_xt.add_argument("--index", required=True, type=_p)
    ap_xt.add_argument("--sheet", required=True, type=_p)
    ap_xt.add_argument("--splits", required=True, type=_p)
    ap_xt.add_argument("--out", required=True, type=_p)
    ap_xt.add_argument("--repeats", type=int, default=100)
    ap_xt.add_argument("--seed", type=int, default=7)
    ap_xt.add_argument("--top", type=int, default=12)

    ap_xc = sub.add_parser("xaicnn", help="CNN explainability: Grad-CAM panels and sanity checks")
    ap_xc.add_argument("--index", required=True, type=_p)
    ap_xc.add_argument("--sheet", required=True, type=_p)
    ap_xc.add_argument("--splits", required=True, type=_p)
    ap_xc.add_argument("--run", required=True, type=_p, help="CNN run dir containing model.pt, train.json, metrics.json")
    ap_xc.add_argument("--out", required=True, type=_p)
    ap_xc.add_argument("--n", type=int, default=2, help="examples per TP/FP/TN/FN group")
    ap_xc.add_argument("--split", choices=["train", "val", "test"], default="test")

    ap_k = sub.add_parser("cal", help="calibration + uncertainty from predictions (json/csv)")
    ap_k.add_argument("--pred", required=True, type=_p, help="predictions file: JSON(id->p) or CSV(id,p)")
    ap_k.add_argument("--sheet", required=True, type=_p, help="oasis_cross-sectional*.xlsx (provides CDR labels)")
    ap_k.add_argument("--out", required=True, type=_p, help="output dir")
    ap_k.add_argument("--bins", type=int, default=10)

    ap_b = sub.add_parser("emb2d", help="extract 2D CNN embeddings (for fusion)")
    ap_b.add_argument("--index", required=True, type=_p)
    ap_b.add_argument("--sheet", required=True, type=_p)
    ap_b.add_argument("--splits", required=True, type=_p)
    ap_b.add_argument("--weights", required=True, type=_p, help="model.pt from `obench cnn2d`")
    ap_b.add_argument("--out", required=True, type=_p, help="output CSV (id,y,e0..e63)")
    ap_b.add_argument("--pool", choices=["max", "mean", "attn"], default="max", help="embedding pooling (match cnn2d if possible)")
    ap_b.add_argument("--slices", type=int, default=24)
    ap_b.add_argument("--pick", choices=["topnz", "lin", "mid"], default="topnz")
    ap_b.add_argument("--axis", type=int, choices=[0, 1, 2], default=2)
    ap_b.add_argument("--ch", type=int, choices=[1, 3], default=1)
    ap_b.add_argument("--arch", choices=["tiny", "resnet18"], default="tiny")

    ap_f = sub.add_parser("fuse", help="fusion baseline (tabular + 2D CNN embedding)")
    ap_f.add_argument("--index", required=True, type=_p)
    ap_f.add_argument("--sheet", required=True, type=_p)
    ap_f.add_argument("--emb", required=True, type=_p, help="embedding CSV from `obench emb2d`")
    ap_f.add_argument("--splits", required=True, type=_p)
    ap_f.add_argument("--out", required=True, type=_p)
    ap_f.add_argument("--model", choices=["mlp", "logreg"], default="mlp")

    ap_q = sub.add_parser("benchcnn", help="run a CNN benchmark matrix and collect summary")
    ap_q.add_argument("--index", required=True, type=_p)
    ap_q.add_argument("--sheet", required=True, type=_p)
    ap_q.add_argument("--splits", required=True, type=_p)
    ap_q.add_argument("--out", required=True, type=_p)
    ap_q.add_argument("--preset", choices=["best", "quick", "gpu"], default="gpu")
    ap_q.add_argument("--epochs", type=int, default=20)
    ap_q.add_argument("--seed", type=int, default=7)
    ap_q.add_argument("--limit", type=int)

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
    if a.cmd == "tabcv":
        run_tabcv(index=a.index, sheet=a.sheet, out=a.out, folds=a.folds, seed=a.seed)
        return
    if a.cmd == "cnn2d":
        run_cnn2d(
            index=a.index,
            sheet=a.sheet,
            splits=a.splits,
            out=a.out,
            seed=a.seed,
            epochs=a.epochs,
            bs=a.bs,
            lr=a.lr,
            slices=a.slices,
            pick=a.pick,
            pool=a.pool,
            aug=(not a.no_aug),
            axis=a.axis,
            ch=a.ch,
            arch=a.arch,
        )
        return
    if a.cmd == "cnnlit":
        from .cnnlit import run_cnnlit

        run_cnnlit(
            index=a.index,
            sheet=a.sheet,
            splits=a.splits,
            out=a.out,
            seed=a.seed,
            epochs=a.epochs,
            bs=a.bs,
            lr=a.lr,
            slices=a.slices,
            pick=a.pick,
            pool=a.pool,
            aug=(not a.no_aug),
            axis=a.axis,
            ch=a.ch,
            arch=a.arch,
            patience=a.patience,
            workers=a.workers,
            precision=a.precision,
        )
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
    if a.cmd == "xaitab":
        from .xai import run_xaitab

        run_xaitab(index=a.index, sheet=a.sheet, splits=a.splits, out=a.out, repeats=a.repeats, seed=a.seed, top=a.top)
        return
    if a.cmd == "xaicnn":
        from .xai import run_xaicnn

        run_xaicnn(index=a.index, sheet=a.sheet, splits=a.splits, run=a.run, out=a.out, n=a.n, split=a.split)
        return
    if a.cmd == "cal":
        run_cal(pred=a.pred, sheet=a.sheet, out=a.out, bins=a.bins)
        return
    if a.cmd == "emb2d":
        run_emb2d(
            index=a.index,
            sheet=a.sheet,
            splits=a.splits,
            weights=a.weights,
            out=a.out,
            pool=a.pool,
            slices=a.slices,
            pick=a.pick,
            axis=a.axis,
            ch=a.ch,
            arch=a.arch,
        )
        return
    if a.cmd == "fuse":
        run_fuse(index=a.index, sheet=a.sheet, emb=a.emb, splits=a.splits, out=a.out, model=a.model)
        return
    if a.cmd == "benchcnn":
        run_benchcnn(index=a.index, sheet=a.sheet, splits=a.splits, out=a.out, preset=a.preset, epochs=a.epochs, seed=a.seed, limit=a.limit)
        return

    raise SystemExit(f"unknown cmd: {a.cmd}")
