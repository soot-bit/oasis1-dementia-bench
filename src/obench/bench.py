from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from .cal import run_cal
from .cnn2d import run_cnn2d
from .utils.fp import mk


@dataclass(frozen=True)
class CnnJob:
    name: str
    axis: int
    ch: int
    arch: str
    pool: str
    pick: str
    slices: int
    bs: int
    lr: float
    aug: bool = True


def _preset(name: str) -> list[CnnJob]:
    if name == "best":
        return [
            CnnJob(name="coronal3_tiny_mean", axis=1, ch=3, arch="tiny", pool="mean", pick="topnz", slices=24, bs=8, lr=3e-4),
        ]
    if name == "quick":
        return [
            CnnJob(name="axial1_tiny", axis=2, ch=1, arch="tiny", pool="max", pick="topnz", slices=24, bs=8, lr=1e-3),
            CnnJob(name="coronal3_tiny", axis=1, ch=3, arch="tiny", pool="max", pick="topnz", slices=24, bs=8, lr=1e-3),
        ]
    if name == "gpu":
        return [
            CnnJob(name="axial1_tiny", axis=2, ch=1, arch="tiny", pool="max", pick="topnz", slices=24, bs=8, lr=1e-3),
            CnnJob(name="coronal3_tiny", axis=1, ch=3, arch="tiny", pool="max", pick="topnz", slices=24, bs=8, lr=1e-3),
            CnnJob(name="coronal3_r18", axis=1, ch=3, arch="resnet18", pool="max", pick="topnz", slices=24, bs=4, lr=3e-4),
            CnnJob(name="axial3_r18", axis=2, ch=3, arch="resnet18", pool="max", pick="topnz", slices=24, bs=4, lr=3e-4),
        ]
    raise ValueError(f"unknown preset: {name}")


def run_benchcnn(
    index: Path,
    sheet: Path,
    splits: Path,
    out: Path,
    preset: str,
    epochs: int,
    seed: int,
    limit: int | None = None,
) -> None:
    con = Console()
    jobs = _preset(preset)
    if limit is not None:
        jobs = jobs[: max(0, int(limit))]

    root = mk(out)
    rows: list[dict[str, object]] = []

    t0 = Table(title="benchcnn plan", show_lines=False)
    t0.add_column("name")
    t0.add_column("axis")
    t0.add_column("ch")
    t0.add_column("arch")
    t0.add_column("pool")
    t0.add_column("bs")
    t0.add_column("lr")
    for j in jobs:
        t0.add_row(j.name, str(j.axis), str(j.ch), j.arch, j.pool, str(j.bs), f"{j.lr:g}")
    con.print(t0)

    for j in jobs:
        run_dir = root / j.name
        run_cnn2d(
            index=index,
            sheet=sheet,
            splits=splits,
            out=run_dir,
            seed=seed,
            epochs=epochs,
            bs=j.bs,
            lr=j.lr,
            slices=j.slices,
            pick=j.pick,
            pool=j.pool,
            aug=j.aug,
            axis=j.axis,
            ch=j.ch,
            arch=j.arch,
        )
        run_cal(
            pred=run_dir / "run" / "pred.json",
            sheet=sheet,
            out=run_dir / "cal",
            bins=10,
        )

        met = json.loads((run_dir / "run" / "metrics.json").read_text())
        pred = json.loads((run_dir / "run" / "pred_stats.json").read_text())
        cal = json.loads((run_dir / "cal" / "metrics.json").read_text())
        rows.append(
            {
                "name": j.name,
                "axis": j.axis,
                "ch": j.ch,
                "arch": j.arch,
                "pool": j.pool,
                "pick": j.pick,
                "slices": j.slices,
                "bs": j.bs,
                "lr": j.lr,
                "epochs": epochs,
                "auc": met["auc"],
                "auc_bb_lo": met["bayes"]["auc_bb"]["lo"],
                "auc_bb_hi": met["bayes"]["auc_bb"]["hi"],
                "bal_acc": met["bal_acc"],
                "bal_acc05": met["bal_acc05"],
                "bal_acc_lo": met["bayes"]["cls_thr"]["bal_acc"]["lo"],
                "bal_acc_hi": met["bayes"]["cls_thr"]["bal_acc"]["hi"],
                "thr": met["thr"],
                "p_std": pred["p_std"],
                "brier": cal["brier"],
                "ece": cal["ece"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["auc", "p_std"], ascending=[False, False])
    df.to_csv(root / "summary.csv", index=False)

    t1 = Table(title="benchcnn summary", show_lines=False)
    for col in ["name", "arch", "axis", "ch", "auc", "bal_acc", "p_std", "ece"]:
        t1.add_column(col, justify="right" if col in {"axis", "ch", "auc", "bal_acc", "p_std", "ece"} else "left")
    for _, r in df.iterrows():
        t1.add_row(
            str(r["name"]),
            str(r["arch"]),
            str(r["axis"]),
            str(r["ch"]),
            f"{float(r['auc']):.4f}",
            f"{float(r['bal_acc']):.4f}",
            f"{float(r['p_std']):.6f}",
            f"{float(r['ece']):.4f}",
        )
    con.print(t1)
