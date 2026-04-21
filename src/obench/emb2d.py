from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cnn2d import Cfg, ResNet18, TinyNet, Vol2D, _collate, _emb_pool
from .io import read_lines, read_sheet
from .utils.fp import mk


def run_emb2d(
    index: Path,
    sheet: Path,
    splits: Path,
    weights: Path,
    out: Path,
    pool: str = "max",
    slices: int = 24,
    pick: str = "topnz",
    axis: int = 2,
    ch: int = 1,
    arch: str = "tiny",
) -> None:
    idx = pd.read_csv(index)
    sh = read_sheet(sheet).rename(columns={"ID": "id"})
    df = idx.merge(sh, on="id", how="inner")
    df = df[df["id"].str.endswith("_MR1")].copy()
    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df[~df["CDR"].isna()].copy()
    df["y"] = (df["CDR"] > 0).astype(int)
    df = df[df["t88_mask"].astype(str).str.len() > 0].copy()

    ids = sorted(
        set(read_lines(splits / "train.txt") + read_lines(splits / "val.txt") + read_lines(splits / "test.txt"))
    )
    df = df[df["id"].isin(ids)].copy()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch == "resnet18":
        net = ResNet18(in_ch=int(ch)).to(dev)
    else:
        net = TinyNet(in_ch=int(ch)).to(dev)
    sd = torch.load(weights, map_location="cpu")
    net.load_state_dict(sd)
    net.eval()

    cfg = Cfg(pool=pool, slices=int(slices), pick=str(pick), axis=int(axis), ch=int(ch), aug=False)
    dl = DataLoader(Vol2D(df, cfg), batch_size=4, shuffle=False, num_workers=0, collate_fn=_collate)

    rows = []
    with torch.no_grad():
        for xb, yb, idb in tqdm(dl, desc="emb2d"):
            xb = xb.to(dev)
            e = _emb_pool(net, xb, pool=cfg.pool).cpu().numpy()  # B,64
            for i, sid in enumerate(idb):
                rows.append({"id": sid, "y": int(yb[i].item()), **{f"e{i2}": float(e[i, i2]) for i2 in range(e.shape[1])}})

    out_p = mk(out.parent)
    pd.DataFrame(rows).sort_values("id").to_csv(out, index=False)
