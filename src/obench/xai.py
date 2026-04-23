from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from sklearn.inspection import permutation_importance

from .cnn2d import Cfg, ResNet18, TinyNet, Vol2D, _collate, _eval, _step_pool
from .io import read_lines, read_sheet
from .tab import MODELS, TabCfg, _feat_names, _load, _pipe, _subset, _y
from .utils.fp import mk

con = Console()


def _bar(df: pd.DataFrame, x: str, y: str, out: Path, title: str, color: str = "#2a6fdb") -> None:
    plot = df.copy().sort_values(x, ascending=True)
    plt.figure(figsize=(7.0, max(3.2, 0.34 * len(plot))))
    plt.barh(plot[y].astype(str), plot[x].astype(float), color=color, alpha=0.9)
    plt.xlabel(x)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def _md_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    view = df[cols].head(n).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda val: f"{val:.4f}")
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in cols) + " |" for _, row in view.iterrows()]
    return "\n".join([head, sep, *rows])


def run_xaitab(index: Path, sheet: Path, splits: Path, out: Path, repeats: int, seed: int, top: int) -> None:
    cfg = TabCfg()
    df = _load(index=index, sheet=sheet, cfg=cfg)
    train_ids = read_lines(splits / "train.txt")
    test_ids = read_lines(splits / "test.txt")
    train = _subset(df, train_ids)
    test = _subset(df, test_ids)
    feats = list(cfg.num + cfg.cat)
    test_y = _y(test)

    run = mk(out)
    perm_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []

    for model, name in MODELS:
        pipe = _pipe(cfg, name=model)
        pipe.fit(train[feats], _y(train))
        perm = permutation_importance(
            pipe,
            test[feats],
            test_y,
            scoring="roc_auc",
            n_repeats=int(repeats),
            random_state=int(seed),
        )
        for feat, mean, std in zip(feats, perm.importances_mean, perm.importances_std, strict=True):
            perm_rows.append(
                {
                    "model": model,
                    "name": name,
                    "feature": feat,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )

        if model != "logreg":
            continue

        feat_names = _feat_names(pipe, cfg)
        clf = pipe.named_steps["clf"]
        coef = clf.coef_.ravel()
        for feat, val in zip(feat_names, coef, strict=True):
            coef_rows.append({"feature": feat, "coef": float(val), "abs_coef": float(abs(val))})

        xt = pipe.named_steps["pre"].transform(test[feats])
        contrib = xt * coef
        prob = pipe.predict_proba(test[feats])[:, 1]
        for row_ix, row in enumerate(test.reset_index(drop=True).itertuples(index=False)):
            order = np.argsort(-np.abs(contrib[row_ix]))[: int(top)]
            for rank, feat_ix in enumerate(order, start=1):
                local_rows.append(
                    {
                        "id": row.id,
                        "subj": row.subj,
                        "y": int(test_y[row_ix]),
                        "p": float(prob[row_ix]),
                        "rank": int(rank),
                        "feature": feat_names[int(feat_ix)],
                        "contribution": float(contrib[row_ix, feat_ix]),
                        "abs_contribution": float(abs(contrib[row_ix, feat_ix])),
                    }
                )

    perm_df = pd.DataFrame(perm_rows).sort_values(["model", "importance_mean"], ascending=[True, False])
    coef_df = pd.DataFrame(coef_rows).sort_values("abs_coef", ascending=False)
    local_df = pd.DataFrame(local_rows).sort_values(["id", "rank"])

    perm_df.to_csv(run / "perm.csv", index=False)
    coef_df.to_csv(run / "logreg_coef.csv", index=False)
    local_df.to_csv(run / "logreg_local.csv", index=False)

    log_perm = perm_df[perm_df["model"] == "logreg"].head(int(top))
    if not log_perm.empty:
        _bar(log_perm, "importance_mean", "feature", run / "logreg_perm.png", "Logistic regression permutation importance")
    if not coef_df.empty:
        signed = coef_df.head(int(top)).sort_values("coef", ascending=True)
        colors = ["#b23b3b" if val < 0 else "#2a6fdb" for val in signed["coef"]]
        plt.figure(figsize=(7.0, max(3.2, 0.34 * len(signed))))
        plt.barh(signed["feature"], signed["coef"], color=colors, alpha=0.9)
        plt.axvline(0, color="#111", linewidth=1)
        plt.xlabel("standardised coefficient")
        plt.title("Logistic regression coefficients")
        plt.tight_layout()
        plt.savefig(run / "logreg_coef.png", dpi=200)
        plt.close()

    top_perm = perm_df.groupby("feature", as_index=False)["importance_mean"].mean().sort_values("importance_mean", ascending=False)
    summary = [
        "# Tabular XAI",
        "",
        "Permutation importance is computed on the held-out test split with ROC-AUC scoring.",
        "Logistic coefficients are based on the fitted preprocessing pipeline, so numeric coefficients are standardised.",
        "",
        "## Top Permutation Features",
        "",
        _md_table(top_perm, ["feature", "importance_mean"], n=int(top)),
        "",
        "## Logistic Coefficients",
        "",
        _md_table(coef_df, ["feature", "coef", "abs_coef"], n=int(top)) if not coef_df.empty else "No coefficients written.",
        "",
        "## Local Explanations",
        "",
        "Per-subject linear contributions are written to `logreg_local.csv`.",
    ]
    (run / "summary.md").write_text("\n".join(summary) + "\n")

    tab = Table(title="xaitab", show_lines=False)
    tab.add_column("artifact")
    tab.add_column("path")
    for name in ["perm.csv", "logreg_coef.csv", "logreg_local.csv", "summary.md"]:
        tab.add_row(name, str(run / name))
    con.print(tab)


def _cnn_df(index: Path, sheet: Path, splits: Path, split: str) -> pd.DataFrame:
    idx = pd.read_csv(index)
    meta = read_sheet(sheet).rename(columns={"ID": "id"})
    df = idx.merge(meta, on="id", how="inner")
    df = df[df["id"].astype(str).str.endswith("_MR1")].copy()
    df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
    df = df[~df["CDR"].isna()].copy()
    df["y"] = (df["CDR"] > 0).astype(int)
    df = df[df["t88_mask"].astype(str).str.len() > 0].copy()
    ids = read_lines(splits / f"{split}.txt")
    return df[df["id"].isin(ids)].copy()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def _net(arch: str, ch: int) -> torch.nn.Module:
    return ResNet18(in_ch=int(ch)) if arch == "resnet18" else TinyNet(in_ch=int(ch))


def _target(net: torch.nn.Module, arch: str) -> torch.nn.Module:
    return net.l4[-1] if arch == "resnet18" else net.c3


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(arr)) if arr.size else 0.0
    hi = float(np.nanmax(arr)) if arr.size else 0.0
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _gradcam(net: torch.nn.Module, x: torch.Tensor, pool: str, arch: str, dev: torch.device) -> tuple[float, np.ndarray]:
    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    layer = _target(net, arch)

    def fwd(_, __, out):
        acts.append(out)

    def bwd(_, __, grad_out):
        grads.append(grad_out[0])

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)
    try:
        net.zero_grad(set_to_none=True)
        xb = x.unsqueeze(0).to(dev)
        logit = _step_pool(net, xb, pool=pool)
        score = float(torch.sigmoid(logit).detach().cpu().item())
        logit.sum().backward()
        act = acts[-1].detach()
        grad = grads[-1].detach()
        weight = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weight * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze(1).detach().cpu().numpy()
        cam_np = np.stack([_norm(one) for one in cam_np], axis=0)
        return score, cam_np
    finally:
        h1.remove()
        h2.remove()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _pick_cases(df: pd.DataFrame, n: int) -> pd.DataFrame:
    parts = []
    rules = [
        ("fp", (df["y"] == 0) & (df["yhat"] == 1), False),
        ("fn", (df["y"] == 1) & (df["yhat"] == 0), True),
        ("tp", (df["y"] == 1) & (df["yhat"] == 1), False),
        ("tn", (df["y"] == 0) & (df["yhat"] == 0), True),
    ]
    for kind, mask, asc in rules:
        part = df[mask].sort_values("p", ascending=asc).head(int(n)).copy()
        part["kind"] = kind
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()


def _panel(row: pd.Series, x: torch.Tensor, cam: np.ndarray, out: Path) -> dict[str, object]:
    center = x.shape[1] // 2
    base = x[:, center].detach().cpu().numpy()
    energy = cam.reshape(cam.shape[0], -1).sum(axis=1)
    order = np.argsort(-energy)[:3]
    if float(energy[order].sum()) == 0.0:
        brain = np.count_nonzero(base != 0, axis=(1, 2))
        order = np.argsort(-brain)[:3]

    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.5))
    axes = np.atleast_1d(axes)
    frac = []
    for ax, sl_ix in zip(axes, order, strict=True):
        img = base[int(sl_ix)]
        heat = cam[int(sl_ix)]
        vals = img[img != 0]
        vmin, vmax = (np.percentile(vals, [1, 99]) if len(vals) else (-1, 1))
        mask = np.abs(img) > 0
        total = float(heat.sum())
        frac.append(float(heat[mask].sum() / total) if total > 0 else float("nan"))
        ax.imshow(np.rot90(img), cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(np.rot90(heat), cmap="magma", alpha=0.42, vmin=0, vmax=1)
        ax.set_title(f"slice {int(sl_ix)}")
        ax.axis("off")
    fig.suptitle(f"{row['id']} | {row['kind']} | y={int(row['y'])} p={float(row['p']):.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close()
    return {
        "slices": ";".join(str(int(val)) for val in order),
        "brain_cam_frac": float(np.nanmean(frac)) if frac else float("nan"),
    }


def run_xaicnn(index: Path, sheet: Path, splits: Path, run: Path, out: Path, n: int, split: str) -> None:
    train = _read_json(run / "train.json")
    metrics = _read_json(run / "metrics.json")
    pred = _read_json(run / "pred.json")

    arch = str(train.get("arch", "tiny"))
    ch = int(train.get("ch", 1))
    axis = int(train.get("axis", 2))
    slices = int(train.get("slices", 24))
    pick = str(train.get("pick", "topnz"))
    pool = str(train.get("pool", "max"))
    thr = float(metrics.get("thr", train.get("thr", 0.5)))

    data = _cnn_df(index=index, sheet=sheet, splits=splits, split=split)
    if pred:
        data["p"] = data["id"].map({str(k): float(v) for k, v in pred.items()})
    if "p" not in data.columns or data["p"].isna().all():
        cfg = Cfg(slices=slices, pick=pick, aug=False, pool=pool, axis=axis, ch=ch)
        dl = torch.utils.data.DataLoader(Vol2D(data, cfg), batch_size=8, shuffle=False, num_workers=0, collate_fn=_collate)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmp = _net(arch, ch).to(dev)
        tmp.load_state_dict(torch.load(run / "model.pt", map_location=dev))
        _, _, eval_pred = _eval(tmp, dl, dev, pool=pool)
        data["p"] = data["id"].map(eval_pred)
    data = data[~data["p"].isna()].copy()
    data["yhat"] = (data["p"] >= thr).astype(int)

    cfg = Cfg(slices=slices, pick=pick, aug=False, pool=pool, axis=axis, ch=ch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _net(arch, ch).to(dev)
    net.load_state_dict(torch.load(run / "model.pt", map_location=dev))
    net.eval()

    torch.manual_seed(0)
    rand = _net(arch, ch).to(dev).eval()

    xdir = mk(out)
    cases = _pick_cases(data, n=n)
    rows: list[dict[str, object]] = []

    for case in cases.itertuples(index=False):
        row = pd.Series(case._asdict())
        sample = Vol2D(pd.DataFrame([row]), cfg)[0]
        img = sample[0].float()
        score, cam = _gradcam(net, img, pool=pool, arch=arch, dev=dev)
        _, rand_cam = _gradcam(rand, img, pool=pool, arch=arch, dev=dev)
        name = f"{row['kind']}_{row['id']}.png"
        extra = _panel(row=row, x=img, cam=cam, out=xdir / name)
        rows.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "y": int(row["y"]),
                "yhat": int(row["yhat"]),
                "p": float(row["p"]),
                "cam_score": float(score),
                "rand_cam_corr": _corr(cam, rand_cam),
                "brain_cam_frac": extra["brain_cam_frac"],
                "slices": extra["slices"],
                "figure": name,
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(xdir / "index.csv", index=False)
    summary = [
        "# CNN XAI",
        "",
        f"Source run: `{run}`",
        f"Architecture: `{arch}`, axis `{axis}`, channels `{ch}`, slices `{slices}`, pick `{pick}`, pool `{pool}`.",
        f"Threshold: `{thr:.4f}`.",
        "",
        "Grad-CAM overlays are generated for representative true/false positives and negatives.",
        "`rand_cam_corr` compares the trained heatmap against a randomly initialised model as a sanity check; high correlation is suspicious.",
        "`brain_cam_frac` reports how much heatmap mass falls inside nonzero brain voxels.",
        "",
        "## Generated Panels",
        "",
        _md_table(res, ["kind", "id", "y", "yhat", "p", "rand_cam_corr", "brain_cam_frac", "figure"], n=max(1, len(res))) if not res.empty else "No cases available.",
    ]
    (xdir / "summary.md").write_text("\n".join(summary) + "\n")

    tab = Table(title="xaicnn", show_lines=False)
    tab.add_column("artifact")
    tab.add_column("path")
    tab.add_row("index.csv", str(xdir / "index.csv"))
    tab.add_row("summary.md", str(xdir / "summary.md"))
    con.print(tab)
