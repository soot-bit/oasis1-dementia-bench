from __future__ import annotations

import numpy as np


def _q(x: np.ndarray, q: tuple[float, float, float] = (0.025, 0.5, 0.975)) -> dict[str, float]:
    lo, mid, hi = np.clip(np.quantile(x, q), 0.0, 1.0)
    return {"lo": float(lo), "mid": float(mid), "hi": float(hi)}


def _beta(ok: int, bad: int, draws: int, rng: np.random.Generator, a: float = 0.5, b: float = 0.5) -> np.ndarray:
    return rng.beta(ok + a, bad + b, size=draws)


def cls_ci(y: np.ndarray, p: np.ndarray, thr: float = 0.5, draws: int = 20000, seed: int = 7) -> dict[str, object]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)

    tp = int(np.sum((y == 1) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))

    sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    bal = float((sens + spec) / 2.0) if np.isfinite(sens) and np.isfinite(spec) else float("nan")

    rng = np.random.default_rng(seed)
    sens_d = _beta(ok=tp, bad=fn, draws=draws, rng=rng)
    spec_d = _beta(ok=tn, bad=fp, draws=draws, rng=rng)
    bal_d = (sens_d + spec_d) / 2.0

    return {
        "thr": float(thr),
        "draws": int(draws),
        "sens": {"point": sens, **_q(sens_d)},
        "spec": {"point": spec, **_q(spec_d)},
        "bal_acc": {"point": bal, **_q(bal_d)},
        "counts": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def _auc_w(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        return float("nan")

    pp = p[pos][:, None]
    pn = p[neg][None, :]
    wp = w[pos][:, None]
    wn = w[neg][None, :]

    win = (pp > pn).astype(float) + 0.5 * (pp == pn).astype(float)
    den = float(w[pos].sum() * w[neg].sum())
    if den <= 0:
        return float("nan")
    return float(np.clip(np.sum(win * (wp * wn)) / den, 0.0, 1.0))


def auc_bb_ci(y: np.ndarray, p: np.ndarray, draws: int = 4000, seed: int = 7) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {"point": float("nan"), "lo": float("nan"), "mid": float("nan"), "hi": float("nan"), "draws": int(draws)}

    rng = np.random.default_rng(seed)
    point = _auc_w(y=y, p=p, w=np.ones(len(y), dtype=float))
    d = np.empty(draws, dtype=float)
    for i in range(draws):
        w = rng.dirichlet(np.ones(len(y), dtype=float))
        d[i] = _auc_w(y=y, p=p, w=w)
    return {"point": point, **_q(d), "draws": int(draws)}
