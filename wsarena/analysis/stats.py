"""Paired, clustered inference. Unit of inference = checkpoint (task x ckpt): LLM repeats are averaged inside a
checkpoint; episodes inside a run are never treated as independent samples. Stratified by task."""
from __future__ import annotations
import numpy as np
import pandas as pd


def paired_diff_bootstrap(df: pd.DataFrame, value: str, arm_col: str, a, b, cluster: str = "ckpt",
                          strata: str = "task_id", n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean of (value|arm=a) - (value|arm=b) over checkpoints present in BOTH arms; percentile CI by resampling
    checkpoints within each task stratum."""
    d = df[df[arm_col].isin([a, b])]
    per = d.groupby([strata, cluster, arm_col])[value].mean().unstack(arm_col).dropna(subset=[a, b])
    if per.empty:
        return {"n_clusters": 0, "mean_diff": None, "ci95": None}
    diff = (per[a] - per[b]).reset_index(name="diff")
    rng = np.random.default_rng(seed)
    groups = [g["diff"].to_numpy() for _, g in diff.groupby(strata)]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = [g[rng.integers(0, len(g), len(g))] for g in groups]
        boots[i] = np.mean(np.concatenate(s))
    return {"n_clusters": int(len(diff)), "mean_diff": float(diff["diff"].mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); out[k] = running
    return out


def sign_flip_pvalue(diffs: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on per-cluster differences."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0: return float("nan")
    rng = np.random.default_rng(seed); obs = abs(diffs.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    return float(((np.abs((flips * diffs).mean(axis=1)) >= obs - 1e-15).sum() + 1) / (n_perm + 1))
