"""Figure data layer: reads only versioned/recorded artifacts (run.json, manifest, SUMMARY/*.json); never recomputes results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..analysis.metrics import load_runs
ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIRS = [("baselines-v1", "a1"), ("baselines-greedy-v1", "a1"), ("baselines-gain-v1", "a1"), ("baselines-mw-v1", "a1"), ("baselines-mwpeg-v1", "a1")]


def load_baselines() -> pd.DataFrame:
    parts = [load_runs(ROOT / "runs" / e / a) for e, a in BASELINE_DIRS if (ROOT / "runs" / e / a).exists()]
    d = pd.concat(parts, ignore_index=True)
    sp = d["run_id"].str.split("__")
    d["arm_key"] = sp.str[2]
    d["arm"] = d["arm_key"].replace({"random5": "random", "greedy_f02_s01": "random_greedy", "greedy_f05_s02": "random_greedy"})
    d["kind"] = sp.str[1].str.rsplit("-seed", n=1).str[0]
    return d


def ckpt_level(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, checkpoint, arm): mean over repeats/variants (the cluster unit of inference is the checkpoint)."""
    cols = ["d_sr_id", "d_sr_ood", "recovery_id", "regression_id", "sr_base_id", "sr_base_ood", "frac_changed", "rel_l2"]
    return d.groupby(["task_id", "ckpt", "ckpt_class", "kind", "arm"], as_index=False)[cols].mean()


def boot_ci(x, n=4000, seed=0):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed); b = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return (float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def manifest() -> pd.DataFrame:
    return pd.DataFrame(json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"])


def summary_json(name):
    p = ROOT / "runs/SUMMARY" / name
    return json.loads(p.read_text()) if p.exists() else None
