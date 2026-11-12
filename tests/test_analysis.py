import numpy as np, pandas as pd
from wsarena.analysis import stats

def test_bootstrap_detects_true_shift_and_clusters_not_episodes():
    rows = []
    for t in ("a", "b"):
        for c in range(8):
            base = np.random.default_rng(c).uniform(0, .4)
            for arm, dv in (("X", 0.2), ("Y", 0.0)):
                for rep in range(3):
                    rows.append(dict(task_id=t, ckpt=f"{t}{c}", arm=arm, v=base + dv + np.random.default_rng(rep).normal(0, .02)))
    r = stats.paired_diff_bootstrap(pd.DataFrame(rows), "v", "arm", "X", "Y")
    assert r["n_clusters"] == 16 and 0.15 < r["mean_diff"] < 0.25 and r["ci95"][0] > 0.1

def test_holm_and_signflip():
    h = stats.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert h["a"] == 0.03 and h["c"] == 0.5 and h["b"] >= h["a"]
    assert stats.sign_flip_pvalue(np.full(12, 0.1)) < 0.01
    assert stats.sign_flip_pvalue(np.array([0.1, -0.1] * 6)) > 0.5
