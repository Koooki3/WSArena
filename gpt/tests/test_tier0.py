import pandas as pd
import pytest

from gpt import tier0


def test_c3_uses_equal_task_weights_with_unequal_cluster_counts():
    rows = pd.DataFrame(
        [("a", "L3", .4), ("a", "L2", 0.), ("a", "L2", 0.), ("a", "L2", 0.),
         ("b", "L3", 0.), ("b", "L2", .2)],
        columns=["task_id", "level", "value"],
    )
    result = tier0.c3(rows)
    assert result["mean_diff"] == pytest.approx(.1)
    assert result["cluster_weighted_diff"] == pytest.approx(.15)
    assert result["ci95"] == pytest.approx([.1, .1])
    assert 0 < result["p"] <= 1


def test_analysis_refuses_incomplete_grid(tmp_path, monkeypatch):
    cfg, items, clusters = tier0.planned()
    monkeypatch.setattr(tier0, "ROOT", tmp_path)
    monkeypatch.setattr(tier0.identity, "capture", lambda *args: {})
    with pytest.raises(ValueError, match=r"Tier-0 incomplete \(64 of 64\)"):
        tier0.analyze(cfg, items, clusters, {}, "empty")
