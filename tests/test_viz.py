"""Figure-system tests: every baseline figure renders in both themes with a table view; LLM-era templates stay 'pending' without
data (no placeholder charts) and render correctly when fed synthetic LLM-labelled rows; palette tokens equal the validated hexes."""
import json
import numpy as np, pandas as pd
import pytest
from wsarena.viz import figures, style
from wsarena.viz import llm_figures as lf

BASE = [f for f in figures.REGISTRY if not f.fid.startswith("LLM")]

def test_validated_palette_tokens():
    assert style.CAT["light"][:6] == ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    assert style.CAT["dark"][:6] == ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
    assert len({v for v in style.ENTITY_SLOT.values()}) == len(style.ENTITY_SLOT)          # one slot per entity, never shared
    assert style.color("null", "light") == style.CHROME["light"]["muted"]                  # reference entity is gray, not a hue

@pytest.mark.parametrize("fig", BASE, ids=lambda f: f.fid)
def test_baseline_figures_render_both_themes(fig, tmp_path):
    for mode in ("light", "dark"):
        f, tab, note = fig.fn(mode)
        assert len(f.axes) >= 1 and isinstance(tab, pd.DataFrame) and len(tab) > 0 and note
        paths = style.save_all(f, fig.fid, tmp_path, mode, formats=("png",)); assert paths

def test_llm_figures_pending_without_data(monkeypatch):
    monkeypatch.setattr(lf, "ROOT", lf.ROOT / "nonexistent")
    for fid in ("LLM-01", "LLM-02", "LLM-03", "LLM-04", "LLM-05"):
        f = next(x for x in figures.REGISTRY if x.fid == fid)
        with pytest.raises(figures.PendingData):
            f.fn("light")

def test_llm_figures_render_on_synthetic_runs(monkeypatch, tmp_path):
    from wsarena.viz import data as D
    base = D.load_baselines(); b = base[base.arm.isin(["analytic_ridge", "random_greedy"])].copy()
    rng = np.random.default_rng(0); models = [("claude-sonnet-5-5", "medium"), ("claude-sonnet-5-5", "high"), ("claude-opus-5-5", "medium"), ("claude-opus-5-5", "high")]
    parts = []
    for i, (m, e) in enumerate(models):
        q = b.sample(60, random_state=i).copy(); q["model"], q["effort"], q["provider"] = m, e, "claude-cli"
        q["entity"] = lf.MODEL_LABEL[m]; q["exp_id"] = "pilot-v0"; q["tok_out"] = rng.integers(8_000, 90_000, len(q)); parts.append(q)
    d = pd.concat(parts, ignore_index=True)
    monkeypatch.setattr(lf, "_llm_runs", lambda: d)
    for fid in ("LLM-01", "LLM-02", "LLM-03", "LLM-04", "LLM-05"):
        f = next(x for x in figures.REGISTRY if x.fid == fid)
        for mode in ("light", "dark"):
            fig, tab, note = f.fn(mode); assert len(tab) > 0; style.save_all(fig, fid, tmp_path, mode, formats=("png",))


def test_smoke_runs_are_never_rendered_as_llm_results(tmp_path, monkeypatch):
    """PREREG s8: cost-smoke output is excluded from every analysis; the templates must stay 'pending' on smoke-only data."""
    import json, shutil
    from pathlib import Path
    root = tmp_path / "proj"; (root / "runs" / "smoke-v0" / "a1" / "r").mkdir(parents=True)
    src = next((lf.ROOT / "runs" / "smoke-v0" / "a1").glob("*/run.json"), None)
    if src: shutil.copy(src, root / "runs" / "smoke-v0" / "a1" / "r" / "run.json")
    monkeypatch.setattr(lf, "ROOT", root)
    with pytest.raises(figures.PendingData):
        lf._llm_runs()
