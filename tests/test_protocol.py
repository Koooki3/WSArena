import json
import pytest
from wsarena import evaluation, prompts, tasks
from wsarena.llm.base import Audit, Gateway, TurnResult, Usage
from wsarena.llm.scripted import ScriptedGateway
from wsarena.protocol import RunInterrupted, RunSpec, run_wsp

def spec(h, rounds=3, **kw):
    return RunSpec(run_id="t", exp_id="e", attempt="a", task_id="c01-reach", ckpt_sha256=h, ckpt_class="S1",
                   provider="scripted", model="m", effort="none", rounds=rounds, hidden_n=60, **kw)

class Flaky(ScriptedGateway):
    """Scripted gateway that can fail / return custom text on given rounds and counts calls."""
    def __init__(self, kind, fail_round=None, fail_status="rate_limited", text=None, tool_free=True):
        super().__init__(kind); self.calls = 0; self.fail_round = fail_round; self.fail_status = fail_status
        self.text = text; self.tool_free = tool_free
    def complete(self, req):
        self.calls += 1
        if req.turn_idx == self.fail_round:
            return TurnResult(status=self.fail_status, audit=Audit(tool_free=True), provider="x", model="m", effort="e",
                              system_sha256="0", error="usage limit reached")
        r = super().complete(req)
        if self.text is not None:
            r.text = self.text
        r.audit = Audit(tool_free=self.tool_free)
        return r

def test_null_editor_changes_nothing(store, reach_s1, tmp_path):
    h, _ = reach_s1
    out = run_wsp(spec(h), ScriptedGateway("null"), store, tmp_path / "r")
    assert out["status"] == "verified"
    assert out["identity"]["final_sha256"] == h
    assert out["final"]["hidden"]["id"]["delta_sr"] == 0.0
    assert len(out["rounds"]) == 3

def test_oracle_restore_recovers_and_localizes(store, reach_s1, tmp_path):
    h, _ = reach_s1
    out = run_wsp(spec(h), ScriptedGateway("oracle_restore"), store, tmp_path / "r")
    assert out["final"]["hidden"]["id"]["delta_sr"] > 0.3
    loc = out["final"]["localization"]
    assert loc["recall"] > 0.9 and loc["precision"] > 0.9
    assert out["final"]["hidden"]["id"]["recovery"] > 0.3

def test_dev_budget_accounting_and_hidden_never_in_prompt(store, reach_s1, tmp_path):
    h, _ = reach_s1
    run_wsp(spec(h, rounds=2), ScriptedGateway("null"), store, tmp_path / "r")
    rounds = json.load(open(tmp_path / "r" / "rounds.json"))
    assert [r["dev_report"]["episodes_used"] for r in rounds] == [16, 16]
    for p in (tmp_path / "r" / "turns").glob("*.prompt.txt"):
        t = p.read_text()
        assert "hid_id" not in t and "hid_ood" not in t
        for leak in ("planted", "corrupt", "S1", "ground truth", "broken", "damaged"):
            assert leak not in t, leak

def test_budget_overflow_raises():
    b = evaluation.Budget(rounds=1, episodes_per_round=4, initial_report_episodes=4)
    b.charge_episodes(4); b.charge_episodes(4)
    with pytest.raises(evaluation.BudgetExceeded):
        b.charge_episodes(1)

def test_interrupt_then_resume_replays_without_new_calls(store, reach_s1, tmp_path):
    h, _ = reach_s1
    d = tmp_path / "r"
    g1 = Flaky("random", fail_round=3)
    with pytest.raises(RunInterrupted):
        run_wsp(spec(h, rounds=4), g1, store, d)
    assert g1.calls == 3 and (d / "interrupted.json").exists()
    assert not (d / "turns" / "03.result.json").exists()          # nothing saved for the failed turn
    g2 = Flaky("random")
    out = run_wsp(spec(h, rounds=4), g2, store, d)
    assert g2.calls == 2 and out["status"] == "verified" and len(out["rounds"]) == 4   # rounds 3,4 only

def test_tool_use_invalidates_run(store, reach_s1, tmp_path):
    h, _ = reach_s1
    out = run_wsp(spec(h, rounds=2), Flaky("null", tool_free=False), store, tmp_path / "r")
    assert out["status"] == "failed" and out["invalid_reason"] == "tool_use_detected"

def test_failures_stay_in_denominator(store, reach_s1, tmp_path):
    h, _ = reach_s1
    out = run_wsp(spec(h, rounds=3), Flaky("null", text="I think the weights are fine."), store, tmp_path / "r")
    assert out["turns_total"] == 3 and out["valid_patch_rounds"] == 0
    assert all(r["patch_status"] == "no_block" for r in out["rounds"])
    out2 = run_wsp(spec(h, rounds=2).model_copy(update={"run_id": "t2"}), Flaky("null", fail_round=1, fail_status="output_budget_exhausted"),
                   store, tmp_path / "r2")
    assert out2["rounds"][0]["patch_status"] == "turn_failed:output_budget_exhausted"
    assert out2["status"] == "verified"

def test_hash_mismatch_refuses(store, reach_s1, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_wsp(spec("0" * 64), ScriptedGateway("null"), store, tmp_path / "r")

def test_prompt_identical_across_classes():
    t = tasks.TASKS["c01-reach"]
    z = tasks.build_s0(t)["weights"]
    rep = {"episodes_used": 16}
    a = prompts.build_first_user_message(t, z, rep, 8, 16)
    assert "broken" not in a and "defect" not in a and "fault" not in a.lower().replace("default", "")


def test_independent_recompute_matches_run_json(store, reach_s1, tmp_path, monkeypatch):
    """checks/recompute.py (imports nothing from wsarena) must reproduce the hidden-eval numbers exactly."""
    import importlib.util, json
    from pathlib import Path
    h, _ = reach_s1
    run_wsp(spec(h, rounds=2), ScriptedGateway("oracle_restore"), store, tmp_path / "r")
    monkeypatch.setenv("WSA_CKPT_ROOT", str(store.root))
    sp = importlib.util.spec_from_file_location("recompute", Path(__file__).resolve().parents[1] / "checks" / "recompute.py")
    rc = importlib.util.module_from_spec(sp); sp.loader.exec_module(rc)
    j = json.loads((tmp_path / "r" / "run.json").read_text())
    wb, wf = rc.load(j["identity"]["base_sha256"]), rc.load(j["identity"]["final_sha256"])
    hid = j["final"]["hidden"]
    for name, split, ood in (("id", "hid_id", False), ("ood", "hid_ood", True)):
        n = hid[name]["n"]
        sb, sf = rc.sr("c01-reach", ood, wb, split, n), rc.sr("c01-reach", ood, wf, split, n)
        assert abs(sb.mean() - hid[name]["sr_base"]) < 1e-12 and abs(sf.mean() - hid[name]["sr_edited"]) < 1e-12


def test_localization_is_none_without_planted_truth():
    from wsarena.protocol import localization
    from wsarena.mlp import init_weights
    w = init_weights(4, 6, 2, zeros=True)
    assert localization(w, w, None) is None
    assert localization(w, w, {"mode": "flawed_damping", "kd": 1.2}) is None      # S2 meta (regression test)


def test_random_greedy_never_drops_below_best_seen_dev_sr_structure(store, reach_s1, tmp_path):
    """The greedy control re-applies the best-so-far weights then perturbs; it must run through the protocol and
    keep every round's weights retrievable."""
    h, _ = reach_s1
    out = run_wsp(spec(h, rounds=3), ScriptedGateway("random_greedy", frac=0.02, scale=0.1), store, tmp_path / "r")
    assert out["status"] == "verified" and out["valid_patch_rounds"] == 3
    for r in (1, 2, 3):
        assert store.has(out["identity"]["round_sha256"][str(r)] if str(r) in out["identity"]["round_sha256"] else out["identity"]["round_sha256"][r])


def test_gateway_workdir_is_neutral_and_raw_events_go_elsewhere(store, reach_s1, tmp_path):
    """The model can see its cwd path: it must never contain task / class / arm / run names (found by the live isolation probe)."""
    h, _ = reach_s1; seen = []
    class Rec(ScriptedGateway):
        def complete(self, req):
            seen.append((req.workdir, req.raw_dir)); return super().complete(req)
    sp = spec(h, rounds=2).model_copy(update={"run_id": "c01-reach__s1-weak_axis-seed1__opus55-high__r0"})
    run_wsp(sp, Rec("null"), store, tmp_path / "r")
    import os
    for wd, rd in seen:
        assert os.path.basename(wd).startswith("w") and len(os.path.basename(wd)) <= 12
        assert "c01-reach" not in wd and "weak_axis" not in wd and str(tmp_path) not in wd
        assert rd is not None and "raw" in rd
    assert not any(os.path.exists(wd) for wd, _ in seen)          # neutral dirs are removed after the run


def test_gain_greedy_runs_through_protocol_and_scales_only_w3(store, reach_s1, tmp_path):
    from wsarena.edits import diff_stats
    h, _ = reach_s1
    out = run_wsp(spec(h, rounds=3), ScriptedGateway("gain_greedy"), store, tmp_path / "r")
    assert out["status"] == "verified" and out["valid_patch_rounds"] == 3
    base, fin = store.get(h), store.get(out["identity"]["final_sha256"])
    d = diff_stats(base, fin)["per_tensor"]
    assert d["W3"]["changed"] > 0 and all(d[k]["changed"] == 0 for k in ("W1", "b1", "W2", "b2", "b3"))


def test_prompt_variants_differ_only_in_the_weights_block():
    import re
    t = tasks.TASKS["c01-reach"]; w = __import__("wsarena.mlp", fromlist=["x"]).init_weights(*t.arch(), __import__("numpy").random.default_rng(3))
    rep = {"episodes_used": 16}
    full = prompts.build_first_user_message(t, w, rep, 4, 16)
    blind = prompts.build_first_user_message(t, w, rep, 4, 16, weights_view="blind")
    shuf = prompts.build_first_user_message(t, w, rep, 4, 16, weights_view="shuffled", shuffle_seed=7)
    assert full == prompts.build_first_user_message(t, w, rep, 4, 16, weights_view="full")       # default unchanged (pilot prompts stay valid)
    strip = lambda s: re.sub(r"Current weights \(rounded.*?\n\nHOW TO WRITE WEIGHTS", "<W>\n\nHOW TO WRITE WEIGHTS", s, flags=re.S)
    assert "W1 shape=" in full and "W1 shape=" in shuf and "W1 shape=" not in blind and "NOT shown" in blind
    nums = lambda s: sorted(re.findall(r"-?\d+\.?\d*e?-?\d*", s.split("Current weights")[1].split("HOW TO WRITE")[0]))
    assert nums(shuf) == nums(full) and shuf != full                                          # same values, different positions
    head = lambda s: s.split("Current weights")[0]
    assert head(full) == head(blind) == head(shuf)
    assert full.split("HOW TO WRITE")[1] == blind.split("HOW TO WRITE")[1] == shuf.split("HOW TO WRITE")[1]
