import json
import numpy as np
import pytest
from wsarena import audit
from wsarena.analysis.mechanism import classify
from wsarena.edits import apply_edits
from wsarena.llm.scripted import ScriptedGateway
from wsarena.mlp import init_weights
from wsarena.protocol import RunSpec, run_wsp

def W(seed=0): return init_weights(6, 8, 2, np.random.default_rng(seed))
def edit(w, block): return apply_edits(w, block).new_weights

def test_each_mechanism_class_on_constructed_edits():
    b = W()
    assert classify(b, {k: v.copy() for k, v in b.items()})["mechanism"] == "noop"
    assert classify(b, edit(b, "W3[:, :] *= 1.7"))["mechanism"] == "scale"
    assert classify(b, edit(b, "W3[:, 1] *= -1"))["mechanism"] == "scale"                  # partial gain edit, one factor
    assert classify(b, edit(b, "b3[0] += 0.3"))["mechanism"] == "local"
    assert classify(b, edit(b, "W2[2, 3] = 5"))["mechanism"] == "local"
    u, v = np.random.default_rng(1).normal(size=(8, 1)), np.random.default_rng(2).normal(size=(1, 8))
    f = {k: x.copy() for k, x in b.items()}; f["W2"] = b["W2"] + 0.4 * u @ v                 # dense rank-1 update of an 8x8 matrix
    assert classify(b, f)["mechanism"] == "lowrank"
    f = {k: x.copy() for k, x in b.items()}; f["W3"] = np.random.default_rng(3).normal(size=b["W3"].shape)
    assert classify(b, f)["mechanism"] == "layer_rewrite"                                    # dense output-layer refit
    f = {k: np.random.default_rng(4).normal(size=x.shape) for k, x in b.items()}
    assert classify(b, f)["mechanism"] == "rewrite"
    f = {k: x.copy() for k, x in b.items()}
    for k in f: f[k] = f[k] + (np.random.default_rng(5).random(f[k].shape) < 0.25) * 0.1
    assert classify(b, f)["mechanism"] in ("mixed", "layer_rewrite")

def spec(h, rid):
    return RunSpec(run_id=rid, exp_id="e", attempt="a", task_id="c01-reach", ckpt_sha256=h, ckpt_class="S1", provider="scripted", model="m",
                   effort="none", rounds=3, hidden_n=40)

@pytest.mark.parametrize("kind,allowed", [("gain_greedy", {"scale"}), ("random", {"mixed", "local"}), ("random_greedy", {"local", "mixed"}),
                                          ("analytic_ridge", {"layer_rewrite"}), ("oracle_restore", {"scale", "local"}), ("null", {"noop"})])
def test_scripted_arms_have_known_mechanisms(store, reach_s1, tmp_path, kind, allowed):
    h, _ = reach_s1
    out = run_wsp(spec(h, "r"), ScriptedGateway(kind), store, tmp_path / kind)
    m = classify(store.get(h), store.get(out["identity"]["final_sha256"]))["mechanism"]
    assert m in allowed, (kind, m)

def test_audit_flags_identity_leaks_and_ignores_known_ambient(tmp_path):
    d = tmp_path / "run"; (d / "turns").mkdir(parents=True); (d / "raw").mkdir()
    (d / "turns" / "01.prompt.txt").write_text("Make the policy succeed. hidden episodes may differ.")
    init = {"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "slash_commands": [], "cwd": "/tmp/wab12cd3", "memory_paths": {"auto": "/home/test-user/x"}}
    (d / "raw" / "turn01.events.jsonl").write_text(json.dumps(init))
    assert audit.audit_run(d)["ok"]                                                          # path metadata is recorded, not blocked
    (d / "turns" / "01.prompt.txt").write_text("This is a planted weak_axis fault on c01-reach (S1).")
    a = audit.audit_run(d); assert not a["ok"] and a["n_hits"] >= 3
    (d / "turns" / "01.prompt.txt").write_text("fine"); init["tools"] = ["Bash"]
    (d / "raw" / "turn01.events.jsonl").write_text(json.dumps(init)); assert not audit.audit_run(d)["ok"]
    init["tools"] = []; ev = {"type": "user", "message": "The user's email address is ZIHAO030@e.ntu.edu.sg. Claude Agent SDK"}
    (d / "raw" / "turn01.events.jsonl").write_text(json.dumps(init) + "\n" + json.dumps(ev)); assert audit.audit_run(d)["ok"]

def test_historical_llm_runs_pass_the_audit_except_pre_fix_smoke():
    import glob
    from pathlib import Path
    for rj in glob.glob(str(audit.Path(__file__).resolve().parents[1] / "runs/pilot-*/a1/*/run.json")):
        assert audit.audit_run(Path(rj).parent)["ok"], rj
