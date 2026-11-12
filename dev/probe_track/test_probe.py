import sys, numpy as np, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsarena import mlp
import probe as P

def W(seed=0, n=12): return mlp.init_weights(6, n, 2, np.random.default_rng(seed))

def test_fence_last_block_only():
    t = "x\n```probe\nslice W1[0:1,0:2]\n```\ny\n```probe\nstats W1 axis=0\n```"
    assert P.last_probe_block(t).strip() == "stats W1 axis=0"; assert P.last_probe_block("no block") is None

def test_slice_values_and_limits():
    w = W(); b = P.ProbeBudget(); r = P.answer_probes(w, "slice W1[0:2, 0:3]", b)
    assert r.n_valid == 1 and r.numbers == 6 and "W1[0:2, 0:3]" in r.text
    big = mlp.init_weights(6, 300, 2, np.random.default_rng(1)); r2 = P.answer_probes(big, "slice W2[0:300, 0:300]", P.ProbeBudget())
    assert r2.n_invalid == 1 and r2.numbers == 0 and "exceeds" in r2.text

def test_forward_matches_mlp_and_hidden_slice():
    w = W(); o = [[0.1, -0.2, 0.3, 0.0, 0.5, -0.5]]; r = P.answer_probes(w, f"forward obs={o} layers=1", P.ProbeBudget())
    assert r.n_valid == 1 and "h1[:16]" in r.text
    import json; ans = json.loads(r.text.split("forward -> ")[1].split("\n")[0]); np.testing.assert_allclose(ans["out"], mlp.forward(w, np.array(o)), rtol=1e-3, atol=1e-4)

def test_forbidden_commands_are_invalid_and_counted():
    for cmd in ("rollout n=10", "grad W1", "optimise W1", "import os", "forward obs=[[1,2]]"):
        r = P.answer_probes(W(), cmd, P.ProbeBudget()); assert r.n_invalid == 1 and r.n_valid == 0, cmd

def test_budgets():
    w = W(); b = P.ProbeBudget(max_subturns=2, max_cmds_per_turn=2, max_numbers=10)
    r = P.answer_probes(w, "slice W1[0:2,0:3]\nslice W1[0:2,0:3]\nslice W1[0:1,0:1]", b)   # 6 ok, second would exceed 10, third beyond cmd cap
    assert r.n_valid == 1 and r.n_invalid == 2 and b.used_numbers == 6
    P.answer_probes(w, "topk W1 k=2", b); r3 = P.answer_probes(w, "topk W1 k=2", b); assert r3.refused

def test_probe_never_changes_weights_and_is_deterministic():
    w = W(); before = {k: v.copy() for k, v in w.items()}; a = P.answer_probes(w, "stats W2 axis=1\ntopk W3 k=3", P.ProbeBudget()).text; b = P.answer_probes(w, "stats W2 axis=1\ntopk W3 k=3", P.ProbeBudget()).text
    assert a == b and all(np.array_equal(before[k], w[k]) for k in w)
