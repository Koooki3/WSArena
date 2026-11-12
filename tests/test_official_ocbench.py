"""OCBench backend tests (run with venvs/ocbench/bin/python -m pytest tests/test_official_ocbench.py; skipped without ocbench)."""
import numpy as np
import pytest
pytest.importorskip("ocbench")
from wsarena import mlp, tasks
from wsarena.edits import weights_sha256
from wsarena.official import ocbench_task as ot

T = tasks.get_task("ocb-block")

def test_registry_shapes_and_description():
    assert T.arch() == (28, 24, 7) and T.act_dim == 7
    d = T.describe(); assert "Observation (28 numbers)" in d and "\\n" not in d and "ood" not in d.lower()

def test_oracle_solves_the_task_and_rollout_is_deterministic():
    seeds = T._seeds("select", 0, 6)
    a = ot._run(T.env_name, seeds, "expert", {}, noise=0.0); b = ot._run(T.env_name, seeds, "expert", {}, noise=0.0)
    assert all(e["success"] for e in a) and [e["len"] for e in a] == [e["len"] for e in b]       # seeded global RNG => reproducible oracle
    w = mlp.init_weights(*T.arch(), np.random.default_rng(0))
    r1, r2 = T.rollout(w, "select", 0, 8), T.rollout(w, "select", 0, 8)
    assert (r1.success == r2.success).all() and np.array_equal(r1.ret, r2.ret) and r1.failure == r2.failure

def test_parallel_equals_serial_and_weights_untouched(monkeypatch):
    w = mlp.init_weights(*T.arch(), np.random.default_rng(1)); h = weights_sha256(w)
    par = T.rollout(w, "dev", 0, 8); assert weights_sha256(w) == h
    monkeypatch.setattr(ot, "NPROC", 1); ser = T.rollout(w, "dev", 0, 8)
    assert par.failure == ser.failure and np.array_equal(par.ret, ser.ret)

def test_namespaces_and_folded_normalisation():
    assert T._seeds("dev", 0, 5) != T._seeds("hid_id", 0, 5) and len(set(T._seeds("select", 0, 100))) == 100
    X, Y = T.expert_dataset(4, 2); mu = X.mean(0); sd = np.where(X.std(0) < 1e-3, 1.0, X.std(0))
    w = mlp.init_weights(*T.arch(), np.random.default_rng(0)); f = {k: v.copy() for k, v in w.items()}
    f["b1"] = f["b1"] - (mu / sd) @ f["W1"]; f["W1"] = f["W1"] / sd[:, None]
    assert np.allclose(mlp.forward(w, (X - mu) / sd), mlp.forward(f, X), atol=1e-8) and np.abs(f["W1"]).max() < 1e4
