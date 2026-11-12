"""Meta-World backend tests. Run with the benchmark venv:  venvs/metaworld/bin/python -m pytest tests/test_official_metaworld.py
(skipped automatically in the main .venv, which has no simulator)."""
import importlib, os
import numpy as np
import pytest
pytest.importorskip("metaworld")
from wsarena import mlp, tasks
from wsarena.official import metaworld_task as mw
from wsarena.edits import weights_sha256

T = tasks.get_task("mw-push")

@pytest.fixture(scope="module")
def policy():
    X, Y = T.expert_dataset(24, 3)
    return T.train_policy(X, Y, 1, epochs=60)

def test_registry_and_shapes():
    assert tasks.get_task("mw-pickplace").arch() == (39, 24, 4) and T.act_dim == 4
    with pytest.raises(KeyError):
        tasks.get_task("nope")

def test_rollout_is_deterministic_and_order_independent(policy):
    a = T.rollout(policy, "select", 0, 8); b = T.rollout(policy, "select", 0, 8)
    T.rollout(policy, "select", 100, 4)                          # a different split of seeds in between
    c = T.rollout(policy, "select", 0, 8)
    for x in (b, c):
        assert (a.success == x.success).all() and np.array_equal(a.ret, x.ret) and a.failure == x.failure

def test_parallel_equals_serial(policy, monkeypatch):
    par = T.rollout(policy, "select", 0, 8)
    monkeypatch.setattr(mw, "NPROC", 1)
    ser = T.rollout(policy, "select", 0, 8)
    assert (par.success == ser.success).all() and np.array_equal(par.ret, ser.ret) and par.failure == ser.failure

def test_rollout_does_not_modify_weights_and_namespaces_differ(policy):
    h = weights_sha256(policy); T.rollout(policy, "dev", 0, 4)
    assert weights_sha256(policy) == h
    assert T._seeds("dev", 0, 5) != T._seeds("hid_id", 0, 5) and len(set(T._seeds("select", 0, 200))) == 200

def test_folded_normalisation_equals_normalised_policy():
    X, Y = T.expert_dataset(10, 5)
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-3, 1.0, sd)
    w_n = mlp.init_weights(*T.arch(), np.random.default_rng(0))
    folded = {k: v.copy() for k, v in w_n.items()}
    folded["b1"] = folded["b1"] - (mu / sd) @ folded["W1"]; folded["W1"] = folded["W1"] / sd[:, None]
    assert np.allclose(mlp.forward(w_n, (X - mu) / sd), mlp.forward(folded, X), atol=1e-8)
    assert np.isfinite(folded["W1"]).all() and np.abs(folded["W1"]).max() < 1e4          # no 1/1e-6 blow-ups on constant inputs

def test_ood_gain_changes_behaviour_and_zero_weights_never_succeed(policy):
    z = mlp.init_weights(*T.arch(), zeros=True)
    r = T.rollout(z, "select", 0, 6); assert r.success.sum() == 0 and set(r.failure) <= {"never_near_object", "no_progress", "partial_progress"}
    assert T.describe().count("Observation (39 numbers)") == 1 and "ood" not in T.describe().lower()

def test_expert_dataset_shapes_and_gap_filter():
    X, Y = T.expert_dataset(6, 2, goal_filter={"goal_x_gt": -0.02})
    assert X.shape[1] == 39 and Y.shape[1] == 4 and (X[:, -3] > -0.02).all() and np.abs(Y).max() <= 1.0
