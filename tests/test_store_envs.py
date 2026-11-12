import os
import numpy as np
import pytest
from wsarena import envs, mlp, tasks, evaluation
from wsarena.store import CheckpointStore, IntegrityError

def test_store_roundtrip_immutable_tamper(tmp_path):
    s = CheckpointStore(tmp_path); w = mlp.init_weights(4, 6, 2, np.random.default_rng(0))
    h = s.put(w, {"class": "X"}); assert s.has(h) and s.meta(h)["class"] == "X"
    assert s.put(w) == h
    w2 = s.get(h); assert all(np.array_equal(w[k], w2[k]) for k in w)
    npz = s._paths(h)[0]; os.chmod(npz, 0o644)
    bad = dict(w); bad["b1"] = bad["b1"] + 1
    np.savez(npz, **bad)
    with pytest.raises(IntegrityError):
        s.get(h)

def test_seed_namespaces_disjoint_and_deterministic():
    a = {envs.seed_for("t", sp, i) for sp in ("dev", "hid_id", "hid_ood", "select") for i in range(500)}
    assert len(a) == 2000
    assert envs.seed_for("t", "dev", 3) == envs.seed_for("t", "dev", 3)

@pytest.mark.parametrize("tid", list(tasks.TASKS))
def test_rollout_deterministic_and_expert_beats_zero(tid):
    t = tasks.TASKS[tid]; seeds = [envs.seed_for(tid, "select", i) for i in range(40)]
    ex = lambda o: tasks.expert_action(t.params_id, o)
    r1 = envs.rollout(t.params_id, ex, seeds); r2 = envs.rollout(t.params_id, ex, seeds)
    assert (r1.success == r2.success).all() and np.allclose(r1.ret, r2.ret)
    z = envs.rollout(t.params_id, lambda o: np.zeros((len(o), 2)), seeds)
    assert z.success.mean() == 0.0 and r1.success.mean() > 0.5

def test_probe_is_pure():
    t = tasks.TASKS["c01-reach"]; w = mlp.init_weights(*t.arch(), np.random.default_rng(1))
    before = {k: v.copy() for k, v in w.items()}
    out = evaluation.probe(w, [[0.1, 0.2, 0, 0]] * 40, max_inputs=32)
    assert out["n"] == 32 and all(np.array_equal(before[k], w[k]) for k in w)
    with pytest.raises(ValueError):
        evaluation.probe(w, [[1, 2, 3]], 5)

def test_reparameterize_preserves_function():
    t = tasks.TASKS["c02-avoid"]; rng = np.random.default_rng(3)
    w = mlp.init_weights(*t.arch(), rng); x = rng.normal(size=(50, t.obs_dim))
    w2 = mlp.reparameterize(w, rng)
    assert np.max(np.abs(mlp.forward(w, x) - mlp.forward(w2, x))) < 1e-10
    assert not np.allclose(w["W1"], w2["W1"])


def test_train_bc_snapshots_and_noise_s3_not_in_manifest_builder():
    import re
    from pathlib import Path
    t = tasks.TASKS["c01-reach"]
    X, Y = tasks.collect_dataset(t, 10, 1)
    ss = tasks.train_bc(t.arch(), X, Y, 1, epochs=4, lr=1e-3, snapshots=[2, 4])
    assert [e for e, _ in ss] == [2, 4] and not np.array_equal(ss[0][1]["W1"], ss[1][1]["W1"])
    src = (Path(__file__).resolve().parents[1] / "scripts" / "make_tasks.py").read_text()
    assert "build_s3(" in src and "build_s3_noise" not in src          # v1 noise construction must not come back


def test_avoid_expert_defaults_unchanged_by_refactor():
    """The tang/rep_radius parameters were added with defaults equal to the v1 constants: behaviour must be identical."""
    t = tasks.TASKS["c02-avoid"]; rng = np.random.default_rng(0); obs = rng.normal(size=(30, 6))
    a = tasks.expert_action(t.params_id, obs)
    b = tasks.expert_action(t.params_id, obs, k_rep=0.25, rep_radius=0.55, tang=1.3, kd=3.6)
    assert np.array_equal(a, b)
