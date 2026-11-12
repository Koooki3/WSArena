"""Feasibility of Meta-World as a WSArena official-benchmark tier (run with venvs/metaworld python):
can small MLP BC policies (39->24->24->4, ~1.7k params) reach diagnosable SR bands, and how fast is evaluation?
Uses the 'select' seed namespace only. Output: runs/SUMMARY/feasibility-metaworld.json (not part of any main analysis)."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import gymnasium as gym, metaworld
from metaworld.policies import ENV_POLICY_MAP
from wsarena import mlp, envs, tasks

def seed(task, i): return envs.seed_for("mw-" + task, "select", i) % (2**31)

def rollout(name, policy_fn, seeds, record=None, noise=0.0, rng=None):
    env = gym.make("Meta-World/MT1", env_name=name, seed=0); ok = []; steps = 0
    for s in seeds:
        ob, _ = env.reset(seed=int(s)); done = False; succ = False
        while not done:
            a = policy_fn(ob)
            if record is not None: record[0].append(ob.copy()); record[1].append(np.clip(a, -1, 1))
            if noise and rng is not None: a = a + rng.normal(0, noise, a.shape)
            ob, r, te, tr, info = env.step(np.clip(a, -1, 1)); steps += 1
            succ |= bool(info.get("success", 0)); done = te or tr
        ok.append(succ)
    env.close(); return np.array(ok), steps

out = {}
for name in sys.argv[1:] or ["reach-v3", "push-v3", "pick-place-v3"]:
    t0 = time.time(); pol = ENV_POLICY_MAP[name](); rng = np.random.default_rng(0)
    rec = ([], [])
    rollout(name, pol.get_action, [seed(name, 10_000 + i) for i in range(40)], record=rec, noise=0.15, rng=rng)
    X, Y = np.array(rec[0]), np.array(rec[1]); t_collect = time.time() - t0
    mu, sd = X.mean(0), X.std(0) + 1e-6          # NOTE: input normalisation would have to live in the weights (W1/b1)
    rows = []
    for epochs in (20, 100, 400):
        w = tasks.train_bc((39, 24, 4), (X - mu) / sd, Y, 1, epochs=epochs, lr=3e-3, batch=1024)
        # fold the normalisation into layer 1 so that the policy is a plain MLP on raw observations (no hidden preprocessing)
        w["b1"] = w["b1"] - (mu / sd) @ w["W1"]; w["W1"] = w["W1"] / sd[:, None]
        t1 = time.time(); ok, steps = rollout(name, lambda o: mlp.forward(w, o)[0], [seed(name, i) for i in range(30)])
        rows.append({"epochs": epochs, "sr": float(ok.mean()), "eval_s_per_30ep": round(time.time() - t1, 1), "steps_per_s": round(steps / (time.time() - t1))})
        print(name, rows[-1], flush=True)
    out[name] = {"n_samples": int(len(X)), "collect_s": round(t_collect, 1), "bc": rows}
Path(ROOT / "runs/SUMMARY/feasibility-metaworld.json").write_text(json.dumps(out, indent=1))
