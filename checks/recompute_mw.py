"""INDEPENDENT recomputation for the Meta-World tier (K-S5-02, numeric line). Imports NOTHING from wsarena: re-implements the seed namespace, the
deterministic episode setup (u.seed -> sample rand_vec -> freeze -> reset), the actuator-gain OOD wrapper, the early-stop-on-success rule and the SR,
loads weights straight from the .npz, and compares with the numbers stored in run.json.  usage: recompute_mw.py runs/baselines-mw-v1/a1 [--max-runs N]"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ENVN = {"mw-push": ("push-v3", 0.5), "mw-pickplace": ("pick-place-v3", 0.6)}
def seed_for(task, split, i): return int.from_bytes(hashlib.sha256(f"wsarena|{task}|{split}|{i}".encode()).digest()[:8], "little") % (2**63) % (2**31)
def load(h):
    z = np.load(Path(os.environ.get("WSA_CKPT_ROOT", ROOT / "artifacts/checkpoints")) / "sha256" / h[:2] / f"{h}.npz"); return {k: z[k].astype(float) for k in z.files}
def pol(w, o):
    h1 = np.tanh(o @ w["W1"] + w["b1"]); h2 = np.tanh(h1 @ w["W2"] + w["b2"]); return h2 @ w["W3"] + w["b3"]
_env = {}
def sr(task, w, split, n, gain):
    import gymnasium as gym, metaworld  # noqa
    name = ENVN[task][0]
    if name not in _env: _env[name] = gym.make("Meta-World/MT1", env_name=name, seed=0)
    env = _env[name]; u = env.unwrapped; ok = []
    for i in range(n):
        u._freeze_rand_vec = False; u.seed(seed_for(task, split, i)); env.reset(); rv = u._last_rand_vec.copy(); u._freeze_rand_vec = True; u._last_rand_vec = rv
        ob = env.reset()[0]; done = False
        for t in range(500):
            a = np.clip(pol(w, ob), -1, 1); ob, r, te, tr, info = env.step(np.clip(a * gain, -1, 1))
            if float(info.get("success", 0)) > 0.5: done = True; break
            if te or tr: break
        ok.append(done)
    return np.array(ok)
if __name__ == "__main__":
    root = Path(sys.argv[1]); mx = int(sys.argv[sys.argv.index("--max-runs") + 1]) if "--max-runs" in sys.argv else 10**9
    bad = n_ck = 0
    for rj in sorted(root.glob("*/run.json"))[:mx]:
        j = json.loads(rj.read_text()); sp, idn, h = j["spec"], j["identity"], j["final"]["hidden"]["id"]
        if idn["base_sha256"] == idn["final_sha256"]: continue
        wb, wf = load(idn["base_sha256"]), load(idn["final_sha256"]); n = h["n"]
        sb, sf = sr(sp["task_id"], wb, "hid_id", n, 1.0), sr(sp["task_id"], wf, "hid_id", n, 1.0)
        d = max(abs(sb.mean() - h["sr_base"]), abs(sf.mean() - h["sr_edited"])); ok = d < 1e-9; bad += (not ok); n_ck += 1
        print(("OK " if ok else "BAD"), sp["run_id"], f"base {sb.mean():.2f} final {sf.mean():.2f} (maxdiff {d:.1e})", flush=True)
    print(f"{n_ck} checks, {bad} mismatches"); sys.exit(1 if bad else 0)
