"""INDEPENDENT recomputation (K-S5-02, numeric line). Imports NOTHING from wsarena: re-implements the seed
namespaces, the point-mass simulator and the hidden evaluation with a different (per-episode, loop-based) structure,
loads weights straight from the .npz in artifacts/checkpoints, and compares with the numbers in run.json.
usage: python checks/recompute.py runs/<exp>/<attempt> [--max-runs N]     exit 1 on any mismatch (tol 1e-9 on SR)
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PARAMS = json.loads((ROOT / "configs/tasks/task_params.json").read_text())
DT, ARENA = 0.1, 1.2

def seed_for(task, split, i):
    return int.from_bytes(hashlib.sha256(f"wsarena|{task}|{split}|{i}".encode()).digest()[:8], "little") % (2**63)

def load(h):
    z = np.load(Path(os.environ.get("WSA_CKPT_ROOT", ROOT / "artifacts/checkpoints")) / "sha256" / h[:2] / f"{h}.npz")
    return {k: z[k].astype(float) for k in z.files}

def policy(w, obs):
    h1 = np.tanh(obs @ w["W1"] + w["b1"]); h2 = np.tanh(h1 @ w["W2"] + w["b2"]); return h2 @ w["W3"] + w["b3"]

def episode(P, w, seed):
    r = np.random.default_rng(seed)
    for _ in range(200):
        p = r.uniform(-0.9, 0.9, 2); g = r.uniform(-0.8, 0.8, 2)
        if np.hypot(*(g - p)) >= 0.7: break
    ob = np.zeros(2)
    if P["kind"] == "avoid":
        d = g - p; n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
        ob = (p + g) / 2 + n * r.uniform(-0.08, 0.08)
    v = np.zeros(2); E = [g - p]; V = [v.copy()]; A = []; collided = False; zone = []
    for t in range(P["T"]):
        e = g - p
        k = P["kind"]
        if k == "reach": o = np.r_[e, v]
        elif k == "avoid": o = np.r_[e, v, ob - p]
        elif k == "hold": o = np.r_[e, v, t / P["T"]]
        else:
            j = max(t - P["delay"], 0)
            a1 = A[t - 1] if t >= 1 else np.zeros(2); a2 = A[t - 2] if t >= 2 else np.zeros(2)
            o = np.r_[E[j], V[j], a1, a2]
        a = np.clip(policy(w, o[None])[0], -1, 1)
        v = v + DT * (P["fmax"] * a / P["mass"] - P["damping"] * v)
        p = p + DT * v
        for ax in range(2):
            if abs(p[ax]) > ARENA: p[ax] = np.clip(p[ax], -ARENA, ARENA); v[ax] = 0.0   # NB: clamp axis-wise below
        if k == "avoid" and np.hypot(*(p - ob)) < P["obstacle_radius"]: collided = True
        d = np.hypot(*(g - p)); zone.append(d < P["goal_radius"] and np.hypot(*v) < P["speed_tol"])
        E.append(g - p); V.append(v.copy()); A.append(a)
    done = zone[-1]
    if P["kind"] == "hold": done = all(zone[-P["hold_steps"]:])
    if P["kind"] == "avoid": done = done and not collided
    return bool(done)

def sr(task, ood, w, split, n):
    P = PARAMS[task]["params_ood" if ood else "params_id"]
    return np.array([episode(P, w, seed_for(task, split, i)) for i in range(n)])

if __name__ == "__main__":
    root = Path(sys.argv[1]); mx = int(sys.argv[sys.argv.index("--max-runs") + 1]) if "--max-runs" in sys.argv else 10**9
    bad = 0; n_ck = 0
    for rj in sorted(root.glob("*/run.json"))[:mx]:
        j = json.loads(rj.read_text())
        if j.get("status") not in ("verified", "failed"): continue
        sp, ident, hid = j["spec"], j["identity"], j["final"]["hidden"]
        wb, wf = load(ident["base_sha256"]), load(ident["final_sha256"])
        for name, split, ood in (("id", "hid_id", False), ("ood", "hid_ood", True)):
            n = hid[name]["n"]; sb, sf = sr(sp["task_id"], ood, wb, split, n), sr(sp["task_id"], ood, wf, split, n)
            diffs = [abs(sb.mean() - hid[name]["sr_base"]), abs(sf.mean() - hid[name]["sr_edited"]),
                     abs((sf.mean() - sb.mean()) - hid[name]["delta_sr"])]
            rec = float(sf[~sb].mean()) if (~sb).any() else None
            if rec is not None and hid[name]["recovery"] is not None: diffs.append(abs(rec - hid[name]["recovery"]))
            ok = max(diffs) < 1e-9; bad += (not ok); n_ck += 1
            print(f"{'OK ' if ok else 'BAD'} {sp['run_id']} {name}: SR base {sb.mean():.3f} final {sf.mean():.3f} (maxdiff {max(diffs):.2e})")
    print(f"{n_ck} checks, {bad} mismatches"); sys.exit(1 if bad else 0)
