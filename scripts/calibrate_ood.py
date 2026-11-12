"""Calibrate the OOD shift strength per task on the 'select' namespace ONLY (never hid_ood: no tuning on hidden data).
Rule (written before looking at results): OOD params(s) = ID + s*(OOD_v1 - ID), s in a fixed grid. Choose the LARGEST s with
   mean OOD SR over S2 U S3 baseline checkpoints >= 0.20   (the shift must not floor the policies under test)
   and mean (ID SR - OOD SR) over the same checkpoints >= 0.15   (the shift must matter).
If no s satisfies both, report the s with the highest OOD SR above 0.15 gap, else mark 'none'. Prints a table and writes
runs/SUMMARY/ood_calibration.json. Does not edit tasks.py (the choice is applied by hand and logged in the changelog)."""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import envs, tasks
from wsarena.store import CheckpointStore
store = CheckpointStore(ROOT / "artifacts/checkpoints")
mf = [e for e in json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"] if e["status"] == "built"]
GRID = (0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)   # v2: low end refined after the first pass found 0.2 already below threshold for c01
def interp(pid, pood, s):
    kw = {}
    for f in ("mass", "damping", "fmax", "obstacle_radius"):
        a, b = getattr(pid, f), getattr(pood, f); kw[f] = a + s * (b - a)
    kw["delay"] = int(round(pid.delay + s * (pood.delay - pid.delay)))
    return replace(pid, **kw)
out = {}
for tid, t in tasks.TASKS.items():
    cks = [store.get(e["sha256"]) for e in mf if e["task_id"] == tid and e["class"] in ("S2", "S3")]
    seeds = [envs.seed_for(tid, "select", i) for i in range(200)]
    sr = lambda p, w: float(envs.rollout(p, envs.make_policy(w), seeds).success.mean())
    id_sr = np.mean([sr(t.params_id, w) for w in cks])
    ex = lambda p: float(envs.rollout(p, lambda o: tasks.expert_action(t.params_id, o), seeds).success.mean())
    rows = []
    for s in GRID:
        p = interp(t.params_id, t.params_ood, s)
        o = np.mean([sr(p, w) for w in cks]); rows.append({"s": s, "ood_sr": float(o), "gap": float(id_sr - o), "expert_ood": ex(p),
            "params": {k: getattr(p, k) for k in ("mass", "damping", "fmax", "obstacle_radius", "delay")}})
    ok = [r for r in rows if r["ood_sr"] >= 0.20 and r["gap"] >= 0.15]
    choice = max(ok, key=lambda r: r["s"]) if ok else None
    out[tid] = {"n_ckpts": len(cks), "id_sr": float(id_sr), "grid": rows, "choice_s": None if choice is None else choice["s"],
                "choice_params": None if choice is None else choice["params"]}
    print(tid, f"n={len(cks)} id_sr={id_sr:.2f}", [(r["s"], round(r["ood_sr"], 2), round(r["gap"], 2)) for r in rows], "->", out[tid]["choice_s"], flush=True)
(ROOT / "runs/SUMMARY/ood_calibration.json").write_text(json.dumps(out, indent=1))
