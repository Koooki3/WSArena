"""OOD calibration for official-benchmark tasks (same rule as scripts/calibrate_ood.py, 'select' namespace only):
choose the STRONGEST shift (lowest actuator gain) with mean OOD SR over S2 U S3 baseline checkpoints >= 0.20 and
mean (ID SR - OOD SR) >= 0.15. Run with the benchmark venv. usage: calibrate_official.py mw-push mw-pickplace
Writes runs/SUMMARY/ood_calibration_official.json; the choice is applied by hand to the task spec and logged."""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks
from wsarena.store import CheckpointStore
store = CheckpointStore(ROOT / "artifacts/checkpoints")
mf = [e for e in json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"] if e["status"] == "built"]
GRID = (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4)
N = 100
out = {}
for tid in sys.argv[1:]:
    t = tasks.get_task(tid)
    cks = [store.get(e["sha256"]) for e in mf if e["task_id"] == tid and e["class"] in ("S2", "S3")]
    id_sr = float(np.mean([t.rollout(w, "select", 0, N).success.mean() for w in cks]))
    rows = []
    for g in GRID:
        tg = replace(t, ood_gain=g)
        o = float(np.mean([tg.rollout(w, "select", 0, N, ood=True).success.mean() for w in cks]))
        rows.append({"gain": g, "ood_sr": o, "gap": id_sr - o}); print(tid, rows[-1], flush=True)
    ok = [r for r in rows if r["ood_sr"] >= 0.20 and r["gap"] >= 0.15]
    out[tid] = {"n_ckpts": len(cks), "id_sr": id_sr, "grid": rows, "choice_gain": min((r["gain"] for r in ok), default=None)}
    print(tid, "n", len(cks), "id_sr", round(id_sr, 2), "->", out[tid]["choice_gain"], flush=True)
(ROOT / "runs/SUMMARY/ood_calibration_official.json").write_text(json.dumps(out, indent=1))
