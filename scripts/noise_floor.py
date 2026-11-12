"""E1 noise scale (K-S3-05): SR of every built checkpoint on 5 mutually disjoint 200-episode sets (ID and OOD).
Evaluation is deterministic, so the spread is pure episode-sampling noise. Writes runs/SUMMARY/noise_floor.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks
from wsarena.store import CheckpointStore
store = CheckpointStore(ROOT / "artifacts/checkpoints")
mf = json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"]
rows = []
for e in mf:
    if e["status"] != "built": continue
    t = tasks.TASKS[e["task_id"]]; w = store.get(e["sha256"]); r = {"ckpt_id": e["ckpt_id"], "class": e["class"]}
    for name, ood in (("id", False), ("ood", True)):
        srs = [float(tasks.eval_weights(t, w, f"nf{k}", 200, ood=ood).success.mean()) for k in range(5)]
        r[f"sr_{name}"] = srs; r[f"mean_{name}"] = float(np.mean(srs)); r[f"sd_{name}"] = float(np.std(srs, ddof=1))
    rows.append(r)
sd = [r["sd_id"] for r in rows]
summ = {"n_ckpts": len(rows), "median_sd_id": float(np.median(sd)), "max_sd_id": float(np.max(sd)),
        "median_sd_ood": float(np.median([r["sd_ood"] for r in rows])), "binomial_bound": 0.0354}
out = ROOT / "runs/SUMMARY"; out.mkdir(parents=True, exist_ok=True)
(out / "noise_floor.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1)); print(summ)
