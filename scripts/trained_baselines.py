"""Run the 'Trained' reference track on all Tier-0 checkpoints and evaluate on the SAME hidden seeds as the LLM arms.
  .venv/bin/python scripts/trained_baselines.py            -> runs/SUMMARY/trained-baselines.csv  (one row per checkpoint x trainer)"""
import json, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import evaluation, trainers
from wsarena.edits import weights_sha256
from wsarena.store import CheckpointStore
from wsarena.tasks import get_task
st = CheckpointStore(ROOT / "artifacts/checkpoints")
mf = [e for e in json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"]
      if e["status"] == "built" and e["class"] in ("S1", "S2", "S3") and e["task_id"].startswith("c0") and e["seed"] in (1, 2, 3)]
CFG = {"bc_ft_small": lambda t, w, s: trainers.bc_finetune(t, w, 10, 50, s), "bc_ft_med": lambda t, w, s: trainers.bc_finetune(t, w, 50, 150, s),
       "bc_ft_full": lambda t, w, s: trainers.bc_finetune(t, w, 300, 300, s), "bc_scratch": lambda t, w, s: trainers.bc_finetune(t, w, 300, 600, s, scratch=True),
       "es_80": lambda t, w, s: trainers.es_finetune(t, w, 80, s), "es_800": lambda t, w, s: trainers.es_finetune(t, w, 800, s)}
rows = []; cache = {}
for e in mf:
    t = get_task(e["task_id"]); base = st.get(e["sha256"])
    for name, fn in CFG.items():
        w, led = fn(t, base, 1000 * e["seed"] + 7)
        h = evaluation.hidden_eval(t, w, base, 200, cache=cache)
        rows.append({"ckpt_id": e["ckpt_id"], "task": e["task_id"], "class": e["class"], "trainer": name, **led,
                     "sr_base_id": h["id"]["sr_base"], "sr_id": h["id"]["sr_edited"], "d_id": h["id"]["delta_sr"], "d_ood": h["ood"]["delta_sr"],
                     "recovery": h["id"]["recovery"], "regression": h["id"]["regression"]})
    print(e["ckpt_id"], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "runs/SUMMARY/trained-baselines.csv", index=False); print("done", len(rows))
