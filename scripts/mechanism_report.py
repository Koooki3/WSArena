"""A1 report: classify the edit mechanism of every completed run in the given experiment dirs and cross-tabulate with the hidden gain.
  .venv/bin/python scripts/mechanism_report.py runs/pilot-v0/a1 runs/pilot-v1/a1 [--controls]
Writes runs/SUMMARY/mechanism-<tag>.csv (one row per run)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.analysis.mechanism import classify, ORDER
from wsarena.store import CheckpointStore
st = CheckpointStore(ROOT / "artifacts/checkpoints")
dirs = [a for a in sys.argv[1:] if not a.startswith("--")]; rows = []
for d in dirs:
    for rj in sorted(Path(d).glob("*/run.json")):
        j = json.loads(rj.read_text())
        if j.get("status") not in ("verified", "failed"): continue
        m = classify(st.get(j["identity"]["base_sha256"]), st.get(j["identity"]["final_sha256"]))
        n = j["spec"]["run_id"].split("__"); h = j["final"]["hidden"]
        rows.append({"exp": j["spec"]["exp_id"], "task": n[0], "ckpt": n[1], "arm": n[2], "class": j["spec"]["ckpt_class"], "mechanism": m["mechanism"],
                     "frac_changed": round(m["frac_changed"], 4), "tensors": ",".join(m["tensors"]), "d_id": h["id"]["delta_sr"], "d_ood": h["ood"]["delta_sr"],
                     "regression": h["id"]["regression"], "recovery": h["id"]["recovery"]})
df = pd.DataFrame(rows); tag = "-".join(Path(d).parts[-2] for d in dirs)[:60]
(ROOT / "runs/SUMMARY").mkdir(exist_ok=True); df.to_csv(ROOT / f"runs/SUMMARY/mechanism-{tag}.csv", index=False)
pd.set_option("display.width", 220)
print(df.groupby(["arm", "mechanism"]).agg(n=("d_id", "size"), dID=("d_id", "mean"), regr=("regression", "mean")).round(2).to_string())
