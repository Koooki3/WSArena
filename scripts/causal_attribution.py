"""A5: patch causal attribution (post-hoc, CPU). For every completed run, per changed tensor k of the final patch:
  drop_if_reverted = SR_hid_ID(final) - SR_hid_ID(final with tensor k reset to base)      (necessity)
  gain_only        = SR_hid_ID(base with ONLY tensor k taken from final) - SR_hid_ID(base)  (sufficiency)
concentration = max_k gain_only / total gain (when total gain >= 0.05): near 1 => the gain lives in one tensor.
usage: causal_attribution.py runs/pilot-v0/a1 runs/pilot-v1/a1 [...]  -> runs/SUMMARY/causal-<tag>.csv"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.store import CheckpointStore
from wsarena.tasks import get_task
st = CheckpointStore(ROOT / "artifacts/checkpoints"); rows = []
for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
    for rj in sorted(Path(d).glob("*/run.json")):
        j = json.loads(rj.read_text())
        if j.get("status") not in ("verified", "failed"): continue
        t = get_task(j["spec"]["task_id"]); base = st.get(j["identity"]["base_sha256"]); fin = st.get(j["identity"]["final_sha256"])
        h = j["final"]["hidden"]["id"]; sb, sf = h["sr_base"], h["sr_edited"]; n = h["n"]; total = sf - sb
        ch = [k for k in base if (fin[k] != base[k]).any()]; per = []
        for k in ch:
            rev = {**fin, k: base[k]}; only = {**base, k: fin[k]}
            sr = lambda w: float(t.rollout(w, "hid_id", 0, n).success.mean())
            per.append((k, sf - sr(rev), sr(only) - sb, float((fin[k] != base[k]).mean())))
        conc = (max(p[2] for p in per) / total) if per and total >= 0.05 else None
        for k, drop, only, frac in per:
            rows.append({"run_id": j["spec"]["run_id"], "task": j["spec"]["task_id"], "class": j["spec"]["ckpt_class"], "tensor": k, "tensor_frac_changed": frac,
                         "drop_if_reverted": drop, "gain_only": only, "total_gain": total, "concentration": conc})
        if not per: rows.append({"run_id": j["spec"]["run_id"], "task": j["spec"]["task_id"], "class": j["spec"]["ckpt_class"], "tensor": "(none)", "total_gain": total})
df = pd.DataFrame(rows); tag = "-".join(Path(d).parts[-2] for d in sys.argv[1:] if not d.startswith("--"))[:60]
df.to_csv(ROOT / f"runs/SUMMARY/causal-{tag}.csv", index=False)
g = df.dropna(subset=["concentration"]).drop_duplicates("run_id"); g["arm"] = g.run_id.str.split("__").str[2]
pd.set_option("display.width", 200); print(g.groupby("arm").agg(n=("run_id", "size"), conc_med=("concentration", "median"), gain=("total_gain", "mean")).round(2).to_string())
