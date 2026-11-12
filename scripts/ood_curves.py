"""A7: OOD curves instead of a single point (post-hoc, CPU). lambda scales the calibrated OOD shift: params(lam) = ID + lam*(OOD - ID),
lam in {0 (=ID params), 0.5, 1 (=the calibrated OOD split), 1.5, 2}. Base and final weights of each run are evaluated on the SAME 100 'hid_ood' episodes.
usage: ood_curves.py runs/pilot-v0/a1 runs/pilot-v1/a1 runs/baselines-v1/a1 runs/baselines-gain-v1/a1 [...] -> runs/SUMMARY/ood-curves.csv
(Tier-0 point-mass tasks only: the shift is defined on their physical parameters.)"""
import json, sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.store import CheckpointStore
from wsarena.tasks import TASKS
st = CheckpointStore(ROOT / "artifacts/checkpoints"); LAM = (0.0, 0.5, 1.0, 1.5, 2.0); N = 100
def shifted(t, lam):
    a, b = t.params_id, t.params_ood; kw = {f: getattr(a, f) + lam * (getattr(b, f) - getattr(a, f)) for f in ("mass", "damping", "fmax", "obstacle_radius")}
    kw["delay"] = int(round(a.delay + lam * (b.delay - a.delay)))
    return replace(t, params_ood=replace(a, **kw))
rows = []; cache = {}
for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
    for rj in sorted(Path(d).glob("*/run.json")):
        j = json.loads(rj.read_text())
        if j.get("status") not in ("verified", "failed") or j["spec"]["task_id"] not in TASKS: continue
        t = TASKS[j["spec"]["task_id"]]; bh, fh = j["identity"]["base_sha256"], j["identity"]["final_sha256"]
        base, fin = st.get(bh), st.get(fh); arm = j["spec"]["run_id"].split("__")[2]
        for lam in LAM:
            tl = shifted(t, lam)
            key = (t.task_id, bh, lam)
            if key not in cache: cache[key] = float(tl.rollout(base, "hid_ood", 0, N, ood=True).success.mean())
            sf = cache[key] if fh == bh else float(tl.rollout(fin, "hid_ood", 0, N, ood=True).success.mean())
            rows.append({"exp": j["spec"]["exp_id"], "run_id": j["spec"]["run_id"], "arm": arm, "task": t.task_id, "class": j["spec"]["ckpt_class"], "ckpt": bh[:12], "lam": lam,
                         "sr_base": cache[key], "sr_final": sf, "d": sf - cache[key]})
df = pd.DataFrame(rows); df.to_csv(ROOT / "runs/SUMMARY/ood-curves.csv", index=False); print(len(df), "rows")
