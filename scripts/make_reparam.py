"""A4: function-preserving reparameterised copies of selected checkpoints (tanh symmetries: unit permutation + sign flips, mlp.reparameterize).
For each source checkpoint: verify f_theta == f_T(theta) on 2000 random observations (max |diff| < 1e-9) AND identical hidden-ID/OOD success on 200
episodes; store T(theta) as an ordinary checkpoint (same class, kind '<kind>+reparam', no ground truth: indices moved) and register it in the manifest."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import mlp
from wsarena.edits import weights_sha256
from wsarena.store import CheckpointStore
from wsarena.tasks import get_task
SRC = ["c02-avoid/s1-weak_axis/seed1", "c02-avoid/s2/seed1", "c02-avoid/s3/seed1", "c03-hold/s2/seed1", "c03-hold/s3/seed1", "c04-delay/s1-weak_axis/seed1"]
st = CheckpointStore(ROOT / "artifacts/checkpoints"); mp = ROOT / "configs/tasks/ckpt_manifest.json"; man = json.loads(mp.read_text()); have = {e["ckpt_id"] for e in man["entries"]}
by = {e["ckpt_id"]: e for e in man["entries"]}
for cid in SRC:
    new_id = cid + "/reparam"
    if new_id in have: continue
    e = by[cid]; t = get_task(e["task_id"]); w = st.get(e["sha256"]); rng = np.random.default_rng(int(e["sha256"][:8], 16))
    w2 = mlp.reparameterize(w, rng); x = np.random.default_rng(0).normal(size=(2000, t.obs_dim))
    d = float(np.max(np.abs(mlp.forward(w, x) - mlp.forward(w2, x)))); assert d < 1e-9, (cid, d)
    s1 = [t.rollout(w, sp, 0, 200, ood=o).success.astype(int).tolist() for sp, o in (("hid_id", False), ("hid_ood", True))]
    s2 = [t.rollout(w2, sp, 0, 200, ood=o).success.astype(int).tolist() for sp, o in (("hid_id", False), ("hid_ood", True))]
    assert s1 == s2, f"{cid}: reparameterisation changed episode outcomes"
    meta = {"class": e["class"], "gt": None, "task_id": e["task_id"], "ckpt_id": new_id, "kind": e["kind"] + "+reparam", "seed": e["seed"], "reparam_of": e["sha256"], "max_abs_diff": d}
    h = st.put(w2, meta)
    man["entries"].append({"ckpt_id": new_id, "task_id": e["task_id"], "class": e["class"], "kind": e["kind"] + "+reparam", "seed": e["seed"], "status": "built", "sha256": h,
                           "select_sr": e.get("select_sr"), "gt_summary": None, "reparam_of": e["sha256"], "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00")})
    mp.write_text(json.dumps(man, indent=1)); print(new_id, "max|diff|=%.1e" % d, "identical outcomes on 400 hidden episodes", flush=True)
