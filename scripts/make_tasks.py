"""Build Tier-0 checkpoints (S0..S3) into artifacts/checkpoints and write configs/tasks/ckpt_manifest.json.

Deterministic given seeds. Infeasible combinations are recorded (never silently dropped): the band-selection
(SR on the 'select' namespace) simply did not hit its target for that task/kind/seed.
usage: python scripts/make_tasks.py [--seeds 1 2 3] [--tasks c01-reach ...]
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks
from wsarena.store import CheckpointStore
from wsarena.version import source_hash

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--tasks", nargs="+", default=list(tasks.TASKS))
ap.add_argument("--only-classes", nargs="+", default=None, help="build only these classes, e.g. S3")
ap.add_argument("--replace-classes", nargs="+", default=[], help="drop existing manifest entries of these classes first (old ones stay in the store)")
a = ap.parse_args()
store = CheckpointStore(ROOT / "artifacts" / "checkpoints")
mf_path = ROOT / "configs" / "tasks" / "ckpt_manifest.json"
manifest = json.loads(mf_path.read_text()) if mf_path.exists() else {"entries": []}
# replacement is scoped to the tasks being built (an unscoped replace once risked dropping other tasks' entries)
manifest["entries"] = [e for e in manifest["entries"] if not (e["class"] in a.replace_classes and e["task_id"] in a.tasks)]
have = {e["ckpt_id"] for e in manifest["entries"]}
want = lambda c: a.only_classes is None or c in a.only_classes
sv = source_hash()

def record(cid, task, cls, kind, seed, ck):
    if cid in have: return
    e = {"ckpt_id": cid, "task_id": task.task_id, "class": cls, "kind": kind, "seed": seed,
         "built_with_source": sv, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime())}
    if ck is None:
        e.update(status="infeasible", reason="target SR band not reached by the construction search")
    else:
        meta = dict(ck["meta"]); meta.update(task_id=task.task_id, ckpt_id=cid, kind=kind, seed=seed)
        e.update(status="built", sha256=store.put(ck["weights"], meta),
                 select_sr=meta.get("select_sr", meta.get("select_sr_after", 0.0)),
                 gt_summary=(meta.get("gt") or {}).get("kind") or (meta.get("gt") or {}).get("mode"))
    manifest["entries"].append(e); have.add(cid)
    mf_path.write_text(json.dumps(manifest, indent=1)); print(cid, e["status"], e.get("select_sr"), flush=True)

for tid in a.tasks:
    task = tasks.get_task(tid)
    if want("S0"): record(f"{tid}/s0/seed0", task, "S0", "zero", 0, tasks.build_s0(task))
    for sd in a.seeds:
        if want("S2") and f"{tid}/s2/seed{sd}" not in have: record(f"{tid}/s2/seed{sd}", task, "S2", "natural", sd, tasks.build_s2(task, sd))
        if want("S3") and f"{tid}/s3/seed{sd}" not in have: record(f"{tid}/s3/seed{sd}", task, "S3", "healthy", sd, tasks.build_s3(task, sd))
        for k in tasks.CORRUPTIONS:
            if want("S1") and f"{tid}/s1-{k}/seed{sd}" not in have: record(f"{tid}/s1-{k}/seed{sd}", task, "S1", k, sd, tasks.build_s1(task, sd, k))
print("done")
