"""Plan + execute an experiment config (resumable, quota-aware, sequential by default).

  python scripts/run_experiment.py configs/experiments/<exp>.yaml --attempt a1 [--max-runs N] [--quota-note "5h=35% wk=20%"] [--dry-run]

* Plan = arms x cells x reps, interleaved round-robin across arms (no arm gets the 'early quota').
* A run with run.json status verified/failed is DONE and never re-run (failures stay in the denominator).
* An interrupted run (rate limit / usage limit / auth / timeout) STOPS the whole invocation: nothing is retried,
  no timer is set; re-invoke later with the same --attempt and the saved turns are replayed (zero new model calls).
* Every invocation appends a QUOTA_LOG row before and after (readings are supplied by the user via --quota-note).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks
from wsarena.llm.claude_cli import ClaudeCLIGateway
from wsarena.llm.scripted import ScriptedGateway
from wsarena.protocol import RunInterrupted, RunSpec, run_wsp
from wsarena.store import CheckpointStore
from wsarena.version import source_hash
sys.path.insert(0, str(ROOT / "scripts")); import quota as _quota

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("--attempt", required=True)
ap.add_argument("--max-runs", type=int, default=0); ap.add_argument("--quota-note", default="")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--stop-5h", type=float, default=None, help="override configs/quota_limits.json: do not START a run when the account 5h utilization is at/above this")
ap.add_argument("--stop-week", type=float, default=None, help="same for the weekly utilization"); a = ap.parse_args()

cfg_text = Path(a.config).read_text(); cfg = yaml.safe_load(cfg_text)
cfg_sha = hashlib.sha256(cfg_text.encode()).hexdigest()[:16]; code = source_hash()
store = CheckpointStore(ROOT / "artifacts" / "checkpoints")
manifest = {e["ckpt_id"]: e for e in json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"]}

def gateway(arm, rep=0):
    if arm["provider"] == "claude-cli":
        gw = ClaudeCLIGateway()
        if arm.get("track") == "probe":          # P6: read-only probes via the gateway wrapper; the arm MUST also use prompt_variant: blind
            assert arm.get("prompt_variant") == "blind", "probe arms require prompt_variant: blind"
            from wsarena.llm.probe_gateway import ProbeGateway
            gw = ProbeGateway(gw, **arm.get("probe", {}))
        return gw
    if arm["provider"].startswith("scripted:"): return ScriptedGateway(arm["provider"].split(":", 1)[1], seed=arm.get("seed", 0) + rep, **arm.get("kw", {}))
    raise SystemExit(f"provider {arm['provider']!r} is not implemented in this checkout (GPT/Codex gateway lives in gpt/)")

if "select" in cfg:      # select: {tasks: [...], classes: [S1,S2,S3], kinds: [...], seeds: [1,2]}
    sel = cfg["select"]
    ids = [cid for cid, e in manifest.items() if e["status"] == "built"
           and e["task_id"] in sel.get("tasks", [e["task_id"]]) and e["class"] in sel.get("classes", [e["class"]])
           and e["kind"] in sel.get("kinds", [e["kind"]]) and e["seed"] in sel.get("seeds", [e["seed"]])]
    cfg["cells"] = [{"ckpt_ids": sorted(ids)}]
plan = []
for rep in range(cfg.get("reps", 1)):
    for cell in cfg["cells"]:
        for cid in cell["ckpt_ids"]:
            e = manifest.get(cid)
            if not e or e["status"] != "built": continue
            for arm in cfg["arms"]:
                if rep < arm.get("reps", cfg.get("reps", 1)):
                    plan.append((rep, cid, e, arm))
if cfg.get("order") == "class_interleave":    # any prefix of the plan spans tasks/classes/seeds (a stop at any point leaves cross-class data)
    plan.sort(key=lambda x: (x[0], x[2]["seed"], x[2]["class"], x[2]["task_id"], x[1], cfg["arms"].index(x[3])))
else:
    plan.sort(key=lambda x: (x[0], x[1], cfg["arms"].index(x[3])))      # arms adjacent => interleaved per checkpoint
exp_dir = ROOT / "runs" / cfg["exp_id"] / a.attempt
qlog = ROOT / "runs" / "QUOTA_LOG.tsv"
def log(event, extra=""):
    qlog.parent.mkdir(parents=True, exist_ok=True)
    if not qlog.exists(): qlog.write_text("time\texp\tattempt\tevent\tuser_reading\tnote\n")
    with qlog.open("a") as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S+08:00')}\t{cfg['exp_id']}\t{a.attempt}\t{event}\t{a.quota_note}\t{extra}\n")

todo = []
for rep, cid, e, arm in plan:
    rid = f"{e['task_id']}__{cid.split('/',1)[1].replace('/','-')}__{arm['key']}__r{rep}"
    rj = exp_dir / rid / "run.json"
    if rj.exists() and json.loads(rj.read_text()).get("status") in ("verified", "failed"): continue
    todo.append((rid, rep, cid, e, arm))
print(f"plan: {len(plan)} runs, {len(plan)-len(todo)} done, {len(todo)} to do")
if a.dry_run: [print(" ", t[0]) for t in todo]; sys.exit(0)
if a.max_runs: todo = todo[:a.max_runs]
log("batch_start", f"{len(todo)} runs")
done = 0
for rid, rep, cid, e, arm in todo:
    spec = RunSpec(run_id=rid, exp_id=cfg["exp_id"], attempt=a.attempt, task_id=e["task_id"], ckpt_sha256=e["sha256"],
                   ckpt_class=e["class"], provider=arm["provider"], model=arm["model"], effort=arm["effort"],
                   protocol=cfg.get("protocol", "WSP-Native"), rounds=cfg["rounds"], eps_per_round=cfg.get("eps_per_round", 16),
                   initial_report_episodes=cfg.get("initial_report_episodes", 16), hidden_n=cfg.get("hidden_n", 200),
                   max_output_tokens=cfg.get("max_output_tokens"), rep=rep, code_version=code, config_sha256=cfg_sha,
                   prompt_variant=arm.get("prompt_variant", "full"))
    t0 = time.time()
    L5, LW = _quota.limits(); L5 = a.stop_5h if a.stop_5h is not None else L5; LW = a.stop_week if a.stop_week is not None else LW
    q = _quota.latest()      # free: newest rate_limit_event in the saved streams (account-level, includes other sessions up to that call)
    if arm["provider"] == "claude-cli" and q and (q["five_hour"] >= L5 or q["seven_day"] >= LW):     # scripted controls use no quota
        log("quota_stop", f"before {rid}: 5h={q['five_hour']*100:.0f}% wk={q['seven_day']*100:.0f}% (limits {L5*100:.0f}/{LW*100:.0f}); 5h resets {q['five_hour_resets_at']}")
        print(f"QUOTA STOP before {rid}: 5h={q['five_hour']*100:.0f}% wk={q['seven_day']*100:.0f}%. Nothing started; resume later with the same --attempt.", flush=True); break
    try:
        out = run_wsp(spec, gateway(arm, rep), store, exp_dir / rid)
        h = out["final"]["hidden"]["id"]
        print(f"[{done+1}/{len(todo)}] {rid}: {out['status']} valid_rounds={out['valid_patch_rounds']}/{len(out['rounds'])} ({time.time()-t0:.0f}s)", flush=True)
        done += 1
    except RunInterrupted as ex:
        log("interrupted", f"{rid}: {ex}")
        print(f"STOP: {rid} interrupted ({ex}). Saved; resume with the same --attempt. Nothing was retried.", flush=True); break
q = _quota.latest(); log("batch_end", f"{done} completed" + (f"; auto 5h={q['five_hour']*100:.0f}% wk={q['seven_day']*100:.0f}%" if q else ""))
