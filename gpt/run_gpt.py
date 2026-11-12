"""GPT-side planner and sequential executor; CLI defaults to a read-only dry run.

No shared runner, evaluator, prompt or quota code is monkey-patched. Runtime
execution remains disabled until explicitly authorized and online checks pass.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from wsarena.protocol import RunInterrupted, RunSpec, run_wsp
from wsarena.version import source_hash
from gpt import identity, quota
from gpt.codex_cli import CodexCLIGateway, MODELS, EFFORTS
from gpt.checkpoints import OverlayStore
from gpt.output_caps import BatchLedger, summarize, pause_reason

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    ckpt_id: str
    entry: dict
    arm: dict
    rep: int


def token(value, label):
    if not isinstance(value, str) or not TOKEN.fullmatch(value) or "__" in value:
        raise ValueError(f"invalid {label}")
    return value


def plan(cfg: dict, manifest: dict):
    token(cfg["exp_id"], "exp_id")
    if cfg.get("protocol", "WSP-Native") != "WSP-Native":
        raise ValueError("only the frozen Native track is implemented")
    for key in ("rounds", "eps_per_round", "initial_report_episodes", "hidden_n"):
        if isinstance(cfg.get(key), bool) or not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise ValueError(f"invalid {key}")
    cap = cfg.get("max_output_tokens")
    if "max_output_tokens" not in cfg or cap is not None or cfg.get("output_cap_policy") != "U14":
        raise ValueError("GPT CLI output policy must explicitly be U14 with max_output_tokens=null")
    arms, seen_keys = cfg["arms"], set()
    for arm in arms:
        key = token(arm["key"], "arm key")
        if not key.startswith("gpt-") or key in seen_keys:
            raise ValueError("GPT arm keys must be unique and start with gpt-")
        seen_keys.add(key)
        if arm["provider"] != "codex-cli" or arm["model"] not in MODELS or arm["effort"] not in EFFORTS:
            raise ValueError("unregistered GPT provider/model/effort")
        if arm.get("prompt_variant", "full") not in {"full", "blind", "shuffled"}:
            raise ValueError("unknown prompt variant")
        if arm.get("track", "native") != "native":
            raise ValueError("GPT Probe adapter not yet validated; refusing silent fallback to Native")
    ids = [cid for cell in cfg["cells"] for cid in cell["ckpt_ids"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("checkpoint list empty or duplicated")
    for cid in ids:
        if cid not in manifest or manifest[cid]["status"] != "built":
            raise ValueError(f"checkpoint unavailable: {cid}")
    reps = cfg.get("reps", 1)
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("invalid reps")
    result = []
    for rep in range(reps):
        for cid in ids:
            e = manifest[cid]
            for arm in arms:
                if rep >= arm.get("reps", reps):
                    continue
                rid = f"{e['task_id']}__{cid.split('/', 1)[1].replace('/', '-')}__{arm['key']}__r{rep}"
                if "/" in rid or "\\" in rid:
                    raise ValueError("invalid run path")
                result.append(PlannedRun(rid, cid, e, arm, rep))
    if cfg.get("order") == "class_interleave":
        result.sort(key=lambda x: (x.rep, x.entry["seed"], x.entry["class"], x.entry["task_id"], x.ckpt_id, arms.index(x.arm)))
    else:
        result.sort(key=lambda x: (x.rep, x.ckpt_id, arms.index(x.arm)))
    return result


def make_spec(item, cfg, attempt, config_sha, code):
    e, arm = item.entry, item.arm
    return RunSpec(run_id=item.run_id, exp_id=cfg["exp_id"], attempt=attempt, task_id=e["task_id"],
                   ckpt_sha256=e["sha256"], ckpt_class=e["class"], provider=arm["provider"],
                   model=arm["model"], effort=arm["effort"], protocol=cfg["protocol"], rounds=cfg["rounds"],
                   eps_per_round=cfg["eps_per_round"], initial_report_episodes=cfg["initial_report_episodes"],
                   hidden_n=cfg["hidden_n"], max_output_tokens=cfg["max_output_tokens"], rep=item.rep,
                   code_version=code, config_sha256=config_sha[:16], prompt_variant=arm.get("prompt_variant", "full"))


def execute_batch(root: Path, config_path: Path, attempt: str, *, gateway_factory,
                  quota_reader, policy: dict, max_runs: int, batch_id: str):
    """Explicit dependency injection enables offline integration tests with fakes.

    The command-line entry point never invokes this without --execute. Every
    resumed run must match its spec/identity before replaying stored results.
    """
    token(attempt, "attempt")
    token(batch_id, "batch_id")
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
        raise ValueError("a positive finite batch size is required")
    cfg = yaml.safe_load(config_path.read_text())
    manifest = {e["ckpt_id"]: e for e in json.loads((root/"configs/tasks/ckpt_manifest.json").read_text())["entries"]}
    items = plan(cfg, manifest)
    ident = identity.capture(root, config_path)
    config_sha = ident["config_sha256"]
    exp_dir = root / "runs" / cfg["exp_id"] / attempt
    # Bind the entire logical attempt, including previous checkpoints no longer
    # present in a supplied config; never mix old policy statistics with new inputs.
    for old in sorted(exp_dir.glob("*gpt-*")):
        bound = old/"gpt_identity.json"
        if (old/"spec.json").exists() or (old/"run.json").exists():
            if not bound.is_file():
                raise ValueError("previous attempt run lacks GPT identity")
            identity.bind(bound, ident)
            identity.verify_evidence(old)
        if (old/"run.json").exists():
            previous = json.loads((old/"run.json").read_text())
            if previous.get("status") in {"verified", "failed"}:
                if not (old/"output_cap.json").is_file():
                    raise ValueError("terminal run missing U14 accounting; preserve and review before resuming")
                accounting = json.loads((old/"output_cap.json").read_text())
                if any(accounting.get(k) != v for k, v in summarize(previous).items()):
                    raise ValueError("U14 accounting differs from the sealed terminal run")
    logical_batch = f"{cfg['exp_id']}/{attempt}/{batch_id}"
    caps = BatchLedger(exp_dir, logical_batch)
    for old in exp_dir.glob("*gpt-*/gpt_batch.json"):
        previous_batch = json.loads(old.read_text())
        if previous_batch["batch_id"] == logical_batch and previous_batch["max_runs"] != max_runs:
            raise ValueError("logical batch size changed; do not silently expand a resumed batch")
    qlog = root / "runs/QUOTA_LOG_gpt.tsv"
    done, outcomes = 0, []
    def safe_reading():
        try:
            return quota_reader()
        except (OSError, ValueError, TypeError, KeyError):
            return None
    quota.log(qlog, cfg["exp_id"], attempt, "batch_start", note=f"bounded to {max_runs} runs")
    try:
        for item in items:
            if caps.completed_runs >= max_runs:
                break
            path = exp_dir / item.run_id
            if not path.resolve().is_relative_to((root/"runs").resolve()):
                raise ValueError("run symlink escapes the allowed runs root")
            spec = make_spec(item, cfg, attempt, config_sha, source_hash())
            if (path/"spec.json").exists():
                saved = json.loads((path/"spec.json").read_text())
                if saved != spec.model_dump():
                    raise ValueError("saved spec differs; refusing replay")
                if not (path/"gpt_identity.json").exists():
                    raise ValueError("saved run lacks GPT identity")
            if (path/"gpt_identity.json").exists():
                identity.bind(path/"gpt_identity.json", ident)
            identity.verify_evidence(path)
            if (path/"security_abort.json").exists():
                return {"completed": done, "status": "security_blocked", "outcomes": outcomes}
            if (path/"run.json").exists():
                state = json.loads((path/"run.json").read_text()).get("status")
                if state in {"verified", "failed"}:
                    continue
            if item.arm["key"] in caps.paused:
                continue
            bound_batch = path/"gpt_batch.json"
            if bound_batch.exists() and json.loads(bound_batch.read_text())["batch_id"] != logical_batch:
                raise ValueError("interrupted run belongs to another logical batch")
            reading = safe_reading()
            allowed, why = quota.check(reading, policy)
            quota.log(qlog, cfg["exp_id"], attempt, "before_run", reading, why)
            if not allowed:
                return {"completed": done, "status": "quota_blocked", "reason": why, "outcomes": outcomes}
            gw = gateway_factory(item.arm)
            # Bind all inputs before the first model call, without putting any
            # identifying value into the editor's neutral workdir.
            identity.bind(path/"gpt_identity.json", ident)
            if not bound_batch.exists():
                with bound_batch.open("x", encoding="utf-8") as f:
                    json.dump({"batch_id": logical_batch, "max_runs": max_runs}, f)
            try:
                store = OverlayStore(root/"artifacts/checkpoints", path/"checkpoints")
                out = run_wsp(spec, gw, store, path)
                # The shared summarizer turns unavailable usage into zero.
                # Preserve unknown GPT measurements before sealing this run.
                for field in out["usage_total"]:
                    values = [r["usage"].get(field) for r in out["rounds"]]
                    out["usage_total"][field] = None if any(v is None for v in values) else sum(values)
                (path/"run.json").write_text(json.dumps(out, indent=1, sort_keys=True))
                caps.record(path, out, item.arm["key"])
            except RunInterrupted as ex:
                if (path/"rounds.json").exists():
                    partial = {"spec": spec.model_dump(), "rounds": json.loads((path/"rounds.json").read_text())}
                    stats = summarize(partial)
                    earlier = caps.totals.get(item.arm["key"], {"rounds": 0, "truncated": 0, "missing": 0})
                    if pause_reason(earlier["rounds"]+stats["rounds"], earlier["truncated"]+stats["truncated_rounds"], earlier["missing"]+stats["missing_output_token_rounds"]):
                        caps.record(path, partial, item.arm["key"], aborted=True)
                if "audit_violation" in str(ex):
                    (path/"security_abort.json").write_text(json.dumps({"status": "invalid", "reason": str(ex)}))
                quota.log(qlog, cfg["exp_id"], attempt, "interrupted", safe_reading(), str(ex))
                return {"completed": done, "status": "interrupted", "reason": str(ex), "outcomes": outcomes, "paused_arms": caps.paused}
            finally:
                identity.seal_evidence(path)
            outcomes.append({"run_id": item.run_id, "status": out["status"]})
            done += 1
            if out["status"] == "failed":
                return {"completed": done, "status": "audit_failed", "outcomes": outcomes}
            if done >= max_runs:
                break
        return {"completed": done, "status": "arms_paused" if caps.paused else "batch_complete", "outcomes": outcomes,
                "paused_arms": caps.paused}
    finally:
        quota.log(qlog, cfg["exp_id"], attempt, "batch_end", safe_reading(), f"{done} completed; no timer or retry")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    token(args.attempt, "attempt")
    cfg = yaml.safe_load(args.config.read_text())
    manifest = {e["ckpt_id"]: e for e in json.loads((ROOT/"configs/tasks/ckpt_manifest.json").read_text())["entries"]}
    items = plan(cfg, manifest)
    ident = identity.capture(ROOT, args.config)
    if args.execute:
        raise SystemExit("BLOCKED: offline preparation only; online authorization, login/isolation, tool-table and GPT quota verification remain pending. U14 removes the 64k cap blocker. No process launched.")
    print(json.dumps({"mode": "dry_run", "runs": len(items), "identity": ident,
                      "run_ids": [i.run_id for i in items], "model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
