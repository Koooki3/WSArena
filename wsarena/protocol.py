"""WSP-8 protocol state machine (server-side enforcement; nothing is left to the editor's good behaviour).

  fixed task + fixed initial checkpoint
  initial report (dev episodes) -> for r in 1..R: [editor reply -> deterministic edit application -> dev evaluation]
  final-round checkpoint frozen -> hidden ID + OOD evaluation (never shown to the editor) -> run.json

Failures stay in the denominator (T4): output-budget exhaustion, no edit block, all statements skipped, and refusals are
recorded per round and are never retried. Provider-side limits (rate limit / usage limit / auth / timeout) INTERRUPT the
run: it is saved, not scored, and resumed later by replaying the saved turns (no repeated model calls).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from . import edits, evaluation, prompts
from .edits import apply_edits, diff_stats, extract_last_edits_block, weights_sha256
from .evaluation import Budget
from .llm.base import Gateway, Message, TurnRequest, TurnResult
from .audit import audit_run
from .store import CheckpointStore
from .tasks import get_task, TaskSpec


_BASE_CACHE: dict = {}      # in-process memo of deterministic base-checkpoint hidden evaluations


class RunInterrupted(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class RunSpec(BaseModel):
    run_id: str
    exp_id: str
    attempt: str
    task_id: str
    ckpt_sha256: str
    ckpt_class: str
    provider: str
    model: str
    effort: str
    protocol: str = "WSP-Native"
    rounds: int = 8
    eps_per_round: int = 16
    initial_report_episodes: int = 16
    hidden_n: int = 200
    max_output_tokens: int | None = None
    rep: int = 0
    code_version: str = "unknown"
    config_sha256: str = "unknown"
    prompt_variant: str = "full"          # A3: full | blind | shuffled (the weights block only)


def _wjson(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def localization(base, final, gt: dict | None) -> dict | None:
    """Element-level precision/recall of the edit against the planted-corruption ground truth (S1 only)."""
    if not gt or "tensors" not in gt:      # S0/S2/S3 have no planted-corruption ground truth
        return None
    changed = set()
    for k in base:
        d = final[k] != base[k]
        for idx in zip(*np.nonzero(d)):
            changed.add((k, tuple(int(i) for i in idx)))
    truth = {(k, tuple(ix)) for k, lst in gt["tensors"].items() for ix in lst}
    tp = len(changed & truth)
    prec = tp / len(changed) if changed else None
    rec = tp / len(truth) if truth else None
    return {"n_changed": len(changed), "n_truth": len(truth), "tp": tp, "precision": prec, "recall": rec}


def run_wsp(spec: RunSpec, gateway: Gateway, store: CheckpointStore, run_dir: str | Path,
            system: str = prompts.SYSTEM_PROMPT) -> dict:
    run_dir = Path(run_dir)
    task = get_task(spec.task_id)
    base = store.get(spec.ckpt_sha256)                          # re-verifies the hash (K-S4-05)
    meta = store.meta(spec.ckpt_sha256)
    gt = meta.get("gt")
    run_dir.mkdir(parents=True, exist_ok=True)
    _wjson(run_dir / "spec.json", spec.model_dump())
    budget = Budget(rounds=spec.rounds, episodes_per_round=spec.eps_per_round,
                    initial_report_episodes=spec.initial_report_episodes)

    cur = {k: v.copy() for k, v in base.items()}
    init_report = evaluation.dev_report(task, cur, 0, spec.initial_report_episodes, 0, spec.rounds, with_trajectory=True)
    budget.charge_episodes(spec.initial_report_episodes)
    messages = [Message(role="user", content=prompts.build_first_user_message(
        task, base, init_report, spec.rounds, spec.eps_per_round,
        weights_view=spec.prompt_variant, shuffle_seed=int(spec.ckpt_sha256[:8], 16)))]
    rounds_rec: list[dict] = []
    ckpts = {0: weights_sha256(cur)}
    best = (init_report["success_rate"], 0, {k: v.copy() for k, v in cur.items()})
    turns_dir = run_dir / "turns"
    raw_dir = run_dir / "raw"
    workdir = Path(tempfile.mkdtemp(prefix="w"))        # neutral, empty, random short name (the editor can see its path)
    interrupted = None

    for r in range(1, spec.rounds + 1):
        tag = f"{r:02d}"
        res_path, reply_path = turns_dir / f"{tag}.result.json", turns_dir / f"{tag}.reply.txt"
        if hasattr(gateway, "set_context"):
            gateway.set_context(dict(weights=cur, base=base, task=task, round=r, run_id=spec.run_id, meta=meta,
                                     dev_sr=[init_report["success_rate"]] + [x["dev_report"]["success_rate"] for x in rounds_rec],
                                     round_weights=[base] + [store.get(ckpts[i]) for i in range(1, r)]))
        if res_path.exists():                                   # resume: replay saved turn, no model call
            result = TurnResult.model_validate_json(res_path.read_text())
        else:
            req = TurnRequest(run_id=spec.run_id, turn_idx=r, model=spec.model, effort=spec.effort, system=system,
                              messages=messages, max_output_tokens=spec.max_output_tokens, workdir=str(workdir), raw_dir=str(raw_dir))
            (turns_dir).mkdir(parents=True, exist_ok=True)
            (turns_dir / f"{tag}.prompt.txt").write_text(messages[-1].content)
            result = gateway.complete(req)
            if result.status in ("rate_limited", "timeout") or (result.status == "error"):
                interrupted = (result.status, result.error)
                _wjson(run_dir / "interrupted.json", {"round": r, "status": result.status, "error": result.error})
                break
            reply_path.write_text(result.text)
            _wjson(res_path, result.model_dump())
        text = reply_path.read_text() if reply_path.exists() else result.text

        rec = {"round": r, "turn_status": result.status, "usage": result.usage.model_dump(),
               "tool_free": result.audit.tool_free, "latency_s": result.latency_s, "stop_reason": result.stop_reason}
        block = extract_last_edits_block(text) if result.status == "ok" else None
        if result.status != "ok":
            rec["patch_status"] = f"turn_failed:{result.status}"
            esum = {"block_found": False, "note": f"no usable reply ({result.status})"}
        elif block is None:
            rec["patch_status"] = "no_block"
            esum = {"block_found": False}
        else:
            er = apply_edits(cur, block, reset_to=base)
            esum = er.summary()
            if er.n_applied == 0 and er.n_total > 0:
                rec["patch_status"] = "all_skipped"
            elif er.n_total == 0:
                rec["patch_status"] = "empty_block"
            else:
                new_w = er.new_weights
                changed = weights_sha256(new_w) != weights_sha256(cur)
                rec["patch_status"] = "valid_changed" if changed else "valid_noop"
                rec["diff_vs_prev"] = {k: v for k, v in diff_stats(cur, new_w).items() if k != "per_tensor"}
                cur = new_w
            rec["edit_kinds"] = {k: sum(1 for s in er.statements if s.applied and s.kind == k)
                                 for k in ("element", "range", "scale", "reset")}
        rec["edit_summary"] = esum
        start = budget.initial_report_episodes + (r - 1) * spec.eps_per_round
        budget.charge_episodes(spec.eps_per_round)
        rep = evaluation.dev_report(task, cur, start, spec.eps_per_round, r, spec.rounds - r)
        rec["dev_report"] = rep
        rec["diff_vs_base"] = {k: v for k, v in diff_stats(base, cur).items() if k != "per_tensor"}
        ckpts[r] = store.put(cur, {"run_id": spec.run_id, "round": r, "parent_base": spec.ckpt_sha256})
        if rep["success_rate"] > best[0]:
            best = (rep["success_rate"], r, {k: v.copy() for k, v in cur.items()})
        rounds_rec.append(rec)
        messages.append(Message(role="assistant", content=text if result.status == "ok" else "(no usable reply)"))
        messages.append(Message(role="user", content=prompts.build_round_message(rep, esum, r + 1, spec.rounds)))
        _wjson(run_dir / "rounds.json", rounds_rec)

    shutil.rmtree(workdir, ignore_errors=True)
    if interrupted:
        out = {"status": "interrupted", "interrupted": interrupted, "rounds_done": len(rounds_rec)}
        _wjson(run_dir / "run.json", {"spec": spec.model_dump(), **out})
        raise RunInterrupted(*interrupted)

    final_hash = weights_sha256(cur)
    hidden = evaluation.hidden_eval(task, cur, base, spec.hidden_n, cache=_BASE_CACHE)
    best_hidden = hidden if best[1] == spec.rounds else evaluation.hidden_eval(task, best[2], base, spec.hidden_n, cache=_BASE_CACHE)
    usage_tot = {k: sum((r["usage"].get(k) or 0) for r in rounds_rec)
                 for k in ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens")}
    n_ok = sum(r["turn_status"] == "ok" for r in rounds_rec)
    tool_free_all = all(r["tool_free"] for r in rounds_rec)
    leak = audit_run(run_dir)                                      # A6: identity-leak audit of prompts and raw provider events
    status = "verified" if (tool_free_all and leak["ok"]) else "failed"   # a tool call or an identity leak invalidates the run (K-S5-01)
    out = {
        "spec": spec.model_dump(),
        "status": status,
        "invalid_reason": None if (tool_free_all and leak["ok"]) else ("tool_use_detected" if not tool_free_all else "identity_leak"),
        "leak_audit": leak,
        "identity": {"base_sha256": spec.ckpt_sha256, "final_sha256": final_hash, "round_sha256": ckpts,
                     "task_class": meta.get("class"), "code_version": spec.code_version},
        "rounds": rounds_rec,
        "turns_ok": n_ok, "turns_total": len(rounds_rec),
        "valid_patch_rounds": sum(r.get("patch_status") == "valid_changed" for r in rounds_rec),
        "usage_total": usage_tot,
        "final": {"hidden": hidden, "diff_vs_base": diff_stats(base, cur),
                  "localization": localization(base, cur, gt)},
        "best_dev_track": {"round": best[1], "dev_sr": best[0], "hidden": best_hidden},
        "initial_report": init_report,
    }
    for k in ("id", "ood"):                                       # drop per-episode arrays from the summary copy
        pass
    _wjson(run_dir / "run.json", out)
    return out
