"""Flatten run.json files into one row per run. Runs with status != verified are KEPT (flag only): failures stay in
the denominator; the summariser refuses runs whose identity fields are incomplete (K-S4-05)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REQUIRED_ID = ("base_sha256", "final_sha256", "code_version")


def run_row(path: Path) -> dict | None:
    j = json.loads(Path(path).read_text())
    if j.get("status") == "interrupted":
        return {"run_id": j["spec"]["run_id"], "status": "interrupted", "path": str(path)}
    ident, sp = j["identity"], j["spec"]
    miss = [k for k in REQUIRED_ID if not ident.get(k) or ident.get(k) == "unknown"]
    if miss:
        raise ValueError(f"{path}: identity incomplete {miss}")
    hid, rounds = j["final"]["hidden"], j["rounds"]
    best = j["best_dev_track"]["hidden"]
    loc = j["final"]["localization"] or {}
    cnt = lambda s: sum(r.get("patch_status") == s for r in rounds)
    n = len(rounds)
    return {
        "run_id": sp["run_id"], "exp_id": sp["exp_id"], "attempt": sp["attempt"], "task_id": sp["task_id"],
        "ckpt": ident["base_sha256"][:12], "ckpt_class": sp["ckpt_class"], "provider": sp["provider"],
        "model": sp["model"], "effort": sp["effort"], "rep": sp["rep"], "protocol": sp["protocol"],
        "status": j["status"], "n_rounds": n,
        "completion_rate": j["turns_ok"] / max(n, 1),
        "valid_patch_rate": j["valid_patch_rounds"] / max(n, 1),
        "n_no_block": cnt("no_block"), "n_all_skipped": cnt("all_skipped"),
        "n_budget_exhausted": cnt("turn_failed:output_budget_exhausted"),
        "n_no_output": cnt("turn_failed:no_output"), "n_refused": cnt("turn_failed:refused"),
        "sr_base_id": hid["id"]["sr_base"], "sr_final_id": hid["id"]["sr_edited"], "d_sr_id": hid["id"]["delta_sr"],
        "sr_base_ood": hid["ood"]["sr_base"], "sr_final_ood": hid["ood"]["sr_edited"], "d_sr_ood": hid["ood"]["delta_sr"],
        "recovery_id": hid["id"]["recovery"], "regression_id": hid["id"]["regression"],
        "d_sr_id_bestdev": best["id"]["delta_sr"], "best_dev_round": j["best_dev_track"]["round"],
        "frac_changed": j["final"]["diff_vs_base"]["frac_changed"], "rel_l2": j["final"]["diff_vs_base"]["rel_l2_delta"],
        "loc_precision": loc.get("precision"), "loc_recall": loc.get("recall"),
        "tok_in": j["usage_total"]["input_tokens"], "tok_out": j["usage_total"]["output_tokens"],
        "tok_think": j["usage_total"]["thinking_tokens"], "path": str(path),
    }


def load_runs(root: str | Path, pattern: str = "**/run.json") -> pd.DataFrame:
    rows = [run_row(p) for p in sorted(Path(root).glob(pattern))]
    return pd.DataFrame([r for r in rows if r])
