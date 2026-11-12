"""Tier-0 preregistered analysis and bounded online entry point."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import isfinite
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml

from gpt import identity
from gpt.account_limits import read_subscription_quota
from gpt.online import AuditedGateway, ROOT, gates
from gpt.parity import check_pairs
from gpt.run_gpt import execute_batch, plan, token
from wsarena.analysis.metrics import run_row
from wsarena.analysis.stats import holm, paired_diff_bootstrap, sign_flip_pvalue

CONFIG = ROOT / "gpt/configs/main-v1-t0.yaml"
BASELINES = {
    "random5": (("baselines-v0/a2", 3), ("baselines-v1/a1", 3)),
    "analytic_ridge": (("baselines-v0/a2", 1), ("baselines-v1/a1", 1)),
    "gain_greedy": (("baselines-gain-v1/a1", 1),),
    "tensor_gain_greedy": (("baselines-ctrl2-v1/a1", 1),),
    "greedy_f02_s01": (("baselines-greedy-v0/a1", 3), ("baselines-greedy-v1/a1", 3)),
}
PROFILE = ("online.py", "account_limits.py", "codex_cli.py", "wire_audit.py",
           "runtime.json", "model_catalog.json")
SMOKE_ATTEMPT = "native-recheck-20261112"


def planned():
    cfg = yaml.safe_load(CONFIG.read_text())
    manifest = json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"]
    items = plan(cfg, {e["ckpt_id"]: e for e in manifest})
    clusters = {i.entry["sha256"]: i.entry for i in items}
    arms = {i.arm["key"] for i in items}
    pairs = {(i.arm["key"], i.entry["sha256"]) for i in items}
    levels = [e["class"] for e in clusters.values()]
    if (len(items), len(pairs), len(arms), levels.count("S3"), len(clusters)) != (64, 64, 4, 6, 16):
        raise ValueError("Tier-0 requires exactly four arms on each of sixteen checkpoints")
    if set(levels) != {"S1", "S2", "S3"}:
        raise ValueError("unexpected Tier-0 checkpoint classes")
    return cfg, items, clusters


def score(path, entry, cfg):
    result = json.loads(path.read_text())
    sp = result["spec"]
    if result["status"] not in {"verified", "failed"}:
        raise ValueError(f"non-terminal run: {path}")
    for key, expected in {
        "task_id": entry["task_id"], "ckpt_sha256": entry["sha256"], "ckpt_class": entry["class"],
        **{k: cfg[k] for k in ("protocol", "rounds", "eps_per_round", "initial_report_episodes", "hidden_n")},
    }.items():
        if sp[key] != expected:
            raise ValueError(f"run identity mismatch for {key}: {path}")
    if sp.get("track", "native") != "native" or sp.get("prompt_variant", "full") != "full":
        raise ValueError(f"unexpected track or prompt: {path}")
    row = run_row(path)
    if not isfinite(row["d_sr_id"]):
        raise ValueError(f"invalid hidden-ID score: {path}")
    return row["d_sr_id"], result["status"], sp


def baseline_scores(cfg, clusters):
    l2 = {sha: e for sha, e in clusters.items() if e["class"] in {"S1", "S2"}}
    values, evidence = defaultdict(list), {}
    for key, sources in BASELINES.items():
        for source, repeats in sources:
            seen = defaultdict(set)
            for path in sorted((ROOT / "runs" / source).glob(f"*__{key}__r*/run.json")):
                sp = json.loads(path.read_text())["spec"]
                sha = sp["ckpt_sha256"]
                if sha not in l2:
                    continue
                value, status, sp = score(path, l2[sha], cfg)
                if status != "verified" or sp["rep"] in seen[sha] or path.parent.name != sp["run_id"]:
                    raise ValueError(f"unexpected baseline status or duplicate: {path}")
                seen[sha].add(sp["rep"])
                evidence[str(path.relative_to(ROOT))] = identity.sha(path)
                values[key, sha].append(value)
            if set(seen) != set(l2) or any(reps != set(range(repeats)) for reps in seen.values()):
                raise ValueError(f"incomplete baseline coverage: {key} in {source}")
    return {key: float(np.mean(v)) for key, v in values.items()}, evidence


def c3(rows):
    groups = [(g.loc[g.level == "L3", "value"].to_numpy(), g.loc[g.level == "L2", "value"].to_numpy())
              for _, g in rows.groupby("task_id", sort=True)]
    if not groups or any(not len(a) or not len(b) for a, b in groups):
        raise ValueError("C3 requires both levels in every task")
    rng = np.random.default_rng(0)
    task_diffs = [a.mean() - b.mean() for a, b in groups]
    boot, perm = [], []
    for a, b in groups:
        boot.append(rng.choice(a, (5000, len(a))).mean(axis=1) - rng.choice(b, (5000, len(b))).mean(axis=1))
        v = np.concatenate((a, b))
        shuffled = rng.permuted(np.tile(v, (5000, 1)), axis=1)
        perm.append(shuffled[:, :len(a)].mean(axis=1) - shuffled[:, len(a):].mean(axis=1))
    observed = float(np.mean(task_diffs))
    return {
        "mean_diff": observed,
        "cluster_weighted_diff": float(rows.loc[rows.level == "L3", "value"].mean() - rows.loc[rows.level == "L2", "value"].mean()),
        "ci95": np.percentile(np.mean(boot, axis=0), [2.5, 97.5]).tolist(),
        "p": float((1 + np.count_nonzero(np.abs(np.mean(perm, axis=0)) >= abs(observed) - 1e-12)) / 5001),
    }


def analyze(cfg, items, clusters, baseline, attempt):
    directory = ROOT / "runs" / cfg["exp_id"] / attempt
    expected = {i.run_id for i in items}
    extra = sorted(p.parent.name for p in directory.glob("*/run.json") if p.parent.name not in expected)
    if extra:
        raise ValueError(f"unplanned runs: {extra}")
    current = identity.capture(ROOT, CONFIG)
    rows, failed, incomplete = [], [], []
    for item in items:
        folder = directory / item.run_id
        path = folder / "run.json"
        if not path.is_file() or json.loads(path.read_text())["status"] == "interrupted":
            incomplete.append(item.run_id)
            continue
        identity.verify_evidence(folder)
        saved = json.loads((folder / "gpt_identity.json").read_text())
        identity.bind(folder / "gpt_identity.json", current)
        value, status, sp = score(path, item.entry, cfg)
        expected_spec = {k: item.arm[k] for k in ("model", "effort", "provider")}
        expected_spec.update(run_id=item.run_id, exp_id=cfg["exp_id"], attempt=attempt, rep=item.rep,
                             max_output_tokens=None, config_sha256=saved["config_sha256"][:16])
        if any(sp[k] != v for k, v in expected_spec.items()):
            raise ValueError(f"unexpected GPT run identity: {path}")
        if status == "failed":
            failed.append(item.run_id)
        rows.append(dict(arm=item.arm["key"], ckpt=item.entry["sha256"], value=value))
    if incomplete:
        raise ValueError(f"Tier-0 incomplete ({len(incomplete)} of 64): {incomplete}")
    parity = prompt_parity(cfg, attempt)
    if parity["checked"] != 64:
        raise ValueError("first-prompt parity requires all 64 runs")
    frame = pd.DataFrame(rows + [dict(arm=arm, ckpt=sha, value=v) for (arm, sha), v in baseline.items()])
    frame["task_id"] = frame.ckpt.map({sha:e["task_id"] for sha,e in clusters.items()})
    frame["level"] = frame.ckpt.map({sha:"L3" if e["class"] == "S3" else "L2" for sha,e in clusters.items()})
    l2 = frame[frame.level == "L2"]
    primary, secondary = {}, {}
    for arm in sorted({i.arm["key"] for i in items}):
        for key, ref in zip(("C1", "C2", "gain_greedy", "tensor_gain_greedy", "greedy_f02_s01"), BASELINES):
            sub = l2[l2.arm.isin([arm, ref])]
            pair = sub.pivot(index=["task_id", "ckpt"], columns="arm", values="value")
            if len(pair) != 10 or pair.isna().any().any():
                raise ValueError(f"incomplete L2 pairing: {arm} and {ref}")
            result = paired_diff_bootstrap(sub, "value", "arm", arm, ref)
            result.update(reference=ref, p=sign_flip_pvalue((pair[arm] - pair[ref]).to_numpy()))
            family = primary if key in {"C1", "C2"} else secondary
            family[f"{arm}|{key}"] = result
        primary[f"{arm}|C3"] = c3(frame[frame.arm == arm])
    for family in (primary, secondary):
        if len(family) != 12:
            raise ValueError("Holm family must contain exactly 12 tests")
        for key, p in holm({key: result["p"] for key, result in family.items()}).items():
            family[key]["p_holm"] = p
    return dict(primary=primary, secondary=secondary, failed_runs_retained=failed)


def prompt_parity(cfg, attempt):
    reference = ROOT / "runs/main-v1-t0/a1"
    directory = ROOT / "runs" / cfg["exp_id"] / attempt
    pairs = []
    for path in sorted(directory.glob("*/run.json")):
        if json.loads(path.read_text())["status"] not in {"verified", "failed"}:
            continue
        task, checkpoint, _, rep = path.parent.name.split("__")
        pairs.append((reference / f"{task}__{checkpoint}__sonnet55-medium__{rep}", path.parent))
    if not pairs:
        return {"checked": 0}
    result = check_pairs(pairs, output_cap_policy="U14")
    if not result["ok"]:
        raise ValueError(f"first-prompt parity failed: {result['errors']}")
    return {"checked": result["checked"], "output_cap_disclosures": len(result["output_cap_disclosures"])}


def launch(attempt, batch, cfg):
    smoke_root = ROOT / "runs/gpt-smoke-v1" / SMOKE_ATTEMPT
    smoke, = smoke_root.glob("*gpt-*")
    identity.verify_evidence(smoke)
    run = json.loads((smoke / "run.json").read_text())
    if run["status"] != "verified" or len(run["rounds"]) != 4:
        raise ValueError("verified four-round smoke required")
    saved = json.loads((smoke / "gpt_identity.json").read_text())
    current = identity.capture(ROOT, CONFIG)
    for name in PROFILE:
        key = f"gpt/{name}"
        if saved["files"][key] != current["files"][key]:
            raise ValueError(f"audited request profile changed: {key}")
    prompt_parity(cfg, attempt)
    runtime = json.loads((ROOT / "gpt/runtime.json").read_text())
    limits = json.loads((ROOT / "configs/quota_limits.json").read_text())
    policy = dict(stop_five_hour=limits["five_hour"], stop_week=limits["seven_day"],
                  authorization_source="用户要求沿用 Claude 端两项额度停线", max_age_s=300)
    gates(runtime, policy)
    os.environ["TMPDIR"] = runtime["neutral_tmp_root"]
    tempfile.tempdir = runtime["neutral_tmp_root"]
    result = execute_batch(ROOT, CONFIG, attempt, gateway_factory=lambda _: AuditedGateway(runtime, policy),
                           quota_reader=lambda: read_subscription_quota(runtime), policy=policy,
                           max_runs=8, batch_id=batch)
    print(json.dumps({**result, "prompt_parity": prompt_parity(cfg, attempt)}), flush=True)
    if result["status"] != "batch_complete":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run", "analyze"))
    parser.add_argument("--attempt")
    parser.add_argument("--batch")
    args = parser.parse_args()
    cfg, items, clusters = planned()
    baseline, evidence = baseline_scores(cfg, clusters)
    lock = ROOT / "gpt/tier0-baselines.json"
    if args.phase == "preflight":
        print(json.dumps(dict(runs=len(items), clusters=len(clusters), baseline_files=evidence), indent=2))
        return
    if identity.sha(lock) != cfg["baseline_lock_sha256"] or json.loads(lock.read_text()) != evidence:
        raise ValueError("baseline source hashes changed")
    attempt = token(args.attempt, "attempt")
    if cfg.get("status") != "preregistered":
        raise ValueError("Tier-0 protocol must be frozen before execution or analysis")
    if args.phase == "run":
        launch(attempt, token(args.batch, "batch"), cfg)
    else:
        print(json.dumps(analyze(cfg, items, clusters, baseline, attempt), indent=2))


if __name__ == "__main__":
    main()
