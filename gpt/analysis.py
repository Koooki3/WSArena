"""Explicit data selection and checkpoint aggregation; no hidden directory globbing.

This is a descriptive data preparation command, not an implementation of a new
confirmatory hypothesis test. Primary C1/C2/C3 rules still need GPT preregistration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsarena.analysis.metrics import run_row
from gpt.identity import verify_evidence
from gpt.output_caps import summarize


def collect(directories, *, providers=("codex-cli",), arms=None, require_gpt_identity=True):
    seen, groups, missing, failed, caps = set(), {}, [], [], []
    allowed = set(arms) if arms is not None else None
    for directory in directories:
        for path in sorted(Path(directory).glob("*/run.json")):
            resolved = path.resolve()
            if resolved in seen:
                raise ValueError("same run included more than once")
            seen.add(resolved)
            j = json.loads(path.read_text())
            sp = j["spec"]
            key = sp["run_id"].split("__")[2]
            if sp["provider"] not in providers or (allowed is not None and key not in allowed):
                continue
            if require_gpt_identity and sp["provider"] == "codex-cli" and not (path.parent/"gpt_identity.json").is_file():
                raise ValueError(f"missing GPT identity: {path}")
            if require_gpt_identity and sp["provider"] == "codex-cli":
                verify_evidence(path.parent)
            if j["status"] not in {"verified", "failed"}:
                missing.append({"path": str(path), "status": j["status"]})
                continue
            row = run_row(path)
            caps.append({"path": str(path), **summarize(j)})
            if row["status"] == "failed":
                failed.append(str(path))
            group = (sp["model"], sp["effort"], sp.get("prompt_variant", "full"), sp["task_id"], sp["ckpt_sha256"])
            groups.setdefault(group, []).append(row["d_sr_id"])
    rows = [{"model": k[0], "effort": k[1], "prompt_variant": k[2], "task_id": k[3], "checkpoint": k[4],
             "n_runs": len(v), "mean_d_sr_id": sum(v)/len(v)} for k, v in sorted(groups.items())]
    return {"checkpoint_means": rows, "incomplete_runs": missing, "failed_runs_retained": failed,
            "output_cap_records": caps,
            "output_cap_comparison_disclosure": "Requested/effective caps are not matched under U14; do not generalize Claude-only sensitivity evidence to GPT.",
            "n_run_files": len(seen), "inferential_unit": "checkpoint; repeats averaged within model/effort/variant/checkpoint"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directories", nargs="+", type=Path, help="explicit attempt directories, include repeat directories explicitly")
    args = ap.parse_args()
    print(json.dumps(collect(args.directories), indent=2))


if __name__ == "__main__":
    main()
