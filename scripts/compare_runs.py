"""Bit-for-bit regression check between two attempts of the same experiment (e.g. before/after a refactor).
Compares, for every common run-id: final/round checkpoint hashes, hidden-eval numbers (all fields), per-round patch status
and dev reports. Ignores code_version/attempt/config hash. Exit 1 on any difference. usage: compare_runs.py dirA dirB"""
import json, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2]); bad = 0; n = 0
def strip(j):
    return {"final_sha": j["identity"]["final_sha256"], "rounds_sha": j["identity"]["round_sha256"], "hidden": j["final"]["hidden"],
            "best": j["best_dev_track"], "rounds": [{k: r[k] for k in ("patch_status", "dev_report", "diff_vs_base")} for r in j["rounds"]],
            "initial": j["initial_report"], "loc": j["final"]["localization"]}
for ra in sorted(A.glob("*/run.json")):
    rb = B / ra.parent.name / "run.json"
    if not rb.exists(): continue
    n += 1; ja, jb = strip(json.loads(ra.read_text())), strip(json.loads(rb.read_text()))
    if ja != jb:
        bad += 1; keys = [k for k in ja if ja[k] != jb[k]]; print("DIFF", ra.parent.name, keys)
print(f"{n} runs compared, {bad} differ"); sys.exit(1 if bad else 0)
