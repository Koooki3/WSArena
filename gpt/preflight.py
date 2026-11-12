"""Offline-only runtime report. Does not perform auth status, prompts or probes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt.codex_cli import sha256_file
from gpt.identity import capture
from gpt.prepare_configs import generate
from gpt.run_gpt import plan


def check(root: Path, binary: Path, scratch: Path):
    scratch.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(scratch), "CODEX_HOME": str(scratch/"cli-home"),
           "TMPDIR": str(scratch), "LANG": "C.UTF-8"}
    version = subprocess.run([str(binary), "--version"], cwd=scratch, env=env, capture_output=True, text=True, timeout=30)
    help_result = subprocess.run([str(binary), "exec", "--help"], cwd=scratch, env=env, capture_output=True, text=True, timeout=30)
    if version.returncode or help_result.returncode:
        raise RuntimeError("native CLI offline help/version failed")
    flags = ["--strict-config", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--json"]
    missing = [f for f in flags if f not in help_result.stdout]
    man = {e["ckpt_id"]: e for e in json.loads((root/"configs/tasks/ckpt_manifest.json").read_text())["entries"]}
    cfgs = generate(root)
    counts = {name: len(plan(cfg, man)) for name, cfg in cfgs.items()}
    ident = capture(root, root/"gpt/configs/main-v1-t0.yaml")
    return {"offline_engineering": "pass" if not missing else "failed", "native_binary": str(binary),
            "binary_sha256": sha256_file(binary), "cli_version": version.stdout.strip(), "missing_flags": missing,
            "candidate_config_run_counts": counts, "total_candidate_runs": sum(counts.values()),
            "identity": ident, "live_ready": False,
            "output_cap_policy": "U14; CLI default, ceiling unverified; observed maxima are not ceilings",
            "live_blockers": ["No online authorization/login/isolation or actual tool-table verification",
                              "No actual output-token smoke validation or GPT quota policy"],
            "model_calls": 0, "auth_files_read": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[1], args.binary.resolve(), args.scratch.resolve()), indent=2))


if __name__ == "__main__":
    main()
