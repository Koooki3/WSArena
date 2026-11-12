"""Bind GPT code and configuration in addition to the unchanged shared source hash."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(root: Path, config: Path):
    contract = root / "docs/handoff/FROZEN_CONTRACT.json"
    frozen = json.loads(contract.read_text())
    drift = [p for p, h in frozen["files"].items() if not (root/p).is_file() or sha(root/p) != h]
    if drift:
        raise ValueError("frozen contract drift: " + ", ".join(drift))
    files = {str(p.relative_to(root)): sha(p) for p in sorted((root/"gpt").rglob("*.py"))}
    files.update({str(p.relative_to(root)): sha(p) for p in sorted((root/"gpt/configs").glob("*.yaml"))})
    for name in ("model_catalog.json", "runtime.json"):
        p = root/"gpt"/name
        if p.is_file():
            files[str(p.relative_to(root))] = sha(p)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    policy = root/"docs/handoff/OUTPUT_CAP_POLICY_ZH.md"
    return {"schema": 1, "gpt_code_sha256": digest, "files": files, "git_head": head,
            "output_cap_policy_sha256": sha(policy) if policy.is_file() else None,
            "contract_version": frozen["version"], "contract_sha256": sha(contract),
            "checkpoint_manifest_sha256": sha(root/"configs/tasks/ckpt_manifest.json"),
            "config_sha256": sha(config)}


def bind(path: Path, identity: dict):
    if path.exists():
        previous = json.loads(path.read_text())
        # A later harmless Git commit is permitted; actual bound inputs must match.
        keys = ("gpt_code_sha256", "contract_sha256", "checkpoint_manifest_sha256", "config_sha256", "output_cap_policy_sha256")
        if any(previous.get(k) != identity.get(k) for k in keys):
            raise ValueError("resume identity differs; do not overwrite or replay under changed inputs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(identity, f, indent=2)


def evidence_files(run_dir: Path):
    paths = list((run_dir/"turns").glob("*")) + list((run_dir/"raw").glob("*"))
    if (run_dir/"spec.json").is_file():
        paths.append(run_dir/"spec.json")
    if (run_dir/"gpt_identity.json").is_file():
        paths.append(run_dir/"gpt_identity.json")
    if (run_dir/"gpt_batch.json").is_file():
        paths.append(run_dir/"gpt_batch.json")
    if (run_dir/"output_cap.json").is_file():
        paths.append(run_dir/"output_cap.json")
    if (run_dir/"output_cap_abort.json").is_file():
        paths.append(run_dir/"output_cap_abort.json")
    summary = run_dir/"run.json"
    if summary.is_file() and json.loads(summary.read_text()).get("status") in {"verified", "failed"}:
        paths.append(summary)
    return {str(p.relative_to(run_dir)): sha(p) for p in sorted(paths) if p.is_file()}


def verify_evidence(run_dir: Path):
    receipts = sorted(run_dir.glob("gpt_evidence.v*.json"))
    current = evidence_files(run_dir)
    if not current:
        return
    if not receipts:
        raise ValueError("unsealed prior turn evidence; preserve and review before resuming")
    previous = json.loads(receipts[-1].read_text())
    if previous["files"] != current or previous.get("rounds_sha256") != rounds_identity(run_dir)[0]:
        raise ValueError("saved turn evidence changed or is unsealed; refusing replay")


def rounds_identity(run_dir: Path):
    path = run_dir/"rounds.json"
    if not path.is_file():
        return None, []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("invalid saved rounds")
    hashes = [hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in data]
    return sha(path), hashes


def seal_evidence(run_dir: Path):
    receipts = sorted(run_dir.glob("gpt_evidence.v*.json"))
    files = evidence_files(run_dir)
    rounds_sha, round_hashes = rounds_identity(run_dir)
    if receipts:
        saved = json.loads(receipts[-1].read_text())
        previous = saved["files"]
        if any(files.get(k) != v for k, v in previous.items()):
            raise ValueError("existing turn evidence modified during run")
        prefix = saved.get("round_hashes", [])
        if round_hashes[:len(prefix)] != prefix:
            raise ValueError("saved rounds prefix was modified during replay")
        if files == previous and rounds_sha == saved.get("rounds_sha256"):
            return
    target = run_dir/f"gpt_evidence.v{len(receipts)+1:04d}.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump({"schema": 2, "files": files, "rounds_sha256": rounds_sha, "round_hashes": round_hashes}, f, indent=2)
