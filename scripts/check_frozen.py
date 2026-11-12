"""Verify (or regenerate) the frozen shared contract: files both Claude-side and GPT-side runs depend on.
  python scripts/check_frozen.py            # verify; exit 1 and list drift if anything changed
  python scripts/check_frozen.py --update   # Claude-side lead only, after a deliberate change + CONTRACT_CHANGELOG entry
Run it before starting any batch; a mismatch means the contract moved under you: STOP and read the changelog.
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FROZEN = ["wsarena/edits.py", "wsarena/mlp.py", "wsarena/envs.py", "wsarena/tasks.py", "wsarena/store.py",
          "wsarena/evaluation.py", "wsarena/prompts.py", "wsarena/protocol.py", "wsarena/llm/base.py",
          "wsarena/llm/conformance.py", "wsarena/llm/scripted.py", "wsarena/official/metaworld_task.py", "wsarena/official/ocbench_task.py", "wsarena/audit.py", "wsarena/analysis/mechanism.py", "wsarena/analysis/metrics.py", "wsarena/analysis/stats.py",
          "wsarena/probe.py", "wsarena/llm/probe_gateway.py",
          "configs/tasks/ckpt_manifest.json"]
OUT = ROOT / "docs/handoff/FROZEN_CONTRACT.json"
cur = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN if (ROOT / f).exists()}
if "--update" in sys.argv:
    prev = json.loads(OUT.read_text()) if OUT.exists() else {"version": 0}
    OUT.write_text(json.dumps({"version": prev.get("version", 0) + 1, "files": cur}, indent=1)); print("updated to version", prev.get("version", 0) + 1); sys.exit(0)
exp = json.loads(OUT.read_text())
bad = [f for f in FROZEN if exp["files"].get(f) != cur.get(f)]
print(f"contract version {exp['version']}: " + ("OK" if not bad else "DRIFT in " + ", ".join(bad)))
sys.exit(1 if bad else 0)
