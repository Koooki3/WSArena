"""Fair-comparison check: for every (task, checkpoint) present in BOTH provider arms of an attempt, the round-1 user
prompt must be byte-identical. usage: python scripts/check_prompt_parity.py runs/<exp>/<attempt>"""
import sys, hashlib, collections, json
from pathlib import Path
root = Path(sys.argv[1]); groups = collections.defaultdict(dict)
for p in root.glob("*/turns/01.prompt.txt"):
    task, ckpt, arm, rep = p.parent.parent.name.split("__")
    var = json.loads((p.parent.parent / "spec.json").read_text()).get("prompt_variant", "full")      # only same-variant prompts must match
    groups[(task, ckpt, rep, var)][arm] = hashlib.sha256(p.read_bytes()).hexdigest()
bad = [(k, v) for k, v in groups.items() if len(set(v.values())) > 1]
print(f"{len(groups)} (task,ckpt,rep) groups; {len(bad)} with differing round-1 prompts")
for k, v in bad[:10]: print(" MISMATCH", k, {a: h[:10] for a, h in v.items()})
sys.exit(1 if bad else 0)
