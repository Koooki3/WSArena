"""Cross-directory first-prompt parity, with explicit expected pair coverage."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def inspect(run_dir: Path):
    spec = json.loads((run_dir/"spec.json").read_text())
    prompt = run_dir/"turns/01.prompt.txt"
    return {"spec": spec, "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()}


def check_pairs(pairs, *, output_cap_policy=None):
    if not pairs:
        return {"ok": False, "errors": ["no expected pairs supplied"], "checked": 0}
    errors, disclosures = [], []
    fields = ("task_id", "ckpt_sha256", "protocol", "rounds", "eps_per_round", "initial_report_episodes", "hidden_n")
    seen = set()
    for left, right in pairs:
        pair = (str(Path(left).resolve()), str(Path(right).resolve()))
        if pair in seen or pair[0] == pair[1]:
            errors.append("duplicate or self pair")
            continue
        seen.add(pair)
        try:
            l, r = inspect(Path(left)), inspect(Path(right))
        except (OSError, ValueError) as ex:
            errors.append(f"missing pair evidence: {ex}")
            continue
        for field in fields:
            if field not in l["spec"] or field not in r["spec"] or l["spec"][field] != r["spec"][field]:
                errors.append(f"spec mismatch: {field}: {pair}")
        caps = [l["spec"].get("max_output_tokens"), r["spec"].get("max_output_tokens")]
        if caps[0] != caps[1]:
            if output_cap_policy != "U14":
                errors.append(f"output cap requests differ without explicit U14 disclosure: {pair}")
            else:
                disclosures.append({"pair": pair, "requested_caps": caps, "effective_caps": "unverified", "policy": "U14"})
        if l["spec"].get("prompt_variant", "full") != r["spec"].get("prompt_variant", "full"):
            errors.append(f"prompt variant mismatch: {pair}")
        if l["spec"].get("provider") == r["spec"].get("provider"):
            errors.append(f"not a cross-provider pair: {pair}")
        if l["prompt_sha256"] != r["prompt_sha256"]:
            errors.append(f"prompt bytes mismatch: {pair}")
    return {"ok": not errors, "errors": errors, "checked": len(pairs), "output_cap_disclosures": disclosures,
            "cap_parity_established": False if output_cap_policy == "U14" else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pairs", type=Path, help="JSON list of [Claude run directory, GPT run directory] pairs")
    ap.add_argument("--output-cap-policy", choices=["U14"])
    args = ap.parse_args()
    result = check_pairs(json.loads(args.pairs.read_text()), output_cap_policy=args.output_cap_policy)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
