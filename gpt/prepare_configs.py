"""Generate GPT-only candidate configs from exact frozen checkpoint lists.

No model call, training, shared-file edit or preregistration freeze is performed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import yaml

ARMS = [{"key": f"gpt-{key}-{effort}", "provider": "codex-cli", "model": model, "effort": effort}
        for key, model in (("sol61", "gpt-6.1-sol"), ("astra6", "gpt-6-astra")) for effort in ("medium", "high")]
SOURCES = {
    "main-v1-t0": "main-v1-t0", "main-v1-t0-rep": "main-v1-t0-rep", "main-v1-t0-blind": "main-v1-t0-blind",
    "main-v1-mw": "main-v1-mw", "main-v1-mw-rep": "main-v1-mw-rep",
    "scale-v1-w64": "scale-v1-w64", "scale-v1-w128": "scale-v1-w128",
    "scale-v1-w64-blind": "scale-v1-w64-blind", "scale-v1-w128-blind": "scale-v1-w128-blind",
}


def generate(root: Path):
    configs = {}
    for name, source in SOURCES.items():
        path = root / f"configs/experiments/{source}.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["max_output_tokens"] = None
        cfg["output_cap_policy"] = "U14"
        variant = cfg["arms"][0].get("prompt_variant", "full")
        cfg["arms"] = copy.deepcopy(ARMS)
        if variant != "full":
            for arm in cfg["arms"]:
                arm["prompt_variant"] = variant
        cfg.update(exp_id=f"gpt-{name}", source_config=str(path.relative_to(root)),
                   source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status="candidate_not_preregistered")
        configs[name] = cfg
    base = configs["main-v1-t0"]
    for name, ids, arms in (
        ("smoke-v1", ["c01-reach/s1-weak_axis/seed1"], ARMS[:1]),
        ("pilot-v1", ["c02-avoid/s1-weak_axis/seed1", "c04-delay/s1-weak_axis/seed1"], ARMS),
        ("scale-v1-w24-blind-seed3", [f"c02-avoid/{kind}/seed3" for kind in ("s1-weak_axis", "s2", "s3")], ARMS),
    ):
        cfg = copy.deepcopy(base)
        cfg.update(exp_id=f"gpt-{name}", cells=[{"ckpt_ids": ids}], arms=copy.deepcopy(arms))
        if "blind" in name:
            for arm in cfg["arms"]:
                arm["prompt_variant"] = "blind"
        configs[name] = cfg
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    target = args.root/"gpt/configs"
    target.mkdir(parents=True, exist_ok=True)
    for name, cfg in generate(args.root).items():
        path = target/f"{name}.yaml"
        content = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            raise ValueError(f"existing GPT config differs; review before replacing: {path}")
        if not path.exists():
            path.write_text(content)
    print(json.dumps({"generated": len(generate(args.root)), "model_calls": 0}))


if __name__ == "__main__":
    main()
