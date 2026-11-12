"""Completeness guard (P8 review F-S2): run before any analysis. Fails loudly if a main-study experiment is missing runs, has unverified runs,
or the Holm family would silently shrink. usage: check_completeness.py [t0|mw|all]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPECT = {"main-v1-t0": dict(runs=64, clusters=16, arms=4, holm=12), "main-v1-t0-rep": dict(runs=16), "main-v1-t0-blind": dict(runs=16), "main-v1-cap32k": dict(runs=8),
          "main-v1-mw": dict(runs=34, clusters=17, arms=2, holm=6), "main-v1-mw-rep": dict(runs=16)}
which = sys.argv[1] if len(sys.argv) > 1 else "all"; bad = 0
for exp, e in EXPECT.items():
    if which != "all" and not exp.endswith(which) and exp != f"main-v1-{which}": continue
    rs = [json.loads(p.read_text()) for p in sorted((ROOT / "runs" / exp / "a1").glob("*/run.json"))]
    st = [r.get("status") for r in rs]; ok = len(rs) == e["runs"] and all(s == "verified" for s in st)
    if "clusters" in e:
        ck = {r["spec"]["ckpt_sha256"] for r in rs}; ar = {r["spec"]["run_id"].split("__")[2] for r in rs}; ok &= len(ck) == e["clusters"] and len(ar) == e["arms"]
        ok &= 3 * e["arms"] == e["holm"] if e["arms"] != 2 else 3 * e["arms"] == e["holm"]
    print(f"{exp}: {len(rs)} runs, statuses {sorted(set(st))} -> {'OK' if ok else 'INCOMPLETE'}"); bad += (not ok)
sys.exit(1 if bad else 0)
