"""WSArena cleaner (FILE_MANAGEMENT_ZH.md §7). Default is --dry-run. NEVER touches permanent evidence:
docs/, configs/, wsarena/, tests/, scripts/, checks/, reports/, gpt/, survey/ (except registered external clones),
and EVERYTHING under runs/ (failed/interrupted runs are evidence). Deletion list + sizes are written to
runs/SUMMARY/clean-<time>.json BEFORE anything is removed.

classes: ephemeral = __pycache__, .pytest_cache; optional (flags): --include-venv, --include-external (only clones
registered in survey/sources/REPOS.tsv), --include-ckpts (regenerable S0-S3 checkpoints; rebuild must reproduce hashes).
"""
import argparse, json, shutil, sys, time
from pathlib import Path

PROTECTED = ("docs", "configs", "wsarena", "tests", "scripts", "checks", "reports", "gpt", "runs", "survey")


def size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def plan_clean(root: Path, include_venv=False, include_external=False, include_ckpts=False):
    root = root.resolve(); items = []
    for p in root.rglob("__pycache__"):
        if ".venv" in p.parts or "external" in p.parts: continue
        items.append((p, "ephemeral"))
    pc = root / ".pytest_cache"
    if pc.exists(): items.append((pc, "ephemeral"))
    if include_venv:
        for v in (root / ".venv", *sorted((root / "venvs").glob("*"))):
            if v.exists(): items.append((v, "optional:venv"))
    if include_external and (root / "external").exists():
        reg = root / "survey/sources/REPOS.tsv"
        names = {l.split("\t")[0] for l in reg.read_text().splitlines()[1:]} if reg.exists() else set()
        for d in (root / "external").iterdir():
            if d.name in names: items.append((d, "optional:external-registered"))
    if include_ckpts and (root / "artifacts/checkpoints").exists():
        items.append((root / "artifacts/checkpoints", "regenerable:checkpoints"))
    out = []
    for p, cls in items:
        rel = p.resolve().relative_to(root)                      # raises if outside root
        top = rel.parts[0]
        if top in PROTECTED and not (top == "survey" or top == "wsarena" or top == "scripts" or top == "tests" or top == "gpt"
                                     ) and "__pycache__" not in rel.parts:
            continue
        if top == "runs": continue                               # absolute rule
        if top in ("docs", "configs", "checks", "reports"): continue
        if top in ("wsarena", "scripts", "tests", "gpt", "survey") and p.name != "__pycache__": continue
        out.append({"path": str(p), "class": cls, "bytes": size(p)})
    return out


def apply_clean(root: Path, items):
    rec = root / "runs" / "SUMMARY"; rec.mkdir(parents=True, exist_ok=True)
    receipt = rec / f"clean-{time.strftime('%Y%m%d-%H%M%S')}.json"
    receipt.write_text(json.dumps({"items": items}, indent=1))   # written first
    for it in items:
        p = Path(it["path"])
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--include-venv", action="store_true"); ap.add_argument("--include-external", action="store_true")
    ap.add_argument("--include-ckpts", action="store_true"); a = ap.parse_args()
    items = plan_clean(Path(a.root), a.include_venv, a.include_external, a.include_ckpts)
    for it in items: print(f"{it['class']:32s} {it['bytes']/1e6:9.1f} MB  {it['path']}")
    print(f"total {sum(i['bytes'] for i in items)/1e6:.1f} MB in {len(items)} items ({'APPLY' if a.apply else 'dry-run'})")
    if a.apply and items: print("receipt:", apply_clean(Path(a.root), items))
