"""Source-tree identity (no git yet): sha256 over sorted (relative path, content) of wsarena/**/*.py + configs/**."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_hash() -> str:
    h = hashlib.sha256()
    files = sorted(list((ROOT / "wsarena").rglob("*.py")))
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]
