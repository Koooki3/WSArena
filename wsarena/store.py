"""Content-addressed, immutable checkpoint store (.npz) with sha256 verification.

Layout: <root>/sha256/<hash[:2]>/<hash>.npz ; metadata sidecar <hash>.json (generation config, class, ground truth).
A checkpoint is never overwritten: put() of identical content is a no-op, and get() always re-verifies the hash.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .edits import weights_sha256


class IntegrityError(RuntimeError):
    pass


class CheckpointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _paths(self, h: str):
        d = self.root / "sha256" / h[:2]
        return d / f"{h}.npz", d / f"{h}.json"

    def put(self, weights: dict[str, np.ndarray], meta: dict | None = None) -> str:
        h = weights_sha256(weights)
        npz, js = self._paths(h)
        if npz.exists():
            return h
        npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: np.asarray(v, dtype="<f8") for k, v in weights.items()})
        os.replace(tmp, npz)
        os.chmod(npz, 0o444)
        if meta is not None:
            js.write_text(json.dumps(meta, indent=1, sort_keys=True, default=_jsonable))
            os.chmod(js, 0o444)
        return h

    def get(self, h: str) -> dict[str, np.ndarray]:
        npz, _ = self._paths(h)
        if not npz.exists():
            raise FileNotFoundError(h)
        with np.load(npz) as z:
            w = {k: z[k].astype(np.float64) for k in z.files}
        if weights_sha256(w) != h:
            raise IntegrityError(f"checkpoint {h[:12]} failed hash verification")
        return w

    def meta(self, h: str) -> dict:
        _, js = self._paths(h)
        return json.loads(js.read_text()) if js.exists() else {}

    def has(self, h: str) -> bool:
        return self._paths(h)[0].exists()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
