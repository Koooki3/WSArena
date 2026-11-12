"""A1: edit-mechanism classifier. Deterministic, from (base weights, final weights) only. Thresholds are FROZEN with the preregistration.

  noop           nothing changed
  rewrite        > 50% of all parameters changed (a new network was written: 'neural circuit synthesis')
  scale          every changed tensor has >= 2 changed entries and satisfies new = g * old on them (one factor g per tensor; uniform or partial gain edits)
  lowrank        every changed tensor is a matrix whose update has numerical rank <= 2 AND < min(shape) (so it is informative) and covers
                 >= 25% of the tensor (dense direction / projection edits; sparse edits and 2-column output layers never qualify)
  layer_rewrite  touched tensors are dense (>= 90% entries changed), total change <= 50% (e.g. an output-layer refit)
  local          <= 10% of parameters changed and not dense/scale/lowrank (sparse element edits)
  mixed          everything else (diffuse or multi-mechanism)
Priority in this order. 'repair-type' mechanisms for claims: scale, lowrank, local, layer_rewrite; 'synthesis': rewrite.
"""
from __future__ import annotations

import numpy as np

REPAIR_TYPES = ("scale", "lowrank", "layer_rewrite", "local")
ORDER = ["noop", "scale", "lowrank", "local", "layer_rewrite", "mixed", "rewrite"]
T_REWRITE, T_DENSE, T_LOCAL, RANK_TOL, T_RANK_COVER = 0.5, 0.9, 0.10, 1e-3, 0.25


def _numrank(d: np.ndarray) -> int:
    m = d if d.ndim == 2 else d.reshape(1, -1)
    s = np.linalg.svd(m, compute_uv=False)
    return int((s > RANK_TOL * (s.max() if s.size and s.max() > 0 else 1)).sum())


def classify(base: dict, final: dict) -> dict:
    tot = sum(v.size for v in base.values()); n_ch = 0; touched = {}
    for k in base:
        d = final[k] - base[k]; ch = d != 0
        if ch.any():
            touched[k] = (d, ch, base[k], final[k]); n_ch += int(ch.sum())
    frac = n_ch / max(tot, 1)
    feats = {"frac_changed": frac, "tensors": sorted(touched), "n_changed": n_ch}
    if not touched:
        return {"mechanism": "noop", **feats}
    scale_ok, rank_ok, dense_ok = True, True, True; gains = {}
    for k, (d, ch, old, new) in touched.items():
        o, n = old[ch], new[ch]
        denom = float(o @ o)
        g = float(o @ n) / denom if denom > 0 else np.nan
        res = np.linalg.norm(n - g * o) / max(np.linalg.norm(n - o), 1e-30) if denom > 0 else np.inf
        scale_ok &= bool(denom > 0 and res < 1e-6 and ch.sum() >= 2); gains[k] = g       # one changed entry is trivially a 'scale': require >= 2
        r = _numrank(d)
        rank_ok &= bool(d.ndim == 2 and r <= 2 and r < min(d.shape) and ch.mean() >= T_RANK_COVER)
        dense_ok &= bool(ch.mean() >= T_DENSE)
    feats["gains"] = gains
    if frac > T_REWRITE: mech = "rewrite"
    elif scale_ok: mech = "scale"
    elif rank_ok: mech = "lowrank"
    elif dense_ok: mech = "layer_rewrite"
    elif frac <= T_LOCAL: mech = "local"
    else: mech = "mixed"
    return {"mechanism": mech, **feats}
