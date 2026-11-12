"""Probe track (P6): read-only probes answered by the harness, no CLI tools.
Commands (all act on the CURRENT weights; none can run the environment, train, or search):
  slice T[a:b, c:d]        values of a slice (<= 400 numbers)
  stats T axis=0|1         per-row/column mean, std, max|.|, L2 (<= 200 rows)
  topk T k=N               N largest |entries| with indices (N <= 50)
  forward obs=[[..],[..]] [layers=1,2]   network outputs on <= 8 observations (+ first 16 hidden units of the chosen layers)
Everything else is invalid (counted and reported)."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
import numpy as np

SLICE_MAX, STATS_MAX_ROWS, TOPK_MAX, OBS_MAX, HID_SHOW = 400, 200, 50, 8, 16
_FENCE = re.compile(r"```probe[ \t]*\n(.*?)```", re.S)


@dataclass
class ProbeBudget:
    max_subturns: int = 3          # probe sub-turns per round
    max_cmds_per_turn: int = 4
    max_numbers: int = 8000        # total numbers returned over the whole run
    used_numbers: int = 0
    subturns_used: int = 0

    def exhausted(self) -> bool:
        return self.used_numbers >= self.max_numbers


def last_probe_block(text: str) -> str | None:
    m = _FENCE.findall(text or "")
    return m[-1] if m else None


def _fmt(x) -> str:
    return f"{float(x):.5g}"


def _parse_index(spec: str, shape):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != len(shape): raise ValueError("index rank does not match the tensor")
    idx = []
    for p, n in zip(parts, shape):
        if ":" in p:
            a, b = (p.split(":") + [""])[:2]
            lo = int(a) if a.strip() else 0; hi = int(b) if b.strip() else n
            lo, hi = max(0, lo), min(n, hi)
            if lo >= hi: raise ValueError("empty range")
            idx.append(slice(lo, hi))
        else:
            i = int(p)
            if not 0 <= i < n: raise ValueError("index out of range")
            idx.append(slice(i, i + 1))
    return tuple(idx)


def _run_one(w: dict, line: str, budget: ProbeBudget) -> tuple[str, int, bool]:
    """-> (answer text, numbers charged, valid)"""
    line = line.split("#")[0].strip()
    if not line: return "", 0, True
    m = re.match(r"slice\s+(\w+)\[(.*)\]\s*$", line)
    if m:
        t, spec = m.groups()
        if t not in w: return f"slice {t}: unknown tensor", 0, False
        try: idx = _parse_index(spec, w[t].shape)
        except Exception as e: return f"slice {t}[{spec}]: invalid ({e})", 0, False
        a = w[t][idx]
        if a.size > SLICE_MAX: return f"slice {t}[{spec}]: {a.size} numbers exceeds the per-command limit {SLICE_MAX}", 0, False
        if budget.used_numbers + a.size > budget.max_numbers: return "slice: total probe budget exhausted", 0, False
        return f"{t}[{spec}] = " + json.dumps(np.vectorize(_fmt)(a).astype(float).tolist()), int(a.size), True
    m = re.match(r"stats\s+(\w+)\s+axis=([01])\s*$", line)
    if m:
        t, ax = m.group(1), int(m.group(2))
        if t not in w or w[t].ndim != 2: return f"stats {t}: needs a 2-D tensor", 0, False
        a = w[t]; r = np.abs(a).max(axis=ax); n = a.shape[1 - ax]
        if n > STATS_MAX_ROWS: return f"stats {t}: {n} rows exceeds {STATS_MAX_ROWS}", 0, False
        cost = 4 * n
        if budget.used_numbers + cost > budget.max_numbers: return "stats: total probe budget exhausted", 0, False
        rows = [{"i": i, "mean": float(_fmt(a.mean(axis=ax)[i])), "std": float(_fmt(a.std(axis=ax)[i])), "max_abs": float(_fmt(r[i])), "l2": float(_fmt(np.linalg.norm(a, axis=ax)[i]))} for i in range(n)]
        return f"stats {t} axis={ax} (per {'column' if ax == 0 else 'row'}): " + json.dumps(rows), cost, True
    m = re.match(r"topk\s+(\w+)\s+k=(\d+)\s*$", line)
    if m:
        t, k = m.group(1), int(m.group(2))
        if t not in w: return f"topk {t}: unknown tensor", 0, False
        if not 1 <= k <= TOPK_MAX: return f"topk: k must be 1..{TOPK_MAX}", 0, False
        a = w[t]; flat = np.argsort(-np.abs(a).ravel(), kind="stable")[:k]; cost = 2 * k
        if budget.used_numbers + cost > budget.max_numbers: return "topk: total probe budget exhausted", 0, False
        items = [{"index": [int(x) for x in np.unravel_index(i, a.shape)], "value": float(_fmt(a.ravel()[i]))} for i in flat]
        return f"topk {t} k={k}: " + json.dumps(items), cost, True
    m = re.match(r"forward\s+obs=(\[.*\])(?:\s+layers=([\d,]+))?\s*$", line)
    if m:
        try: obs = np.array(json.loads(m.group(1)), float)
        except Exception: return "forward: obs is not a JSON list of numbers", 0, False
        obs = np.atleast_2d(obs)
        if obs.ndim != 2 or obs.shape[0] > OBS_MAX or obs.shape[1] != w["W1"].shape[0]: return f"forward: need 1..{OBS_MAX} observations of length {w['W1'].shape[0]}", 0, False
        layers = sorted({int(x) for x in (m.group(2) or "").split(",") if x}) if m.group(2) else []
        if any(l not in (1, 2) for l in layers): return "forward: layers must be 1 and/or 2", 0, False
        from wsarena import mlp
        out, h1, h2 = mlp.forward(w, obs, return_hidden=True); hs = {1: h1, 2: h2}
        cost = out.size + sum(min(HID_SHOW, hs[l].shape[1]) * obs.shape[0] for l in layers)
        if budget.used_numbers + cost > budget.max_numbers: return "forward: total probe budget exhausted", 0, False
        ans = {"out": np.vectorize(_fmt)(out).astype(float).tolist()}
        for l in layers: ans[f"h{l}[:{HID_SHOW}]"] = np.vectorize(_fmt)(hs[l][:, :HID_SHOW]).astype(float).tolist()
        return "forward -> " + json.dumps(ans), int(cost), True
    return f"unknown or disallowed command: {line[:60]!r} (allowed: slice, stats, topk, forward; nothing can run the environment, train or search)", 0, False


@dataclass
class ProbeResult:
    text: str
    n_valid: int = 0
    n_invalid: int = 0
    numbers: int = 0
    refused: bool = False
    commands: list = field(default_factory=list)


def answer_probes(w: dict, block: str, budget: ProbeBudget) -> ProbeResult:
    """Execute one probe sub-turn (a probe block) against the current weights; mutates `budget`."""
    res = ProbeResult(text="")
    if budget.subturns_used >= budget.max_subturns:
        res.refused = True; res.text = "PROBE BUDGET: no probe sub-turns left this round; give your edits block now."; return res
    budget.subturns_used += 1
    lines = [l for l in block.splitlines() if l.split("#")[0].strip()]
    out = []
    for i, line in enumerate(lines):
        if i >= budget.max_cmds_per_turn:
            out.append(f"(command {i+1} ignored: at most {budget.max_cmds_per_turn} commands per probe turn)"); res.n_invalid += 1; continue
        ans, cost, ok = _run_one(w, line, budget); budget.used_numbers += cost; res.numbers += cost
        res.n_valid += ok; res.n_invalid += (not ok); res.commands.append({"cmd": line.strip()[:120], "valid": ok, "numbers": cost}); out.append(ans)
    left = budget.max_subturns - budget.subturns_used
    res.text = "\n".join(out) + f"\n[probe budget: {left} probe turn(s) left this round; {budget.max_numbers - budget.used_numbers} numbers left in total]"
    return res
