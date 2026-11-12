"""Developer / hidden evaluators, budget accounting and the structured feedback shown to editors.

Seed namespaces (envs.seed_for): 'select' (checkpoint selection, never shown), 'dev' (feedback to editors),
'hid_id' / 'hid_ood' (final hidden tests; editors never see them), 'probe' (read-only forward probes).
The editor can only request evaluation through the harness; it can never start an environment itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import envs, mlp
from .tasks import TaskSpec


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    rounds: int = 8
    episodes_per_round: int = 16
    initial_report_episodes: int = 16
    probe_inputs_per_round: int = 0     # 0 disables probes (Pilot default, like MIT exp 4/5)
    used_rounds: int = 0
    used_dev_episodes: int = 0
    used_probe_inputs: int = 0

    @property
    def dev_total(self) -> int:
        return self.initial_report_episodes + self.rounds * self.episodes_per_round

    def charge_episodes(self, n: int):
        if self.used_dev_episodes + n > self.dev_total:
            raise BudgetExceeded("dev episode budget")
        self.used_dev_episodes += n


def _traj_summary(res: envs.RolloutResult, idx: int, every: int = 6) -> dict:
    P, V, A = res.traj["P"], res.traj["V"], res.traj["A"]
    T = A.shape[0]
    ts = list(range(0, T, every)) + [T - 1]
    return {
        "episode_outcome": res.failure[idx],
        "goal": np.round(res.traj["goal"][idx], 3).tolist(),
        "obstacle": np.round(res.traj["obstacle"][idx], 3).tolist(),
        "steps": [{"t": int(t), "pos": np.round(P[t, idx], 3).tolist(), "vel": np.round(V[t, idx], 3).tolist(),
                   "action": np.round(A[t, idx], 3).tolist()} for t in ts],
    }


def dev_report(task: TaskSpec, weights, start: int, n: int, round_id: int, remaining_rounds: int,
               with_trajectory: bool = False) -> dict:
    res = task.rollout(weights, "dev", start, n, keep_traj=with_trajectory)
    fail = Counter(f for f in res.failure if f != "success")
    rep = {
        "round": round_id,
        "episodes_used": n,
        "success_rate": round(float(res.success.mean()), 4),
        "mean_return": round(float(res.ret.mean()), 4),
        "failures": dict(sorted(fail.items())),
        "diagnostic": {k: round(v, 4) for k, v in res.diag.items()},
        "remaining_rounds": remaining_rounds,
    }
    if with_trajectory:
        bad = [i for i, f in enumerate(res.failure) if f != "success"]
        if bad:
            rep["example_failure_trajectory"] = task.traj_summary(res, bad[0])
    return rep


def _eval_split(task, weights, split, n, ood):
    r = task.rollout(weights, split, 0, n, ood=ood)
    return r


def hidden_eval(task: TaskSpec, weights, base_weights, n: int = 200, cache=None) -> dict:
    """Final hidden evaluation: paired episodes for edited vs base on ID and OOD splits.
    A no-op edit (identical weights) reuses the base result; `cache` (dict-like, optional) memoises base evaluations
    by (task, base hash, split, n) -- evaluation is deterministic, so this changes nothing but wall time."""
    from .edits import weights_sha256
    h_e, h_b = weights_sha256(weights), weights_sha256(base_weights)
    out = {}
    for name, split, ood in (("id", "hid_id", False), ("ood", "hid_ood", True)):
        key = (task.task_id, h_b, split, n)
        b = cache.get(key) if cache is not None else None
        if b is None:
            b = _eval_split(task, base_weights, split, n, ood)
            if cache is not None:
                cache[key] = b
        e = b if h_e == h_b else _eval_split(task, weights, split, n, ood)
        s_e, s_b = e.success, b.success
        out[name] = {
            "n": n,
            "sr_base": float(s_b.mean()), "sr_edited": float(s_e.mean()),
            "delta_sr": float(s_e.mean() - s_b.mean()),
            "recovery": float(s_e[~s_b].mean()) if (~s_b).any() else None,
            "regression": float((~s_e[s_b]).mean()) if s_b.any() else None,
            "n_base_fail": int((~s_b).sum()), "n_base_succ": int(s_b.sum()),
            "ret_base": float(b.ret.mean()), "ret_edited": float(e.ret.mean()),
            "episode_success_edited": s_e.astype(int).tolist(),
            "episode_success_base": s_b.astype(int).tolist(),
            "failures_edited": dict(Counter(e.failure)),
        }
    return out


def probe(weights, inputs: list[list[float]], max_inputs: int) -> dict:
    """Read-only forward probe: pure function of weights; no state, no gradients."""
    x = np.asarray(inputs[:max_inputs], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != weights["W1"].shape[0]:
        raise ValueError("probe input has wrong dimension")
    out = mlp.forward(weights, x)
    return {"n": len(x), "outputs": np.round(out, 4).tolist()}
