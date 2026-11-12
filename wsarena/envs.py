"""Tier-0 control environments (self-built, vectorised, fully deterministic given seeds).

2-D point mass, force control. State p, v in R^2, arena [-1.2, 1.2]^2 (soft wall: position clamped,
velocity zeroed on contact). dt = 0.1.

    accel = fmax * clip(a, -1, 1) / mass - damping * v ;  v += dt * accel ;  p += dt * p_dot

Tasks (complexity ladder, H5 of the design):
  reach  (L1/L2)  observation  [ex, ey, vx, vy]                       e = goal - p
  avoid  (L3)     observation  [ex, ey, vx, vy, ox, oy]               o = obstacle - p (circle, radius r_o)
  hold   (L4)     observation  [ex, ey, vx, vy, t/T]                  must STAY in the goal zone for the last H steps
  delay  (L5)     observation  [e_{t-d}, v_{t-d}, a_{t-1}, a_{t-2}]   d-step observation delay (needs history/estimation)

Hidden (private) parameters give an in-distribution split and an out-of-distribution split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict

import numpy as np

from . import mlp

DT = 0.1
ARENA = 1.2


@dataclass(frozen=True)
class EnvParams:
    kind: str                       # reach | avoid | hold | delay
    T: int = 80
    mass: float = 1.0
    damping: float = 0.2
    fmax: float = 2.0
    goal_radius: float = 0.08
    speed_tol: float = 0.2
    obstacle_radius: float = 0.18
    hold_steps: int = 25
    delay: int = 3

    def obs_dim(self) -> int:
        return {"reach": 4, "avoid": 6, "hold": 5, "delay": 8}[self.kind]

    def to_dict(self) -> dict:
        return asdict(self)


def seed_for(task_id: str, split: str, index: int) -> int:
    """Disjoint, reproducible seed namespaces per (task, split, episode index)."""
    task_id = task_id.split("@")[0]      # width variants (task@w64) evaluate the SAME episodes as the base task
    h = hashlib.sha256(f"wsarena|{task_id}|{split}|{index}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


def sample_init(params: EnvParams, seeds: list[int]) -> dict[str, np.ndarray]:
    n = len(seeds)
    p0 = np.zeros((n, 2)); goal = np.zeros((n, 2)); obs_c = np.zeros((n, 2))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(s)
        for _ in range(200):
            p = r.uniform(-0.9, 0.9, 2)
            g = r.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(g - p) >= 0.7:
                break
        p0[i], goal[i] = p, g
        if params.kind == "avoid":
            d = g - p
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            obs_c[i] = (p + g) / 2 + nrm * r.uniform(-0.08, 0.08)
    return {"p0": p0, "goal": goal, "obstacle": obs_c}


@dataclass
class RolloutResult:
    success: np.ndarray            # (N,) bool
    ret: np.ndarray                # (N,) float
    failure: list[str]             # per episode: "success" or a rule-based failure type
    diag: dict[str, float]
    traj: dict[str, np.ndarray] = field(default_factory=dict)  # only if keep_traj


def _observe(params, p, v, goal, obstacle, t, hist):
    e = goal - p
    if params.kind == "reach":
        return np.concatenate([e, v], axis=1)
    if params.kind == "avoid":
        return np.concatenate([e, v, obstacle - p], axis=1)
    if params.kind == "hold":
        return np.concatenate([e, v, np.full((len(p), 1), t / params.T)], axis=1)
    # delay: observation of d steps ago + two previous actions
    d = params.delay
    e_d, v_d = hist["e"][max(t - d, 0)], hist["v"][max(t - d, 0)]
    a1 = hist["a"][t - 1] if t >= 1 else np.zeros_like(v)
    a2 = hist["a"][t - 2] if t >= 2 else np.zeros_like(v)
    return np.concatenate([e_d, v_d, a1, a2], axis=1)


def rollout(params: EnvParams, policy, seeds: list[int], keep_traj: bool = False) -> RolloutResult:
    """policy: callable obs (N,D) -> raw output (N,2). Deterministic."""
    init = sample_init(params, seeds)
    n = len(seeds)
    p, v = init["p0"].copy(), np.zeros((n, 2))
    goal, obstacle = init["goal"], init["obstacle"]
    T = params.T
    hist = {"e": [goal - p], "v": [v.copy()], "a": []}
    P = np.zeros((T + 1, n, 2)); V = np.zeros((T + 1, n, 2)); A = np.zeros((T, n, 2))
    P[0] = p
    ret = np.zeros(n)
    collided = np.zeros(n, bool)
    wall_steps = np.zeros(n)
    sat = 0.0; jerk = 0.0
    prev_a = np.zeros((n, 2))
    in_zone = np.zeros((T, n), bool)
    for t in range(T):
        obs = _observe(params, p, v, goal, obstacle, t, hist)
        raw = policy(obs)
        sat += float(np.mean(np.abs(raw) > 1.0))
        a = np.clip(raw, -1.0, 1.0)
        jerk += float(np.mean(np.abs(a - prev_a))); prev_a = a
        accel = params.fmax * a / params.mass - params.damping * v
        v = v + DT * accel
        p = p + DT * v
        hit = np.abs(p) > ARENA
        wall_steps += hit.any(axis=1)
        p = np.clip(p, -ARENA, ARENA)
        v = np.where(hit, 0.0, v)
        if params.kind == "avoid":
            collided |= np.linalg.norm(p - obstacle, axis=1) < params.obstacle_radius
        dist = np.linalg.norm(goal - p, axis=1)
        in_zone[t] = (dist < params.goal_radius) & (np.linalg.norm(v, axis=1) < params.speed_tol)
        ret += -dist * DT
        hist["e"].append(goal - p); hist["v"].append(v.copy()); hist["a"].append(a)
        P[t + 1] = p; V[t + 1] = v; A[t] = a
    dist_f = np.linalg.norm(goal - p, axis=1)
    speed_f = np.linalg.norm(v, axis=1)
    settled = (dist_f < params.goal_radius) & (speed_f < params.speed_tol)
    if params.kind == "hold":
        success = in_zone[-params.hold_steps:].all(axis=0)
    else:
        success = settled
    if params.kind == "avoid":
        success = success & ~collided
    min_dist = np.min(np.linalg.norm(goal[None] - P, axis=2), axis=0)
    failure = []
    for i in range(n):
        if success[i]:
            failure.append("success"); continue
        if params.kind == "avoid" and collided[i]:
            failure.append("collision")
        elif wall_steps[i] >= 5:
            failure.append("wall")
        elif params.kind == "hold" and in_zone[:, i].any():
            failure.append("left_zone")
        elif min_dist[i] < params.goal_radius and not settled[i]:
            failure.append("overshoot_unsettled")
        elif dist_f[i] < 0.3:
            failure.append("near_unsettled")
        elif speed_f[i] < 0.1:
            failure.append("stalled_far")
        else:
            failure.append("drift_far")
    diag = {
        "mean_final_dist": float(dist_f.mean()),
        "mean_final_speed": float(speed_f.mean()),
        "action_saturation_rate": sat / T,
        "mean_action_jerk": jerk / T,
    }
    res = RolloutResult(success=success, ret=ret, failure=failure, diag=diag)
    if keep_traj:
        res.traj = {"P": P, "V": V, "A": A, "goal": goal, "obstacle": obstacle}
    return res


def make_policy(weights: dict[str, np.ndarray]):
    return lambda obs: mlp.forward(weights, obs)
