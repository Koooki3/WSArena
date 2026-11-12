"""OCBench (CPU, lite) backend for WSArena (run inside venvs/ocbench). Same task interface as the Tier-0 TaskSpec / MetaWorldTask.

Determinism (verified, tests/test_official_ocbench.py): `np.random.seed(s)` + `env.reset(seed=s)` fixes the initial state; the scripted oracle
(`ocbench.oracles.BlockController`, used ONLY to build demonstrations) draws from the global numpy RNG, so the seed is set before every episode.
Episode seeds come from envs.seed_for(task_id, split, i) % 2**31 (same disjoint namespaces as every other tier).

Protocol conventions (frozen with the contract): success = the environment's `success` flag at ANY step (episode ends there) or after 200 steps;
OOD = actuator-gain shift (commanded action multiplied by `ood_gain`); policies are plain MLPs on RAW observations (input normalisation folded into
W1/b1); failure types are rule-based from the environment's own info fields.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from .. import mlp
from ..envs import RolloutResult, seed_for

MAX_STEPS = 200
NPROC = int(os.environ.get("WSA_NPROC", "4"))
TRAJ_EVERY = 20
GOAL_XY = (0.425, 0.0)

SPECS = {
    "ocb-block": dict(env_name="block-cpu-lite-single-task1-v0", level="L3", ood_gain=0.7, controller="block"),
}

_ENVS: dict = {}


def _env(name):
    if name not in _ENVS:
        import ocbench
        _ENVS[name] = ocbench.make(name)
    return _ENVS[name]


def _controller(name, env):
    from ocbench.oracles import BlockController
    return BlockController(env=env)


def _episode(name, seed, mode, payload, gain, keep_traj, noise, record):
    env = _env(name); u = env.unwrapped
    np.random.seed(int(seed) % (2**32)); ob, info = env.reset(seed=int(seed))
    rng = np.random.default_rng((int(seed) * 7919 + 13) % (2**32))
    agent = None
    if mode == "expert":
        agent = _controller(name, env); agent.reset(ob, info, transition_idx=0)
    goal_z = float(getattr(u, "_cube_size", 0.02)); goal = np.array([GOAL_XY[0], GOAL_XY[1], goal_z])
    ret = 0.0; ok = False; near = False; touched = False; d0 = None; dmin = 1e9; dlast = 0.0; sat = 0.0; jerk = 0.0; prev = np.zeros(7)
    steps = []; obs_l = []; act_l = []; env_fail = False; t_end = 0
    for t in range(MAX_STEPS):
        if mode == "mlp":
            raw = mlp.forward(payload, ob)[0]; lab = np.clip(raw, -1, 1); exe = lab
        else:
            raw = np.asarray(agent.select_action(ob, info, transition_idx=t), float)
            lab = np.clip(payload.get("gain", 1.0) * raw, -1, 1)
            exe = np.clip(lab + rng.normal(0, noise, lab.shape), -1, 1) if noise else lab
        if record:
            obs_l.append(ob.copy()); act_l.append(lab.copy())
        sat += float(np.mean(np.abs(raw) > 1.0)); jerk += float(np.mean(np.abs(lab - prev))); prev = lab
        ob, r, te, tr, info = env.step(np.clip(exe * gain, -1, 1))
        ret += float(r); t_end = t + 1
        ee = np.asarray(info.get("proprio/effector_pos", np.zeros(3))); blk = np.asarray(info.get("privileged/block_0_pos", np.zeros(3)))
        dist = float(np.linalg.norm(blk - goal)); d0 = dist if d0 is None else d0; dmin = min(dmin, dist); dlast = dist
        near |= float(np.linalg.norm(ee - blk)) < 0.06; touched |= float(np.max(np.asarray(info.get("proprio/gripper_contact", 0.0)))) > 0.5
        if keep_traj and t % TRAJ_EVERY == 0:
            steps.append({"t": t, "effector": np.round(ee, 3).tolist(), "block": np.round(blk, 3).tolist(), "block_to_goal": round(dist, 3),
                          "gripper_contact": round(float(np.max(np.asarray(info.get('proprio/gripper_contact', 0.0)))), 2)})
        if bool(info.get("success", False)):
            ok = True
            break
        if bool(info.get("failure", False)) or not bool(info.get("healthy", True)):
            env_fail = True
            break
        if te or tr:
            break
    if ok: ftype = "success"
    elif env_fail: ftype = "env_failure"
    elif not near: ftype = "never_near_block"
    elif not touched: ftype = "no_contact"
    elif d0 is not None and (d0 - dmin) / (d0 + 1e-9) >= 0.3: ftype = "partial_progress"
    else: ftype = "no_progress"
    return {"success": ok, "ret": ret, "failure": ftype, "sat": sat / max(t_end, 1), "jerk": jerk / max(t_end, 1), "final_dist": dlast, "len": t_end,
            "steps": steps, "obs": obs_l, "act": act_l}


def _chunk(args):
    name, seeds, mode, payload, gain, keep_traj, noise, record = args
    return [_episode(name, s, mode, payload, gain, keep_traj, noise, record) for s in seeds]


_POOL = None


def _pool():
    global _POOL
    if _POOL is None and NPROC > 1:
        _POOL = ProcessPoolExecutor(max_workers=NPROC, mp_context=mp.get_context("fork"))
    return _POOL


def _run(name, seeds, mode, payload, gain=1.0, keep_traj=False, noise=0.0, record=False):
    if NPROC <= 1 or len(seeds) < 2 * NPROC:
        return _chunk((name, seeds, mode, payload, gain, keep_traj, noise, record))
    parts = np.array_split(np.array(seeds, dtype=np.int64), NPROC)
    futs = [_pool().submit(_chunk, (name, [int(x) for x in p], mode, payload, gain, keep_traj, noise, record)) for p in parts if len(p)]
    return [e for f in futs for e in f.result()]


_CUBE = {}


def _cube_size(name):
    if name not in _CUBE:
        _CUBE[name] = float(getattr(_env(name).unwrapped, "_cube_size", 0.02))
    return _CUBE[name]


@dataclass(frozen=True)
class OCBenchTask:
    task_id: str
    level: str
    env_name: str
    controller: str
    ood_gain: float = 0.7
    n_hidden: int = 24
    obs_dim: int = 28

    @property
    def act_dim(self) -> int:
        return 7

    def arch(self):
        return (self.obs_dim, self.n_hidden, 7)

    def _seeds(self, split, start, n):
        return [seed_for(self.task_id, split, start + i) % (2**31) for i in range(n)]

    def rollout(self, weights, split: str, start: int, n: int, ood: bool = False, keep_traj: bool = False):
        eps = _run(self.env_name, self._seeds(split, start, n), "mlp", weights, gain=self.ood_gain if ood else 1.0, keep_traj=keep_traj)
        res = RolloutResult(success=np.array([e["success"] for e in eps]), ret=np.array([e["ret"] for e in eps]), failure=[e["failure"] for e in eps],
                            diag={"mean_final_block_to_goal": float(np.mean([e["final_dist"] for e in eps])), "mean_episode_length": float(np.mean([e["len"] for e in eps])),
                                  "action_saturation_rate": float(np.mean([e["sat"] for e in eps])), "mean_action_jerk": float(np.mean([e["jerk"] for e in eps]))})
        if keep_traj:
            res.traj = {"episodes": [e["steps"] for e in eps]}
        return res

    def traj_summary(self, res, idx: int, every: int = 1) -> dict:
        return {"episode_outcome": res.failure[idx], "steps": res.traj["episodes"][idx][:12]}

    def describe(self) -> str:
        cs = _cube_size(self.env_name)
        return (
            "OCBench 'block, single cube, task 1' (lite variant): a simulated UR5e robot arm with a Robotiq two-finger gripper above a table with ONE cube. "
            f"The goal of this task never changes: move the cube to the table centre, goal position (x, y, z) = (0.425, 0.0, {cs:.3f}) m (z is the resting height of the cube). "
            "The cube starts at a random position on the table (never already at the goal). Success: the cube centre is within 0.04 m of the goal; "
            f"the episode ends at the first success or after {MAX_STEPS} control steps (0.1 s each).\n"
            "Observation (28 numbers): [0:6] arm joint positions (rad); [6:12] arm joint velocities; [12:15] (end-effector position - (0.425, 0, 0)) x 10; "
            "[15] cos(end-effector yaw); [16] sin(end-effector yaw); [17] gripper opening x 3; [18] gripper contact (1 if the fingers touch something); "
            "[19:22] (cube position - (0.425, 0, 0)) x 10; [22:26] cube orientation quaternion (w, x, y, z); [26] cos(cube yaw); [27] sin(cube yaw). "
            f"In observation units the goal cube position is approximately (0, 0, {cs * 10:.2f}).\n"
            "Output (7 numbers, clipped to [-1, 1]): normalised joint-position-delta commands for the 6 arm joints, then the gripper command. "
            "The exact physical parameters of hidden episodes may differ from the nominal ones.")

    # ---- checkpoint-construction hooks (privileged; never shown to editors) ----
    def expert_dataset(self, n_episodes: int, seed: int, noise: float = 0.15, expert_kw=None, goal_filter=None):
        rng = np.random.default_rng(seed)
        seeds = [int(x) for x in rng.integers(0, 2**31, n_episodes)]
        eps = _run(self.env_name, seeds, "expert", expert_kw or {}, noise=noise, record=True)
        return (np.concatenate([np.array(e["obs"]) for e in eps]), np.concatenate([np.array(e["act"]) for e in eps]))

    def good_expert_kw(self) -> dict:
        return {}

    def train_policy(self, X, Y, seed: int, epochs: int, lr: float = 3e-3, snapshots=None):
        from ..tasks import train_bc
        mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-3, 1.0, sd)
        out = train_bc(self.arch(), (X - mu) / sd, Y, seed, epochs=epochs, lr=lr, snapshots=snapshots)

        def fold(w):
            w = {k: v.copy() for k, v in w.items()}
            w["b1"] = w["b1"] - (mu / sd) @ w["W1"]; w["W1"] = w["W1"] / sd[:, None]
            return w
        return [(e, fold(w)) for e, w in out] if snapshots is not None else fold(out)

    @property
    def build_cfg(self) -> dict:
        return {"n_ep": 60, "epochs_good": 400, "epochs_s2": 100, "snap_lr": 3e-3, "snap_epochs": list(range(10, 101, 5)), "noise": 0.15}

    def s2_candidates(self) -> list:
        return [("underpowered_demos", dict(gain=g), None) for g in (0.3, 0.4, 0.5, 0.6, 0.7)]


def get_task(task_id: str) -> OCBenchTask:
    sp = SPECS[task_id]
    return OCBenchTask(task_id=task_id, level=sp["level"], env_name=sp["env_name"], controller=sp["controller"], ood_gain=sp["ood_gain"])
