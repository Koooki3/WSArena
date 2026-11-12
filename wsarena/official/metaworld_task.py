"""Meta-World v3 backend for WSArena (run inside venvs/metaworld).

Same task interface as the Tier-0 TaskSpec (rollout / describe / traj_summary / expert_dataset / train_policy / hooks), so the
protocol, evaluators, controls and analysis are unchanged.

Determinism (verified, tests/test_official_metaworld.py): Meta-World ignores reset(seed=...). An episode is therefore set up by
  u.seed(s); u._freeze_rand_vec=False; env.reset()      -> samples the object/goal configuration from the seeded RNG
  freeze that configuration; env.reset()                -> deterministic initial state and trajectory.
Episode seeds come from envs.seed_for(task_id, split, i) % 2**31 (same disjoint namespaces as Tier-0).

Protocol conventions (frozen with the contract):
  * success = the environment's `success` flag at ANY step; the episode ends at the first success or after 500 steps;
  * OOD = actuator-gain shift (commanded action multiplied by `ood_gain` before it reaches the simulator);
  * policies are plain MLPs on RAW observations: input normalisation is folded into W1/b1 (no hidden preprocessing);
  * failure types are rule-based from the environment's own info fields (never self-reported by the editor).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from .. import mlp
from ..envs import RolloutResult, seed_for

MAX_STEPS = 500
NPROC = int(os.environ.get("WSA_NPROC", "4"))
TRAJ_EVERY = 25

SPECS = {
    "mw-push": dict(env_name="push-v3", level="L3", kind="push", succ_dist=0.05, ood_gain=0.5,
                    blurb="Push the puck on the table to the goal position."),
    "mw-peginsert": dict(env_name="peg-insert-side-v3", level="L5", kind="pickplace", succ_dist=0.07, ood_gain=0.7,
                         blurb="Grasp the peg and insert it into the hole of the box from the side.",
                         succ_text="the peg reaches the goal position given in the observation, within a few centimetres, i.e. it is inserted into the hole",
                         build=(("n_ep", 200), ("epochs_good", 800), ("epochs_s2", 400), ("snap_epochs", tuple(range(100, 801, 50))))),
    "mw-pickplace": dict(env_name="pick-place-v3", level="L4", kind="pickplace", succ_dist=0.07, ood_gain=0.6,
                         blurb="Pick up the puck and place it at the goal position (the goal may be above the table)."),
}

_ENVS: dict = {}      # per-process cache
_POLS: dict = {}


def _env(env_name):
    if env_name not in _ENVS:
        import gymnasium as gym
        import metaworld  # noqa: F401  (registers Meta-World/MT1)
        _ENVS[env_name] = gym.make("Meta-World/MT1", env_name=env_name, seed=0)
    return _ENVS[env_name]


def _expert(env_name):
    if env_name not in _POLS:
        from metaworld.policies import ENV_POLICY_MAP
        _POLS[env_name] = ENV_POLICY_MAP[env_name]()
    return _POLS[env_name]


def _setup(env, s: int):
    u = env.unwrapped
    u._freeze_rand_vec = False
    u.seed(int(s))
    env.reset()
    rv = u._last_rand_vec.copy()
    u._freeze_rand_vec = True
    u._last_rand_vec = rv
    return env.reset()[0]


def _episode(env_name, kind, s, mode, payload, gain, keep_traj, noise, record, gfilter):
    env = _env(env_name)
    ob = _setup(env, s)
    if gfilter is not None and not _pass_filter(ob, gfilter):
        return None
    rng = np.random.default_rng((int(s) * 7919 + 13) % (2**32))
    exp = _expert(env_name) if mode == "expert" else None
    ret = 0.0; ok = False; near = False; grasp = False; o2t0 = None; o2t_min = 1e9; o2t_last = 0.0
    sat = 0.0; jerk = 0.0; prev = np.zeros(4); steps = []; obs_l = []; act_l = []; t_end = 0
    for t in range(MAX_STEPS):
        if mode == "mlp":
            raw = mlp.forward(payload, ob)[0]
            lab = np.clip(raw, -1, 1); exe = lab
        else:
            raw = exp.get_action(ob)
            lab = np.clip(payload.get("gain", 1.0) * raw, -1, 1)
            exe = np.clip(lab + rng.normal(0, noise, lab.shape), -1, 1) if noise else lab
        if record:
            obs_l.append(ob.copy()); act_l.append(lab.copy())
        sat += float(np.mean(np.abs(raw) > 1.0)); jerk += float(np.mean(np.abs(lab - prev))); prev = lab
        ob, r, te, tr, info = env.step(np.clip(exe * gain, -1, 1))
        ret += float(r); t_end = t + 1
        near |= float(info.get("near_object", 0)) > 0.5
        grasp |= float(info.get("grasp_success", 0)) > 0.5
        o2t = float(info.get("obj_to_target", 0.0)); o2t0 = o2t if o2t0 is None else o2t0
        o2t_min = min(o2t_min, o2t); o2t_last = o2t
        if keep_traj and t % TRAJ_EVERY == 0:
            steps.append({"t": t, "hand": np.round(ob[:3], 3).tolist(), "gripper_open": round(float(ob[3]), 3),
                          "object": np.round(ob[4:7], 3).tolist(), "obj_to_goal": round(o2t, 3),
                          "action": np.round(lab, 3).tolist()})
        if float(info.get("success", 0)) > 0.5:
            ok = True
            break
        if te or tr:
            break
    if ok:
        ftype = "success"
    elif not near:
        ftype = "never_near_object"
    elif kind == "pickplace" and not grasp:
        ftype = "no_grasp"
    elif o2t0 is not None and (o2t0 - o2t_min) / (o2t0 + 1e-9) >= 0.3:
        ftype = "partial_progress"
    else:
        ftype = "no_progress"
    return {"success": ok, "ret": ret, "failure": ftype, "sat": sat / max(t_end, 1), "jerk": jerk / max(t_end, 1),
            "final_o2t": o2t_last, "len": t_end, "steps": steps, "obs": obs_l, "act": act_l, "seed": int(s)}


def _pass_filter(ob, spec) -> bool:
    if "goal_x_gt" in spec:
        return float(ob[-3]) > spec["goal_x_gt"]
    return True


def _chunk(args):
    env_name, kind, seeds, mode, payload, gain, keep_traj, noise, record, gfilter = args
    return [_episode(env_name, kind, s, mode, payload, gain, keep_traj, noise, record, gfilter) for s in seeds]


_POOL = None


def _pool():
    global _POOL
    if _POOL is None and NPROC > 1:
        _POOL = ProcessPoolExecutor(max_workers=NPROC, mp_context=mp.get_context("fork"))
    return _POOL


def _run(env_name, kind, seeds, mode, payload, gain=1.0, keep_traj=False, noise=0.0, record=False, gfilter=None):
    if NPROC <= 1 or len(seeds) < 2 * NPROC:
        return _chunk((env_name, kind, seeds, mode, payload, gain, keep_traj, noise, record, gfilter))
    parts = np.array_split(np.array(seeds, dtype=np.int64), NPROC)
    futs = [_pool().submit(_chunk, (env_name, kind, [int(x) for x in p], mode, payload, gain, keep_traj, noise, record, gfilter))
            for p in parts if len(p)]
    return [e for f in futs for e in f.result()]


@dataclass(frozen=True)
class MetaWorldTask:
    task_id: str
    level: str
    env_name: str
    kind: str
    succ_dist: float
    blurb: str
    n_hidden: int = 24
    succ_text: str = ""
    build_override: tuple = ()
    ood_gain: float = 0.7          # per-task value from scripts/calibrate_official.py (runs/SUMMARY/ood_calibration_official.json)

    obs_dim: int = 39

    @property
    def act_dim(self) -> int:
        return 4

    def arch(self):
        return (self.obs_dim, self.n_hidden, 4)

    # ---- rollout interface -------------------------------------------------------------------------
    def _seeds(self, split, start, n):
        return [seed_for(self.task_id, split, start + i) % (2**31) for i in range(n)]

    def rollout(self, weights, split: str, start: int, n: int, ood: bool = False, keep_traj: bool = False):
        eps = _run(self.env_name, self.kind, self._seeds(split, start, n), "mlp", weights,
                   gain=self.ood_gain if ood else 1.0, keep_traj=keep_traj)
        res = RolloutResult(
            success=np.array([e["success"] for e in eps]), ret=np.array([e["ret"] for e in eps]),
            failure=[e["failure"] for e in eps],
            diag={"mean_final_obj_to_goal": float(np.mean([e["final_o2t"] for e in eps])),
                  "mean_episode_length": float(np.mean([e["len"] for e in eps])),
                  "action_saturation_rate": float(np.mean([e["sat"] for e in eps])),
                  "mean_action_jerk": float(np.mean([e["jerk"] for e in eps]))})
        if keep_traj:
            res.traj = {"episodes": [e["steps"] for e in eps]}
        return res

    def traj_summary(self, res, idx: int, every: int = 1) -> dict:
        return {"episode_outcome": res.failure[idx], "steps": res.traj["episodes"][idx][:20]}

    def describe(self) -> str:
        return (
            f"Meta-World v3 '{self.env_name}': a simulated Sawyer robot arm with a gripper above a table. {self.blurb}\n"
            f"The policy runs every control step (at most {MAX_STEPS} steps per episode). An episode ends at the first step where "
            f"the success condition holds ({self.succ_text or f'the object centre is within {self.succ_dist} m of the goal'}) or after {MAX_STEPS} steps.\n"
            "Observation (39 numbers): [0:3] end-effector position (m); [3] gripper opening (about 1 = open, smaller = closed); "
            "[4:7] object position (m); [7:11] object orientation quaternion; [11:18] unused (zeros); [18:36] the same 18 numbers "
            "from the previous step; [36:39] goal position (m). Positions are in the robot's world frame; the table surface is near z = 0.\n"
            "Output (4 numbers, clipped to [-1, 1]): [0:3] desired end-effector displacement per step (x, y, z; 1.0 is the largest "
            "step); [3] gripper command (negative = open, positive = close).\n"
            "Object and goal positions are randomised each episode. The exact physical parameters of hidden episodes may differ "
            "from the nominal ones.")

    # ---- checkpoint-construction hooks (privileged; never shown to editors) -------------------------
    def expert_dataset(self, n_episodes: int, seed: int, noise: float = 0.15, expert_kw=None, goal_filter=None):
        rng = np.random.default_rng(seed)
        seeds = [int(x) for x in rng.integers(0, 2**31, n_episodes * 2 if goal_filter else n_episodes)]
        eps = _run(self.env_name, self.kind, seeds, "expert", expert_kw or {}, noise=noise, record=True, gfilter=goal_filter)
        eps = [e for e in eps if e is not None][:n_episodes]
        if not eps:                      # e.g. a goal filter that no episode passes: the candidate is infeasible, not an error
            return np.zeros((0, self.obs_dim)), np.zeros((0, 4))
        return (np.concatenate([np.array(e["obs"]) for e in eps]), np.concatenate([np.array(e["act"]) for e in eps]))

    def good_expert_kw(self) -> dict:
        return {}

    def train_policy(self, X, Y, seed: int, epochs: int, lr: float = 3e-3, snapshots=None):
        from ..tasks import train_bc
        mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-3, 1.0, sd)       # constant inputs stay unscaled
        out = train_bc(self.arch(), (X - mu) / sd, Y, seed, epochs=epochs, lr=lr, snapshots=snapshots)

        def fold(w):
            w = {k: v.copy() for k, v in w.items()}
            w["b1"] = w["b1"] - (mu / sd) @ w["W1"]
            w["W1"] = w["W1"] / sd[:, None]
            return w
        return [(e, fold(w)) for e, w in out] if snapshots is not None else fold(out)

    @property
    def build_cfg(self) -> dict:
        cfg = {"n_ep": 60, "epochs_good": 400, "epochs_s2": 100, "snap_lr": 3e-3, "snap_epochs": list(range(10, 101, 5)), "noise": 0.15}
        cfg.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in self.build_override})
        return cfg

    def s2_candidates(self) -> list:
        c = [("underpowered_demos", dict(gain=g), None) for g in (0.1, 0.12, 0.15, 0.18, 0.2, 0.25)]   # slow demonstrator -> time-outs
        c.append(("data_gap", {}, {"goal_x_gt": -0.02}))          # demonstrations never saw goals on the far left
        return c


def get_task(task_id: str) -> MetaWorldTask:
    sp = SPECS[task_id]
    return MetaWorldTask(task_id=task_id, level=sp["level"], env_name=sp["env_name"], kind=sp["kind"],
                         succ_dist=sp["succ_dist"], blurb=sp["blurb"], ood_gain=sp["ood_gain"],
                         succ_text=sp.get("succ_text", ""), build_override=sp.get("build", ()))
