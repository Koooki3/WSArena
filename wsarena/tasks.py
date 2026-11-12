"""Tier-0 task registry, expert controllers, BC trainer (used ONLY to build checkpoints, never given to editors),
and the four checkpoint classes S0..S3 (design section 5.3).

  S0 zero          all weights 0
  S1 planted       healthy policy + a controlled, localised corruption (ground truth known)
  S2 natural       trained on flawed demonstrations / data gap; naturally diagnosable failure; SR in [0.2, 0.6]
  S3 healthy       trained, no single-point fault, SR in [0.6, 0.9]
Selection of S2/S3 candidates uses the dedicated 'select' seed namespace (never dev / hidden).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import envs, mlp
from .envs import EnvParams

N_HIDDEN = 24


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    level: str                     # L1..L5 complexity rung
    params_id: EnvParams           # in-distribution hidden params (public nominal values)
    params_ood: EnvParams          # shifted, NOT disclosed to editors
    n_hidden: int = N_HIDDEN

    @property
    def obs_dim(self) -> int:
        return self.params_id.obs_dim()

    def arch(self):
        return (self.obs_dim, self.n_hidden, 2)

    # ---- unified task interface (shared with official-benchmark backends; Tier-0 behaviour is unchanged) ----
    @property
    def act_dim(self) -> int:
        return 2

    def rollout(self, weights, split: str, start: int, n: int, ood: bool = False, keep_traj: bool = False):
        """Evaluate `weights` on n episodes of namespace `split` (seeds start..start+n-1) -> envs.RolloutResult."""
        return eval_weights(self, weights, split, n, start=start, ood=ood, keep_traj=keep_traj)

    def describe(self) -> str:
        """Environment text shown to the editor (nominal parameters only; OOD never disclosed)."""
        from . import prompts
        return prompts._dyn_text(self)

    def traj_summary(self, res, idx: int) -> dict:
        from . import evaluation
        return evaluation._traj_summary(res, idx)

    def expert_dataset(self, n_episodes: int, seed: int, noise: float = 0.15, **kw):
        """Privileged demonstrations (checkpoint construction and the analytic control ONLY; never shown to editors)."""
        return collect_dataset(self, n_episodes, seed, noise=noise, **kw)

    # ---- hooks used by the generic checkpoint builders ----
    def good_expert_kw(self) -> dict:
        return {"kd": 3.6}

    def train_policy(self, X, Y, seed: int, epochs: int, lr: float = 3e-3, snapshots=None):
        return train_bc(self.arch(), X, Y, seed, epochs=epochs, lr=lr, snapshots=snapshots)

    @property
    def build_cfg(self) -> dict:
        return {"n_ep": 300, "epochs_good": 600, "epochs_s2": 300, "snap_lr": 1e-3, "snap_epochs": list(range(4, 17)),
                "noise": 0.15}

    def s2_candidates(self) -> list:
        """(mode, expert_kw, goal_filter) candidates for natural-failure checkpoints, tried in order."""
        cands = []
        if self.params_id.kind == "delay":
            for comp in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                cands.append(("no_delay_compensation", dict(kd=3.6, comp=comp), None))
        else:
            for kd in (0.6, 0.9, 1.2, 1.6, 2.0, 2.4):
                cands.append(("flawed_damping", dict(kd=kd), None))
        gap = lambda init: init["goal"][:, 0] > -0.1  # demonstrations never saw goals on the far left
        cands.append(("data_gap", dict(kd=3.6), gap))
        return cands


def _mk(task_id, level, kind, ood_changes, **kw):
    pid = EnvParams(kind=kind, **kw)
    return TaskSpec(task_id, level, pid, replace(pid, **ood_changes))


TASKS: dict[str, TaskSpec] = {t.task_id: t for t in [
    _mk("c01-reach", "L2", "reach", dict(mass=1.12, damping=0.088, fmax=0.764), T=45, damping=0.1, fmax=0.8),
    _mk("c02-avoid", "L3", "avoid", dict(mass=1.325, damping=0.0675, fmax=0.87, obstacle_radius=0.219),
        T=60, damping=0.1, fmax=1.0),
    _mk("c03-hold", "L4", "hold", dict(mass=1.16, damping=0.084, fmax=0.752), T=60, hold_steps=25,
        damping=0.1, fmax=0.8),
    _mk("c04-delay", "L5", "delay", dict(delay=6), T=40, delay=5, damping=0.1, fmax=0.8),
]}

# ------------------------------------------------------------------ expert controllers

def expert_action(params: EnvParams, obs: np.ndarray, kp: float = 4.0, kd: float = 3.6,
                  k_rep: float = 0.25, comp: float = 1.0, rep_radius: float = 0.55,
                  tang: float = 1.3) -> np.ndarray:
    """Expert in policy-output units (force = fmax*clip(a)); pre-clip."""
    m, fmax = params.mass, params.fmax
    if params.kind in ("reach", "hold"):
        e, v = obs[:, 0:2], obs[:, 2:4]
        return m * (kp * e - kd * v) / fmax
    if params.kind == "avoid":
        e, v, o = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6]
        f = kp * e - kd * v
        d = np.linalg.norm(o, axis=1, keepdims=True) + 1e-9
        away = -o / d
        # tangent chosen so that it also turns toward the goal side (avoids head-on stalemate)
        tang_vec = np.stack([-away[:, 1], away[:, 0]], axis=1)
        side = np.sign(np.sum(tang_vec * e, axis=1, keepdims=True))
        side = np.where(side == 0, 1.0, side)
        w = np.clip((rep_radius - d) / rep_radius, 0, None) ** 2
        f = f + k_rep * 40.0 * w * (away + tang * side * tang_vec)
        return m * f / fmax
    # delay: crude state predictor from the last two actions, then low-gain PD
    e_d, v_d, a1, a2 = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6], obs[:, 6:8]
    dly = params.delay
    a_est = 0.5 * (a1 + a2)
    acc = params.fmax * a_est / m - params.damping * v_d
    v_now = v_d + comp * envs.DT * dly * acc
    e_now = e_d - comp * envs.DT * dly * (v_d + 0.5 * envs.DT * dly * acc)
    return m * (2.2 * e_now - 2.6 * v_now) / fmax


def collect_dataset(task: TaskSpec, n_episodes: int, seed: int, noise: float = 0.15,
                    expert_kw: dict | None = None, goal_filter=None):
    """Roll out the (noisy) expert, record (obs, clean expert action)."""
    params = task.params_id
    expert_kw = expert_kw or {}
    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**62, n_episodes * 2)]
    if goal_filter is not None:
        init = envs.sample_init(params, seeds)
        keep = goal_filter(init)
        seeds = [s for s, k in zip(seeds, keep) if k][:n_episodes]
    else:
        seeds = seeds[:n_episodes]
    X, Y = [], []

    def pol(obs):
        a = expert_action(params, obs, **expert_kw)
        X.append(obs.copy()); Y.append(np.clip(a, -1, 1))
        return a + rng.normal(0, noise, a.shape)

    envs.rollout(params, pol, seeds)
    return np.concatenate(X), np.concatenate(Y)


# ------------------------------------------------------------------ BC trainer (numpy Adam)

def train_bc(arch, X, Y, seed: int, epochs: int = 800, lr: float = 3e-3, batch: int = 1024, snapshots=None):
    """snapshots: optional list of epoch numbers; if given, returns [(epoch, weights_copy), ...] instead of the final weights."""
    snaps = []
    rng = np.random.default_rng(seed)
    w = mlp.init_weights(*arch, rng=rng)
    m = {k: np.zeros_like(v) for k, v in w.items()}
    s = {k: np.zeros_like(v) for k, v in w.items()}
    n = len(X); t = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            x, y = X[b], Y[b]
            h1 = np.tanh(x @ w["W1"] + w["b1"]); h2 = np.tanh(h1 @ w["W2"] + w["b2"])
            out = h2 @ w["W3"] + w["b3"]
            d3 = 2 * (out - y) / len(b)
            g = {"W3": h2.T @ d3, "b3": d3.sum(0)}
            d2 = (d3 @ w["W3"].T) * (1 - h2 ** 2)
            g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
            d1 = (d2 @ w["W2"].T) * (1 - h1 ** 2)
            g["W1"], g["b1"] = x.T @ d1, d1.sum(0)
            t += 1
            for k in w:
                m[k] = 0.9 * m[k] + 0.1 * g[k]; s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                w[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
        if snapshots is not None and (ep + 1) in snapshots:
            snaps.append((ep + 1, {k: v.copy() for k, v in w.items()}))
        if ep == epochs * 3 // 4:
            lr *= 0.3
    return snaps if snapshots is not None else w


# ------------------------------------------------------------------ evaluation helper

def eval_weights(task: TaskSpec, weights, split: str, n: int, start: int = 0, ood: bool = False,
                 keep_traj: bool = False):
    params = task.params_ood if ood else task.params_id
    seeds = [envs.seed_for(task.task_id, split, start + i) for i in range(n)]
    return envs.rollout(params, envs.make_policy(weights), seeds, keep_traj=keep_traj)


def success_rate(task: TaskSpec, weights, split="select", n=100, ood=False) -> float:
    return float(task.rollout(weights, split, 0, n, ood=ood).success.mean())


# ------------------------------------------------------------------ checkpoint classes

def build_s0(task: TaskSpec) -> dict:
    return {"weights": mlp.init_weights(*task.arch(), zeros=True), "meta": {"class": "S0", "gt": None}}


def _train_variant(task, seed, expert_kw, noise, n_ep, epochs, goal_filter=None):
    X, Y = task.expert_dataset(n_ep, seed, noise=noise, expert_kw=expert_kw, goal_filter=goal_filter)
    if len(X) == 0:                      # candidate produced no demonstrations (infeasible filter)
        return None
    return task.train_policy(X, Y, seed, epochs=epochs)


def build_healthy_good(task, seed: int, epochs: int | None = None) -> dict:
    """High-SR trained policy (base for S1 planted corruption)."""
    c = task.build_cfg
    return _train_variant(task, seed, task.good_expert_kw(), c["noise"], c["n_ep"], epochs or c["epochs_good"])


def build_s3(task: TaskSpec, seed: int, lo=0.6, hi=0.9, tries=8) -> dict | None:
    """Naturally imperfect healthy policy (NO injected noise). Tried in order, SR measured on 'select' seeds:
    (a) 'converged': ordinary converged BC from the good expert (e.g. c02-avoid, whose expert itself is ~0.8);
    (b) 'snapshot': an under-trained snapshot of a BC run (lr 1e-3, snapshot every epoch 4..16); the first snapshot
        whose SR is in [lo, hi] is kept. SR vs epoch is a sharp step, so only some seeds hit the band: infeasible
        combinations are recorded, not forced (no diffuse-noise fallback; see build_s3_noise for the v1 construction)."""
    for j in range(4):
        w = build_healthy_good(task, seed * 7 + j)
        sr = success_rate(task, w)
        if lo <= sr <= hi:
            return {"weights": w, "meta": {"class": "S3", "gt": None, "select_sr": sr, "construction": "converged",
                                           "train_seed": seed * 7 + j}}
    for j in range(tries):
        c = task.build_cfg
        X, Y = task.expert_dataset(c["n_ep"], seed * 100 + j, noise=c["noise"], expert_kw=task.good_expert_kw())
        snaps = task.train_policy(X, Y, seed * 100 + j, epochs=max(c["snap_epochs"]), lr=c["snap_lr"],
                                  snapshots=list(c["snap_epochs"]))
        for ep, w in snaps:
            sr = success_rate(task, w)
            if lo <= sr <= hi:
                return {"weights": w, "meta": {"class": "S3", "gt": None, "select_sr": sr, "construction": "snapshot",
                                               "epoch": ep, "data_seed": seed * 100 + j}}
    return None


def build_s3_noise(task: TaskSpec, seed: int, lo=0.6, hi=0.9) -> dict | None:
    """v1 construction (SUPERSEDED, kept for the record): good policy + diffuse Gaussian noise. Recoverable by a
    closed-form refit, hence not a 'healthy' policy; no longer used in the manifest."""
    base = build_healthy_good(task, seed)
    rng = np.random.default_rng(seed + 12345)
    for sigma in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6):
        for rep in range(3):
            w = {k: v + rng.normal(0, sigma * (v.std() + 1e-9), v.shape) for k, v in base.items()}
            sr = success_rate(task, w)
            if lo <= sr <= hi:
                return {"weights": w, "meta": {"class": "S3", "gt": None, "select_sr": sr,
                                               "construction": "diffuse_noise", "sigma": sigma}}
    return None


def build_s2(task, seed: int, lo=0.2, hi=0.6) -> dict | None:
    """Natural failure: imitation of flawed demonstrations or a data gap (candidates are task-specific)."""
    c = task.build_cfg
    for j, (kind, kw, gf) in enumerate(task.s2_candidates()):
        w = _train_variant(task, seed * 100 + j, kw, c["noise"], c["n_ep"], c["epochs_s2"], goal_filter=gf)
        if w is None:
            continue
        sr = success_rate(task, w)
        if lo <= sr <= hi:
            return {"weights": w, "meta": {"class": "S2", "gt": {"mode": kind, **kw}, "select_sr": sr}}
    return None


CORRUPTIONS = ("weak_axis", "bias_shift", "unit_sign_flip")
STRENGTHS = {  # mildest first; builder takes the first that drops SR into the diagnosable band
    "weak_axis": (0.7, 0.5, 0.3, 0.15),
    "bias_shift": (0.05, 0.1, 0.15, 0.2, 0.27, 0.35),
    "unit_sign_flip": (1, 2, 3),
}


def corrupt(weights: dict, kind: str, seed: int, strength=None):
    """Planted, localised corruption. Returns (weights, ground_truth) with tensor/index ground truth."""
    w = {k: v.copy() for k, v in weights.items()}
    rng = np.random.default_rng(seed)
    strength = strength if strength is not None else STRENGTHS[kind][-1]
    if kind == "weak_axis":
        ax = int(rng.integers(0, w["W3"].shape[1]))
        w["W3"][:, ax] *= strength; w["b3"][ax] *= strength
        gt = {"kind": kind, "strength": strength,
              "tensors": {"W3": [[i, ax] for i in range(w["W3"].shape[0])], "b3": [[ax]]},
              "units_layer2": list(range(w["W3"].shape[0])), "output": ax}
    elif kind == "bias_shift":
        sh = rng.choice([-1, 1], w["b3"].size) * strength
        w["b3"] += sh
        gt = {"kind": kind, "strength": strength, "tensors": {"b3": [[i] for i in range(w["b3"].size)]}, "units_layer2": [],
              "shift": sh.tolist()}
    elif kind == "unit_sign_flip":
        imp = np.abs(w["W3"]).sum(axis=1)
        units = [int(u) for u in np.argsort(-imp)[:int(strength)]]
        for u in units:
            w["W3"][u, :] *= -1
        gt = {"kind": kind, "strength": strength,
              "tensors": {"W3": [[u, c] for u in units for c in range(w["W3"].shape[1])]}, "units_layer2": units}
    else:
        raise ValueError(kind)
    return w, gt


def build_s1(task: TaskSpec, seed: int, kind: str, base_min=0.6, post_lo=0.1, post_hi=0.6) -> dict | None:
    """Healthy trained policy + mildest corruption that lands post-SR in the diagnosable band."""
    for j in range(4):
        base = build_healthy_good(task, seed * 10 + j)
        sr0 = success_rate(task, base)
        if sr0 < base_min:
            continue
        for st in STRENGTHS[kind]:
            w, gt = corrupt(base, kind, seed, st)
            sr1 = success_rate(task, w)
            if post_lo <= sr1 <= post_hi and sr0 - sr1 >= 0.25:
                return {"weights": w, "meta": {"class": "S1", "gt": gt, "orig_weights": base,
                                               "select_sr_before": sr0, "select_sr_after": sr1}}
    return None


def get_task(task_id: str):
    """Task registry: Tier-0 tasks live in TASKS; official-benchmark backends are imported lazily by id prefix
    (they need their own venv: venvs/<benchmark>)."""
    if "@w" in task_id:          # scale ladder: same task and episodes, wider hidden layers, e.g. "c02-avoid@w64"
        import dataclasses
        base_id, w = task_id.split("@w"); base = get_task(base_id)
        return dataclasses.replace(base, task_id=task_id, n_hidden=int(w))
    if task_id in TASKS:
        return TASKS[task_id]
    if task_id.startswith("mw-"):
        from .official import metaworld_task
        return metaworld_task.get_task(task_id)
    if task_id.startswith("ocb-"):
        from .official import ocbench_task
        return ocbench_task.get_task(task_id)
    raise KeyError(task_id)
