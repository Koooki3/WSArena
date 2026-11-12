"""'Trained' track: conventional ways to make / repair a network, used ONLY as references for the LLM arms (editors stay forbidden to train).
Every trainer returns (weights, resources) with an honest resource ledger: wall seconds, process-CPU seconds, environment episodes consumed
(rollouts that were used to learn), privileged expert labels consumed, optimiser steps. Evaluation of the result is done elsewhere on the
hidden seeds (never on the seeds used for learning: ES uses the 'dev' namespace like the LLM; BC data uses private expert-data seeds)."""
from __future__ import annotations

import time

import numpy as np

from . import envs, mlp
from .tasks import train_bc


def _ledger(t0, c0, **kw):
    return {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0, **kw}


def bc_finetune(task, weights, n_episodes: int, epochs: int, seed: int, lr: float = 1e-3, scratch: bool = False):
    """Behaviour cloning from privileged expert demonstrations (labels). scratch=True -> random init (a 'normally trained' brain)."""
    t0, c0 = time.perf_counter(), time.process_time()
    X, Y = task.expert_dataset(n_episodes, seed, noise=0.15, expert_kw=task.good_expert_kw())
    arch = task.arch()
    if scratch:
        w = task.train_policy(X, Y, seed, epochs=epochs, lr=3e-3)
        steps = epochs * int(np.ceil(len(X) / 1024))
    else:
        # warm start: same Adam loop initialised at `weights`
        w = _train_from(arch, weights, X, Y, seed, epochs, lr)
        steps = epochs * int(np.ceil(len(X) / 1024))
    return w, _ledger(t0, c0, env_episodes=0, expert_labels=int(len(X)), steps=steps, expert_episodes=n_episodes)


def _train_from(arch, w0, X, Y, seed, epochs, lr, batch=1024):
    rng = np.random.default_rng(seed); w = {k: v.copy() for k, v in w0.items()}
    m = {k: np.zeros_like(v) for k, v in w.items()}; s = {k: np.zeros_like(v) for k, v in w.items()}; n = len(X); t = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]; x, y = X[b], Y[b]
            h1 = np.tanh(x @ w["W1"] + w["b1"]); h2 = np.tanh(h1 @ w["W2"] + w["b2"]); out = h2 @ w["W3"] + w["b3"]
            d3 = 2 * (out - y) / len(b); g = {"W3": h2.T @ d3, "b3": d3.sum(0)}
            d2 = (d3 @ w["W3"].T) * (1 - h2 ** 2); g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
            d1 = (d2 @ w["W2"].T) * (1 - h1 ** 2); g["W1"], g["b1"] = x.T @ d1, d1.sum(0); t += 1
            for k in w:
                m[k] = 0.9 * m[k] + 0.1 * g[k]; s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                w[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
    return w


def es_finetune(task, weights, episode_budget: int, seed: int, sigma: float = 0.05, lr: float = 0.1, pop: int = 8, eps_per_cand: int = 2):
    """OpenAI-style evolution strategy (antithetic pairs), reward = mean episode return on 'dev' seeds. No labels, no gradients of the policy.
    Environment-episode budget is counted exactly (each candidate evaluation costs eps_per_cand episodes) and is capped at `episode_budget`."""
    t0, c0 = time.perf_counter(), time.process_time()
    rng = np.random.default_rng(seed); keys = list(weights); shapes = {k: weights[k].shape for k in keys}
    flat = lambda w: np.concatenate([w[k].ravel() for k in keys])
    unflat = lambda v: {k: v[sum(int(np.prod(shapes[j])) for j in keys[:i]): sum(int(np.prod(shapes[j])) for j in keys[:i + 1])].reshape(shapes[k]) for i, k in enumerate(keys)}
    theta = flat(weights); used = 0; gen = 0
    while used + 2 * pop * eps_per_cand <= episode_budget:
        eps = rng.normal(size=(pop, theta.size)); rewards = np.zeros((pop, 2))
        for i in range(pop):
            for j, sgn in enumerate((+1, -1)):
                r = task.rollout(unflat(theta + sgn * sigma * eps[i]), "dev", 10_000 + gen * 100 + (used % 97), eps_per_cand)
                rewards[i, j] = float(r.ret.mean() + 1.0 * r.success.mean()); used += eps_per_cand
        adv = rewards[:, 0] - rewards[:, 1]; adv = adv / (adv.std() + 1e-8)
        theta = theta + lr / (pop * sigma) * (adv[:, None] * eps).sum(0) * sigma; gen += 1
    return unflat(theta), _ledger(t0, c0, env_episodes=used, expert_labels=0, steps=gen, expert_episodes=0)
