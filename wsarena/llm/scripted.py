"""Deterministic scripted 'editors' used as controls and for offline testing. They go through the SAME gateway
contract, parser, budgets and evaluators as an LLM, so accounting is identical (K-S3-02: strongest simple alternatives).

  null           empty edits block every round (nothing changes)
  random         sparse Gaussian perturbation (matched sparsity is set by `frac`)
  oracle_restore S1 only: restores the corrupted entries exactly (upper bound; carries ground-truth knowledge)
  random_greedy  equal-feedback search control: each round perturbs the best-so-far weights (by the same dev SR the
                 editor would see) and keeps the change only through the next round's dev feedback (greedy acceptance)
  gain_greedy    equal-feedback search over ONE parameter: scale the output layer W3 by a scheduled factor on top of the best-so-far
                 weights (the one-line fix a first LLM smoke run found); greedy acceptance via the dev feedback
  tensor_gain_greedy  like gain_greedy but ~6 knobs: scales one output column / the output layer / W2 / W1 per round (cycled), greedy via dev SR
  compiled_pd    deterministic COMPILER (no LLM, no feedback): writes a PD controller into the MLP's near-linear tanh regime from the
                 nominal task parameters (synthesis upper bound for 'knowing the control law'; Tier-0 point-mass tasks only)
  analytic_ridge Analytical-track reference: closed-form ridge refit of the output layer on expert-labelled data
                 (uses privileged expert labels; reported on its own line, never mixed into the Native ranking)
"""

from __future__ import annotations

import numpy as np

from .. import mlp, tasks
from .base import Audit, Gateway, TurnRequest, TurnResult, Usage, sha256_text


def _fmt(x: float) -> str:
    return repr(float(x))


class ScriptedGateway(Gateway):
    provider = "scripted"

    def __init__(self, kind: str, seed: int = 0, frac: float = 0.05, scale: float = 0.5, ridge: float = 1e-3):
        self.kind, self.seed, self.frac, self.scale, self.ridge = kind, seed, frac, scale, ridge
        self.ctx: dict = {}

    def set_context(self, ctx: dict):
        self.ctx = ctx

    # contract methods ------------------------------------------------------------------------
    def parse_stream(self, lines):
        return "\n".join(lines), Usage(), Audit(tool_free=True), None, None, None

    def fixture_streams(self):
        return {"ok": (["hi"], {"status": "ok", "tool_free": True, "text": "hi"}),
                "tool_use": (["hi"], {"skip": True}), "rate_limited": (["hi"], {"skip": True}),
                "max_tokens": (["hi"], {"skip": True})}

    def complete(self, req: TurnRequest) -> TurnResult:
        text = f"Scripted editor '{self.kind}'.\n```edits\n{self._edits()}\n```\n"
        return TurnResult(status="ok", text=text, usage=Usage(input_tokens=0, output_tokens=0), audit=Audit(tool_free=True),
                          provider=self.provider, model=f"scripted-{self.kind}", effort="none",
                          system_sha256=sha256_text(req.system))

    # editors ---------------------------------------------------------------------------------
    def _edits(self) -> str:
        c = self.ctx
        if self.kind == "null" or not c:
            return ""
        if self.kind == "random":
            rng = np.random.default_rng((self.seed * 1_000_003 + c["round"]) % (2**32))
            lines = []
            for k, w in c["weights"].items():
                n = max(1, int(round(self.frac * w.size)))
                flat = rng.choice(w.size, n, replace=False)
                sd = self.scale * (w.std() + 1e-9)
                for f in flat:
                    idx = np.unravel_index(f, w.shape)
                    lines.append(f"{k}[{', '.join(map(str, idx))}] += {_fmt(rng.normal(0, sd))}")
            return "\n".join(lines)
        if self.kind == "random_greedy":
            rng = np.random.default_rng((self.seed * 1_000_003 + 17 * c["round"]) % (2**32))
            sr = c["dev_sr"]; best_i = int(np.argmax(sr))          # ties -> earliest (round 0 = unedited)
            best_w = c["round_weights"][best_i]; base = c["base"]
            lines = ["RESET"]
            for k in base:                                         # re-create best-so-far relative to the base
                for idx in zip(*np.nonzero(best_w[k] != base[k])):
                    lines.append(f"{k}[{', '.join(map(str, idx))}] = {_fmt(best_w[k][idx])}")
            for k, w in best_w.items():                            # + a fresh sparse perturbation
                n = max(1, int(round(self.frac * w.size)))
                for f in rng.choice(w.size, n, replace=False):
                    idx = np.unravel_index(f, w.shape)
                    lines.append(f"{k}[{', '.join(map(str, idx))}] += {_fmt(rng.normal(0, self.scale * (w.std() + 1e-9)))}")
            return "\n".join(lines)
        if self.kind == "gain_greedy":
            sr = c["dev_sr"]; best_i = int(np.argmax(sr)); best_w = c["round_weights"][best_i]; base = c["base"]
            sched = (1.3, 0.75, 1.7, 1.15, 2.0, 0.5, 1.5, 0.9)
            lines = ["RESET"]
            for k in base:
                for idx in zip(*np.nonzero(best_w[k] != base[k])):
                    lines.append(f"{k}[{', '.join(map(str, idx))}] = {_fmt(best_w[k][idx])}")
            lines.append(f"W3[:, :] *= {sched[(c['round'] - 1) % len(sched)]}")
            return "\n".join(lines)
        if self.kind == "tensor_gain_greedy":
            sr = c["dev_sr"]; best_i = int(np.argmax(sr)); best_w = c["round_weights"][best_i]; base = c["base"]
            a = base["W3"].shape[1]; targets = [f"W3[:, {i}]" for i in range(a)] + ["W3[:, :]", "W2[:, :]", "W1[:, :]"]
            factors = (1.3, 0.8, 1.6, 1.15); r = c["round"] - 1
            lines = ["RESET"]
            for k in base:
                for idx in zip(*np.nonzero(best_w[k] != base[k])):
                    lines.append(f"{k}[{', '.join(map(str, idx))}] = {_fmt(best_w[k][idx])}")
            lines.append(f"{targets[r % len(targets)]} *= {factors[(r // len(targets)) % len(factors)]}")
            return "\n".join(lines)
        if self.kind == "compiled_pd":
            if c["round"] > 1: return ""
            p = c["task"].params_id; kd_, kp_ = (2.6, 2.2) if p.kind == "delay" else (3.6, 4.0)
            kp, kd = p.mass * kp_ / p.fmax, p.mass * kd_ / p.fmax; c1, c2 = 0.1, 0.05
            L = ["RESET", "W1[:, :] = 0", "b1[:] = 0", "W2[:, :] = 0", "b2[:] = 0", "W3[:, :] = 0", "b3[:] = 0"]
            for ax in (0, 1):                                   # layer 1: tanh(c1 * e_ax) and tanh(c1 * v_ax)
                L += [f"W1[{ax}, {ax}] = {c1}", f"W1[{2 + ax}, {2 + ax + 0}] = {c1}"]
            L += [f"W2[0, 0] = {c2 * kp}", f"W2[2, 0] = {-c2 * kd}", f"W2[1, 1] = {c2 * kp}", f"W2[3, 1] = {-c2 * kd}"]
            L += [f"W3[0, 0] = {1 / (c1 * c2)}", f"W3[1, 1] = {1 / (c1 * c2)}"]
            return "\n".join(L)
        if self.kind == "oracle_restore":
            orig = c["meta"].get("orig_weights")
            gt = c["meta"].get("gt")
            if not orig or not gt or c["round"] > 1:
                return ""
            lines = ["RESET"]
            for k, lst in gt["tensors"].items():
                for ix in lst:
                    lines.append(f"{k}[{', '.join(map(str, ix))}] = {_fmt(np.asarray(orig[k])[tuple(ix)])}")
            return "\n".join(lines)
        if self.kind == "analytic_ridge":
            if c["round"] > 1:
                return ""
            task = c["task"]
            X, Y = task.expert_dataset(200, seed=777 + self.seed, noise=0.15)
            _, _, h2 = mlp.forward(c["weights"], X, return_hidden=True)
            H = np.hstack([h2, np.ones((len(h2), 1))])
            sol = np.linalg.solve(H.T @ H + self.ridge * np.eye(H.shape[1]), H.T @ Y)
            lines = [f"W3[{i}, {j}] = {_fmt(sol[i, j])}" for i in range(sol.shape[0] - 1) for j in range(sol.shape[1])]
            lines += [f"b3[{j}] = {_fmt(sol[-1, j])}" for j in range(sol.shape[1])]
            return "\n".join(lines)
        raise ValueError(self.kind)
