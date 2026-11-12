"""Tiny float64 MLP policy: obs -> tanh -> tanh -> linear (action = clip(out, -1, 1) in the env).

Weights are a plain dict {"W1","b1","W2","b2","W3","b3"}; row i of W holds the weights from input i
(same convention as the MIT logs, so edits transfer): h = tanh(x @ W1 + b1) ...
"""

from __future__ import annotations

import numpy as np

TENSOR_ORDER = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_weights(n_in: int, n_h: int, n_out: int, rng: np.random.Generator | None = None, zeros: bool = False):
    if zeros or rng is None:
        return {
            "W1": np.zeros((n_in, n_h)), "b1": np.zeros(n_h),
            "W2": np.zeros((n_h, n_h)), "b2": np.zeros(n_h),
            "W3": np.zeros((n_h, n_out)), "b3": np.zeros(n_out),
        }
    return {
        "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_h)), "b1": np.zeros(n_h),
        "W2": rng.normal(0, 1 / np.sqrt(n_h), (n_h, n_h)), "b2": np.zeros(n_h),
        "W3": rng.normal(0, 0.1 / np.sqrt(n_h), (n_h, n_out)), "b3": np.zeros(n_out),
    }


def forward(w: dict[str, np.ndarray], x: np.ndarray, return_hidden: bool = False):
    x = np.atleast_2d(x)
    h1 = np.tanh(x @ w["W1"] + w["b1"])
    h2 = np.tanh(h1 @ w["W2"] + w["b2"])
    out = h2 @ w["W3"] + w["b3"]
    if return_hidden:
        return out, h1, h2
    return out


def n_params(w: dict[str, np.ndarray]) -> int:
    return int(sum(v.size for v in w.values()))


def shapes(w: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in w.items()}


def reparameterize(w: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Exact function-preserving reparameterisation of the tanh MLP.

    Valid symmetries for tanh hidden layers: (a) permute hidden units of a layer, (b) flip the sign of a
    hidden unit (tanh is odd): negate its incoming weights+bias and its outgoing weights.
    NOT valid for ReLU scaling or LayerNorm -- do not reuse blindly on other architectures.
    """
    w = {k: v.copy() for k, v in w.items()}
    # layer 1 units
    n1 = w["W1"].shape[1]
    perm = rng.permutation(n1)
    sign = rng.choice([-1.0, 1.0], n1)
    w["W1"] = w["W1"][:, perm] * sign
    w["b1"] = w["b1"][perm] * sign
    w["W2"] = w["W2"][perm, :] * sign[:, None]
    # layer 2 units
    n2 = w["W2"].shape[1]
    perm = rng.permutation(n2)
    sign = rng.choice([-1.0, 1.0], n2)
    w["W2"] = w["W2"][:, perm] * sign
    w["b2"] = w["b2"][perm] * sign
    w["W3"] = w["W3"][perm, :] * sign[:, None]
    return w


def format_weights_text(w: dict[str, np.ndarray], sig: int = 5) -> str:
    """Compact, lossless-enough text rendering shown to the editor (rounded to `sig` significant digits)."""
    lines = []
    for k in TENSOR_ORDER:
        a = w[k]
        lines.append(f"{k} shape={list(a.shape)}")
        if a.ndim == 1:
            lines.append("  " + " ".join(f"{v:.{sig}g}" for v in a))
        else:
            for i, row in enumerate(a):
                lines.append(f"  [{i}] " + " ".join(f"{v:.{sig}g}" for v in row))
    return "\n".join(lines)
