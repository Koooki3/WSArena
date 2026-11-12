"""Prompt construction. The SAME text is used for every checkpoint class (S0..S3): the prompt never says whether
the policy is broken, healthy or a zero network, so the class cannot leak through the wording.
The nominal (ID) physical parameters are public; the OOD parameters are never disclosed."""

from __future__ import annotations

import json

from . import mlp
from .tasks import TaskSpec

SYSTEM_PROMPT = "Answer the user's request."   # neutral, one line (same spirit as the MIT harness)

EDIT_LANGUAGE = """\
HOW TO WRITE WEIGHTS (numbers only; nothing is run for you). Write ONE code block whose opening line is three
backticks followed by the word edits, one statement per line ('#' starts a comment; a list may continue over the next
lines). Tensors: W1, b1, W2, b2, W3, b3.

```edits
W1[2, 0] = 2.5                      # one entry: row 2, column 0
W1[0:4, 3] = [1, 0, -1, 0]          # explicit values along a range (as many values as entries)
b1[0:4] = -1                        # fill a range (a:b is half-open; ':' alone = the whole axis)
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   # a 2-D list for a 2-D block
W3[:, 1] *= 2                       # scale (also += and -=)
RESET                               # (optional line) restore the ORIGINAL weights; later statements then apply
```
Ranges: a:b is half-open, a:b:s takes every s-th index. Row i of W holds the weights from input i; column j belongs to
unit j of the next layer. Statements apply in order and later ones overwrite earlier ones. Only the LAST edits block
of your reply is read. A statement that cannot be read is skipped (the others still apply) and you are told how many
were skipped. Values: plain decimal numbers, |v| <= 1e6. Shapes cannot change. An empty edits block keeps the weights.
"""


def _dyn_text(task: TaskSpec) -> str:
    p = task.params_id
    base = (
        f"A point mass moves in 2-D (arena [-1.2, 1.2]^2; touching the wall clamps position and zeroes velocity). "
        f"Simulation step dt = 0.1 s, episode length T = {p.T} steps. Dynamics each step, with action a = clip(output, -1, 1):\n"
        f"  accel = fmax * a / mass - damping * v ;  v <- v + dt * accel ;  p <- p + dt * v\n"
        f"Nominal parameters: mass = {p.mass}, damping = {p.damping}, fmax = {p.fmax}. "
        f"Each episode starts with v = 0, a random start position and a random goal g at distance >= 0.7. "
        f"Let e = g - p (error vector).\n")
    k = p.kind
    if k == "reach":
        obs = "Observation (4): [e_x, e_y, v_x, v_y]."
        succ = (f"Success: at the final step, |e| < {p.goal_radius} AND |v| < {p.speed_tol}.")
    elif k == "avoid":
        obs = ("Observation (6): [e_x, e_y, v_x, v_y, o_x, o_y] where o = (obstacle centre) - p. A circular obstacle of "
               f"radius {p.obstacle_radius} lies roughly between start and goal.")
        succ = (f"Success: at the final step |e| < {p.goal_radius} AND |v| < {p.speed_tol}, and the point mass never "
                f"came closer than {p.obstacle_radius} to the obstacle centre.")
    elif k == "hold":
        obs = f"Observation (5): [e_x, e_y, v_x, v_y, t/T] where t is the step index."
        succ = (f"Success: for ALL of the last {p.hold_steps} steps, |e| < {p.goal_radius} AND |v| < {p.speed_tol}.")
    else:
        obs = (f"Observation (8): [e_x, e_y, v_x, v_y] as they were {p.delay} steps ago (observation delay), then the two "
               f"previous actions [a_(t-1)_x, a_(t-1)_y, a_(t-2)_x, a_(t-2)_y]. The delay applies to the observation only; "
               f"the actions act immediately.")
        succ = (f"Success: at the final step, |e| < {p.goal_radius} AND |v| < {p.speed_tol} (true state, not delayed).")
    return base + obs + "\n" + succ + "\nOutput (2): raw force commands [u_x, u_y] (clipped to [-1, 1] by the simulator)."


BLIND_NOTE = ("(The current weight values are NOT shown to you: they are fixed numbers you cannot see. Only the architecture above is known. "
              "Your edits are applied to those unseen values: you may overwrite any tensor entirely with '=' statements, or scale / shift entries "
              "with '*=', '+=', '-='. RESET restores the original, unseen weights.)")


def shuffled_weights(weights, seed: int):
    """Permute the entries of every tensor (same shapes, same value multiset, function destroyed): the A3 'shuffled' control."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return {k: v.ravel()[rng.permutation(v.size)].reshape(v.shape) for k, v in weights.items()}


def build_first_user_message(task: TaskSpec, weights, initial_report: dict, rounds: int, per_round_eps: int,
                             weights_view: str = "full", shuffle_seed: int = 0) -> str:
    """weights_view: 'full' (the real weights as text), 'blind' (no values), 'shuffled' (entry-permuted values shown as if real).
    Everything except the weights block is byte-identical across views (prompt parity)."""
    n = mlp.n_params(weights)
    arch = task.arch()
    if weights_view == "blind":
        wblock = BLIND_NOTE
    elif weights_view == "shuffled":
        wblock = mlp.format_weights_text(shuffled_weights(weights, shuffle_seed))
    else:
        wblock = mlp.format_weights_text(weights)
    return f"""\
TASK. Below is a small neural-network policy (an MLP) for a control task. Your goal is to make the policy succeed in as
many episodes as possible by editing its weights. There is no training: no data, no loss, no gradients, no optimiser,
no search. You have no tools (no code execution, no calculator): you read the numbers below and write edits yourself.
Do not rely on anything except this message and the feedback messages that follow.

THE ENVIRONMENT.
{task.describe()}
Scoring: the fraction of episodes that succeed, measured on hidden episodes that you never see (drawn from the same
task; the exact physical parameters of the hidden episodes may differ from the nominal ones above).

THE NETWORK ({n} parameters). Architecture: obs({arch[0]}) -> tanh({arch[1]}) -> tanh({arch[1]}) -> linear({arch[2]}).
  h1 = tanh(obs @ W1 + b1)    W1: {arch[0]} x {arch[1]}, b1: {arch[1]}
  h2 = tanh(h1 @ W2 + b2)     W2: {arch[1]} x {arch[1]}, b2: {arch[1]}
  out = h2 @ W3 + b3          W3: {arch[1]} x {arch[2]}, b3: {arch[2]}
Current weights (rounded to 5 significant digits; your edits are applied to the exact weights):
{wblock}

{EDIT_LANGUAGE}
PROTOCOL. You have {rounds} rounds. In each round you reply with a short analysis and one edits block. Edits are
cumulative (round k edits the weights produced by round k-1; use RESET to go back to the original). After each round
the harness evaluates the edited policy on {per_round_eps} fresh development episodes and sends you a JSON report. The
final weights after round {rounds} are what is scored. Keep each reply concise: your whole reply, including any
thinking, must fit in the output limit; a reply without an edits block counts as 'no edit' for that round.

INITIAL REPORT (the unedited policy on {initial_report['episodes_used']} development episodes):
{json.dumps(initial_report, indent=1)}

Now give your round-1 reply."""


def build_round_message(report: dict, edit_summary: dict, round_next: int, rounds: int) -> str:
    return (f"ROUND {report['round']} RESULT. Edit parsing: {json.dumps(edit_summary)}\n"
            f"Development report for the edited policy:\n{json.dumps(report, indent=1)}\n\n"
            + (f"Give your round-{round_next} reply (analysis + one edits block)." if round_next <= rounds
               else "That was the last round."))
