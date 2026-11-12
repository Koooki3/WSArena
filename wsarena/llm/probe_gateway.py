"""Probe track gateway wrapper (P6). Wraps any Gateway; the protocol (protocol.run_wsp) is NOT modified.
Use with prompt_variant="blind": the first message then contains prompts.BLIND_NOTE where the weights would be; this wrapper swaps it for PROBE_NOTE.
Per round the model may spend up to `max_subturns` probe sub-turns (read-only commands answered by wsarena.probe on the CURRENT weights), then must give its edits.
The final TurnResult carries the whole round (all replies + probe answers) as text so later rounds see what was probed; usage and latency are summed;
every sub-call stores its own raw event stream (turn index = 10*round + k) and a probe_log.jsonl is written next to it."""
from __future__ import annotations
import json
from pathlib import Path
from ..edits import extract_last_edits_block
from ..probe import ProbeBudget, answer_probes, last_probe_block
from ..prompts import BLIND_NOTE
from .base import Audit, Gateway, Message, TurnRequest, TurnResult, Usage

PROBE_NOTE = """\
(The weight values are NOT shown: this network is too large to print. You can read it only through read-only probes. To probe, reply with ONE block that starts with three backticks and the word probe,
one command per line; the harness answers and you then continue. Commands (T is a tensor name W1..b3):
  slice T[a:b, c:d]               values of a slice (at most 400 numbers; indices as in the edit language; one index or a:b range per axis)
  stats T axis=0                  per-column (axis=0) or per-row (axis=1) mean, std, max|.| and L2 norm (at most 200 rows)
  topk T k=20                     the k largest |entries| with their indices (k <= 50)
  forward obs=[[...],[...]] layers=1,2   the network's outputs on up to 8 observations of your own choosing, optionally with the first 16 hidden units of layer 1 and/or 2
Nothing can run the environment, train, or search. Limits: at most {P} probe turns per round, at most {C} commands per probe turn, at most {N} numbers returned in total over the whole task;
anything beyond is refused and you must then give your edits. You may probe in some rounds and not in others. When you are ready, give your edits block as described below;
a reply with an edits block ends the round. Your edits are applied to the unseen values: overwrite entries with '=', scale or shift with '*=', '+=', '-='; RESET restores the original weights.)"""


class ProbeGateway(Gateway):
    def __init__(self, inner: Gateway, max_subturns: int = 3, max_cmds_per_turn: int = 4, max_numbers: int = 8000):
        self.inner, self.provider = inner, inner.provider
        self.cfg = dict(max_subturns=max_subturns, max_cmds_per_turn=max_cmds_per_turn, max_numbers=max_numbers)
        self.used_numbers = 0; self.ctx: dict = {}
        self.note = PROBE_NOTE.replace("{P}", str(max_subturns)).replace("{C}", str(max_cmds_per_turn)).replace("{N}", str(max_numbers))

    def set_context(self, ctx):
        self.ctx = ctx
        if hasattr(self.inner, "set_context"): self.inner.set_context(ctx)

    def parse_stream(self, lines): return self.inner.parse_stream(lines)
    def fixture_streams(self): return self.inner.fixture_streams()

    def complete(self, req: TurnRequest) -> TurnResult:
        w = self.ctx.get("weights")
        if w is None: raise RuntimeError("ProbeGateway needs set_context(weights=...) from the protocol before each turn")
        msgs = [m.model_copy() for m in req.messages]
        if BLIND_NOTE not in msgs[0].content: raise RuntimeError("ProbeGateway expects prompt_variant='blind' (BLIND_NOTE not found in the first message)")
        msgs[0] = Message(role="user", content=msgs[0].content.replace(BLIND_NOTE, self.note))
        budget = ProbeBudget(max_subturns=self.cfg["max_subturns"], max_cmds_per_turn=self.cfg["max_cmds_per_turn"], max_numbers=self.cfg["max_numbers"], used_numbers=self.used_numbers)
        parts, usage, lat, tool_free, n_tool = [], Usage(), 0.0, True, 0
        log = Path(req.raw_dir) / "probe_log.jsonl" if req.raw_dir else None
        last: TurnResult | None = None
        for k in range(self.cfg["max_subturns"] + 1):
            last = self.inner.complete(req.model_copy(update={"messages": msgs, "turn_idx": 10 * req.turn_idx + k}))
            for f in ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens", "cache_creation_tokens", "cost_usd"):
                a, b = getattr(usage, f), getattr(last.usage, f); setattr(usage, f, (a or 0) + b if b is not None else a)
            lat += last.latency_s; tool_free &= last.audit.tool_free; n_tool += last.audit.n_tool_events
            if last.status != "ok": break
            parts.append(last.text)
            if extract_last_edits_block(last.text) is not None: break            # edits end the round
            block = last_probe_block(last.text)
            if block is None: break                                              # neither probes nor edits: the round counts as 'no block'
            res = answer_probes(w, block, budget)
            self.used_numbers = budget.used_numbers
            answer = "PROBE RESULTS:\n" + res.text
            parts.append(answer)
            if log:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as f: f.write(json.dumps({"round": req.turn_idx, "subturn": k, "valid": res.n_valid, "invalid": res.n_invalid, "numbers": res.numbers, "refused": res.refused, "commands": res.commands}) + "\n")
            msgs = msgs + [Message(role="assistant", content=last.text), Message(role="user", content=answer + ("\n\nGive your edits block now." if budget.subturns_used >= budget.max_subturns else ""))]
        out = last.model_copy(update={"text": "\n\n".join(parts), "usage": usage, "latency_s": lat, "audit": Audit(tool_free=tool_free, n_tool_events=n_tool, notes=last.audit.notes)})
        return out
