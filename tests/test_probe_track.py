import numpy as np
from wsarena import mlp, prompts
from wsarena.edits import extract_last_edits_block
from wsarena.llm.base import Audit, Gateway, TurnRequest, TurnResult, Usage
from wsarena.llm.probe_gateway import ProbeGateway
import wsarena.probe as P


class Fake(Gateway):
    provider = "fake"
    def __init__(self, replies): self.replies, self.calls = list(replies), []
    def parse_stream(self, lines): raise NotImplementedError
    def fixture_streams(self): return {}
    def complete(self, req):
        self.calls.append(req); t = self.replies.pop(0)
        return TurnResult(status="ok", text=t, usage=Usage(input_tokens=10, output_tokens=5), audit=Audit(tool_free=True), provider="fake", model="m", effort="e", system_sha256="x", latency_s=1.0)


def _req(first):
    return TurnRequest(run_id="r", turn_idx=1, model="m", effort="e", system="s", messages=[{"role": "user", "content": first}], workdir="/tmp")


def _gw(replies, **kw):
    w = mlp.init_weights(6, 12, 2, np.random.default_rng(0)); g = ProbeGateway(Fake(replies), **kw); g.set_context({"weights": w}); return g, w


FIRST = "intro\n" + prompts.BLIND_NOTE + "\nrest"


def test_prompt_swap_and_probe_then_edit():
    g, w = _gw(["look\n```probe\nslice W1[0:1, 0:2]\n```", "now\n```edits\nb1[0] = 1\n```"])
    r = g.complete(_req(FIRST)); assert extract_last_edits_block(r.text) is not None and "PROBE RESULTS" in r.text
    assert prompts.BLIND_NOTE not in g.inner.calls[0].messages[0].content and "read-only probes" in g.inner.calls[0].messages[0].content
    assert r.usage.output_tokens == 10 and r.latency_s == 2.0 and [c.turn_idx for c in g.inner.calls] == [10, 11]


def test_budget_forces_edits_and_no_infinite_loop():
    g, _ = _gw(["```probe\nstats W1 axis=0\n```"] * 3 + ["```edits\n```"], max_subturns=2)
    r = g.complete(_req(FIRST)); assert len(g.inner.calls) == 3 and "Give your edits block now" in g.inner.calls[2].messages[-1].content


def test_no_block_ends_round_and_numbers_budget_persists_across_rounds():
    g, _ = _gw(["just text"]); r = g.complete(_req(FIRST)); assert extract_last_edits_block(r.text) is None and len(g.inner.calls) == 1
    g2, _ = _gw(["```probe\nslice W1[0:2, 0:3]\n```", "```edits\n```", "```probe\nslice W1[0:2, 0:3]\n```", "```edits\n```"], max_numbers=10)
    g2.complete(_req(FIRST)); assert g2.used_numbers == 6; g2.complete(_req(FIRST)); assert g2.used_numbers == 6    # second slice (6 more) exceeds 10 -> refused


def test_requires_blind_prompt():
    g, _ = _gw(["x"]);
    try: g.complete(_req("no note here")); assert False
    except RuntimeError: pass
