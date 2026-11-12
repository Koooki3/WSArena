"""Every available gateway must pass the same conformance suite. Add yours to GATEWAYS (Codex: gpt/codex_cli.py)."""
import importlib, pytest
from wsarena.llm.conformance import check_gateway

def _load(spec):
    mod, cls = spec.split(":")
    try:
        return getattr(importlib.import_module(mod), cls)()
    except ModuleNotFoundError:
        return None

GATEWAYS = ["wsarena.llm.claude_cli:ClaudeCLIGateway", "gpt.codex_cli:CodexCLIGateway"]

@pytest.mark.parametrize("spec", GATEWAYS)
def test_gateway_conforms(spec):
    gw = _load(spec)
    if gw is None:
        pytest.skip(f"{spec} not implemented yet (GPT side: implement gpt/codex_cli.py)")
    assert check_gateway(gw) == []
