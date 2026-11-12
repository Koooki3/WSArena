"""Gateway conformance suite: every provider gateway (Claude, Codex, ...) must pass it unchanged.

Usage in a test:   from wsarena.llm.conformance import check_gateway;  check_gateway(MyGateway())
Offline only: it exercises `parse_stream` on the gateway's own fixture streams and checks static contract rules.
Fixture names required: ok, tool_use, rate_limited, max_tokens.
"""

from __future__ import annotations

from .base import Audit, Gateway, TurnResult, Usage

REQUIRED = ("ok", "tool_use", "rate_limited", "max_tokens")


def check_gateway(gw: Gateway) -> list[str]:
    errs: list[str] = []
    if not getattr(gw, "provider", None):
        errs.append("gateway.provider missing")
    fx = gw.fixture_streams()
    for name in REQUIRED:
        if name not in fx:
            errs.append(f"missing fixture '{name}'")
    for name, (lines, exp) in fx.items():
        if exp.get("skip"):
            continue
        text, usage, audit, stop, override, error = gw.parse_stream(lines)
        if not isinstance(usage, Usage) or not isinstance(audit, Audit):
            errs.append(f"{name}: parse_stream must return Usage and Audit")
            continue
        status = override or "ok"
        if "status" in exp and status != exp["status"]:
            errs.append(f"{name}: status {status!r} != expected {exp['status']!r}")
        if "tool_free" in exp and audit.tool_free != exp["tool_free"]:
            errs.append(f"{name}: tool_free {audit.tool_free} != expected {exp['tool_free']}")
        if "text" in exp and text != exp["text"]:
            errs.append(f"{name}: text mismatch")
    # TurnResult schema smoke
    try:
        TurnResult(status="ok", audit=Audit(tool_free=True), provider="x", model="m", effort="e", system_sha256="0")
    except Exception as e:  # pragma: no cover
        errs.append(f"TurnResult schema broken: {e}")
    return errs
