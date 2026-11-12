"""Gateway contract (FROZEN; Claude and Codex gateways must both satisfy it).

One turn = one stateless model call: the full message history is passed in and rendered by the gateway;
nothing is carried in provider-side session state (replayable, auditable). Gateways MUST:
  1. disable every provider-built-in tool (shell, files, web, sub-agents, plugins) and prove it via `audit`;
  2. isolate the call from user-level memory/instructions of the host machine (no CLAUDE.md/AGENTS.md leakage);
  3. record model id, effort, CLI/SDK version and the exact system prompt hash;
  4. map provider failures onto TurnResult.status (never raise for provider-side limits);
  5. never write outside `workdir` and `raw_dir`; the cwd of the provider process MUST be `workdir` (the model can see its path).
See docs/handoff/CODEX_START_HERE_ZH.md and wsarena/llm/conformance.py.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "rate_limited", "output_budget_exhausted", "no_output", "refused", "timeout", "error"]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TurnRequest(BaseModel):
    run_id: str
    turn_idx: int
    model: str                      # provider model id, e.g. claude-opus-5-5
    effort: str                     # provider-native effort name, e.g. medium|high
    system: str
    messages: list[Message]
    max_output_tokens: int | None = None
    timeout_s: int = 1800
    workdir: str                    # NEUTRAL empty scratch dir (random short name; never contains run/task/class/arm names)
    raw_dir: str | None = None      # where the harness stores the raw event stream (may contain identifying names; never shown to the model)


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None   # only if the provider reports it; never estimated
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cost_usd: float | None = None        # as reported by the CLI; informational under a subscription


class Audit(BaseModel):
    tool_free: bool
    n_tool_events: int = 0
    isolation_ok: bool | None = None     # set by the isolation probe, not per call
    notes: str = ""


class TurnResult(BaseModel):
    status: Status
    text: str = ""
    usage: Usage = Field(default_factory=Usage)
    audit: Audit
    stop_reason: str | None = None
    latency_s: float = 0.0
    provider: str
    model: str
    effort: str
    cli_version: str | None = None
    system_sha256: str
    raw_path: str | None = None          # raw event stream saved by the harness
    error: str | None = None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def render_transcript(messages: list[Message]) -> str:
    """Stateless multi-turn rendering for single-prompt CLIs. Identical for every gateway."""
    parts = []
    for m in messages:
        tag = "USER" if m.role == "user" else "ASSISTANT (your earlier reply)"
        parts.append(f"=== {tag} ===\n{m.content}")
    parts.append("=== YOUR NEXT REPLY ===")
    return "\n\n".join(parts)


class Gateway(ABC):
    provider: str

    @abstractmethod
    def complete(self, req: TurnRequest) -> TurnResult: ...

    @abstractmethod
    def parse_stream(self, lines: list[str]) -> tuple[str, Usage, Audit, str | None, Status | None, str | None]:
        """Pure function: provider event stream -> (text, usage, audit, stop_reason, status_override, error).
        Used by the conformance suite with fixtures; must not touch the network."""

    @abstractmethod
    def fixture_streams(self) -> dict[str, tuple[list[str], dict]]:
        """Named recorded/synthetic streams with expected outcomes, used by conformance tests.
        Required names: 'ok', 'tool_use', 'rate_limited', 'max_tokens'."""
