"""Claude gateway: one stateless `claude -p` call per turn (Claude Code subscription; no API key).

Isolation (verified by scripts/isolation_probe.py before any experiment):
  * empty cwd (harness-owned scratch dir), --tools "" (no built-in tool), --strict-mcp-config with no MCP config,
  * --setting-sources "" and --disable-slash-commands (no user/project settings, skills, hooks),
  * custom --system-prompt (replaces the default coding-agent prompt), --no-session-persistence.
The raw stream-json event stream of EVERY turn is saved and audited: any tool_use event => audit.tool_free=False.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from .base import (Audit, Gateway, Status, TurnRequest, TurnResult, Usage, render_transcript, sha256_text)

_LIMIT_RE = re.compile(r"(usage limit|rate.?limit|limit reached|quota|too many requests|overloaded|429|resets? (at|in))", re.I)
_AUTH_RE = re.compile(r"(not logged in|please run /login|oauth|authenticat)", re.I)


def claude_version(binary: str = "claude") -> str | None:
    try:
        return subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


class ClaudeCLIGateway(Gateway):
    provider = "claude-cli"

    def __init__(self, binary: str = "claude", extra_args: list[str] | None = None):
        self.binary = binary
        self.extra_args = extra_args or []
        self._version = claude_version(binary)

    # ---------------------------------------------------------------- pure parsing (testable offline)
    def parse_stream(self, lines):
        text_parts: list[str] = []
        n_tool = 0
        usage = Usage()
        stop_reason = None
        override: Status | None = None
        error = None
        init_tools = None
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "system" and ev.get("subtype") == "init":
                init_tools = ev.get("tools", [])
            elif t == "assistant":
                for blk in ev.get("message", {}).get("content", []) or []:
                    if blk.get("type") == "text":
                        text_parts.append(blk.get("text", ""))
                    elif blk.get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
                        n_tool += 1
            elif t == "result":
                u = ev.get("usage", {}) or {}
                det = u.get("output_tokens_details", {}) or {}
                usage = Usage(
                    input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"),
                    thinking_tokens=det.get("thinking_tokens"),
                    cache_read_tokens=u.get("cache_read_input_tokens"),
                    cache_creation_tokens=u.get("cache_creation_input_tokens"),
                    cost_usd=ev.get("total_cost_usd"),
                )
                stop_reason = ev.get("stop_reason")
                res_text = ev.get("result")
                if ev.get("is_error"):
                    error = str(res_text)[:500]
                    if _AUTH_RE.search(error or ""):
                        override = "error"
                    elif _LIMIT_RE.search(error or "") or ev.get("api_error_status") in (429, 529):
                        override = "rate_limited"
                    else:
                        override = "error"
                elif not text_parts and isinstance(res_text, str) and res_text:
                    text_parts.append(res_text)
                if stop_reason == "max_tokens" and override is None:
                    override = "output_budget_exhausted"
        tool_free = (n_tool == 0) and (not init_tools)
        audit = Audit(tool_free=tool_free, n_tool_events=n_tool,
                      notes=f"init_tools={init_tools!r}" if init_tools else "")
        return "".join(text_parts), usage, audit, stop_reason, override, error

    def fixture_streams(self):
        init = json.dumps({"type": "system", "subtype": "init", "tools": []})
        def res(**kw):
            base = {"type": "result", "is_error": False, "result": "hello", "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"thinking_tokens": 3}},
                    "total_cost_usd": 0.0}
            base.update(kw)
            return json.dumps(base)
        asst = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}})
        tool = json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}})
        return {
            "ok": ([init, asst, res()], {"status": "ok", "tool_free": True, "text": "hello"}),
            "tool_use": ([init, tool, asst, res()], {"status": "ok", "tool_free": False}),
            "rate_limited": ([init, res(is_error=True, result="Claude usage limit reached. Resets at 3pm")],
                             {"status": "rate_limited"}),
            "max_tokens": ([init, asst, res(stop_reason="max_tokens")], {"status": "output_budget_exhausted"}),
        }

    # ---------------------------------------------------------------- live call
    def build_cmd(self, req: TurnRequest) -> list[str]:
        return [
            self.binary, "-p",
            "--model", req.model, "--effort", req.effort,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--system-prompt", req.system,
            "--output-format", "stream-json", "--verbose",
            "--no-session-persistence",
            *self.extra_args,
        ]

    def complete(self, req: TurnRequest) -> TurnResult:
        wd = Path(req.workdir)
        wd.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_")) and k != "CLAUDE_CODE_MAX_OUTPUT_TOKENS":
                env.pop(k)
        if req.max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(req.max_output_tokens)
        prompt = render_transcript(req.messages)
        t0 = time.time()
        status: Status = "ok"
        err = None
        lines: list[str] = []
        try:
            cp = subprocess.run(self.build_cmd(req), input=prompt, capture_output=True, text=True,
                                timeout=req.timeout_s, cwd=str(wd), env=env)
            lines = cp.stdout.splitlines()
            if cp.returncode != 0 and not lines:
                status, err = "error", (cp.stderr or "")[:500]
        except subprocess.TimeoutExpired as e:
            status, err = "timeout", "timeout"
            lines = (e.stdout or b"").decode(errors="ignore").splitlines() if isinstance(e.stdout, bytes) else []
        text, usage, audit, stop, override, perr = self.parse_stream(lines)
        if override:
            status, err = override, perr or err
        elif status == "ok" and not text.strip():
            status = "no_output"
        rd = Path(req.raw_dir) if req.raw_dir else wd
        rd.mkdir(parents=True, exist_ok=True)
        raw_path = rd / f"turn{req.turn_idx:02d}.events.jsonl"
        raw_path.write_text("\n".join(lines))
        return TurnResult(status=status, text=text, usage=usage, audit=audit, stop_reason=stop,
                          latency_s=round(time.time() - t0, 2), provider=self.provider, model=req.model,
                          effort=req.effort, cli_version=self._version, system_sha256=sha256_text(req.system),
                          raw_path=str(raw_path), error=err)
