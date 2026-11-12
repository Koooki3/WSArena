"""Codex JSONL gateway. Importing/parsing/building commands never calls a model.

Live execution is closed until separately authorized and isolation is validated.
Under U14 the CLI-default output cap is recorded as unverified, not fabricated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

from wsarena.llm.base import Audit, Gateway, TurnRequest, TurnResult, Usage, render_transcript, sha256_text
from wsarena.audit import _scan

MODELS = {"gpt-6.1-sol", "gpt-6-astra"}
EFFORTS = {"medium", "high"}
LABELS = re.compile(r"mw-peginsert|gpt-(?:sol61|astra6)-(?:medium|high)|hid_(?:id|ood)", re.I)
# Feature keys match the pinned research harness. An unsupported key is an
# offline preflight failure, never an excuse to silently enable a capability.
DISABLED_FEATURES = (
    "code_mode", "collaboration_modes", "shell_tool", "unified_exec",
    "unified_exec_tty", "shell_snapshot", "view_image", "image_generation",
    "multi_agent", "apps", "plugins", "remote_plugin", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "in_app_browser",
    "computer_use", "code_mode_host", "skill_search", "skill_mcp_dependency_install",
    "sleep_tool", "tool_suggest", "goals", "hooks", "worktrees", "workspace_dependencies",
)
SAFE_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error"}
SAFE_ITEMS = {"agent_message", "reasoning"}


def identity_hits(text: str) -> list[str]:
    return [h["token"] for h in _scan(text, "codex-context")] + LABELS.findall(text)


def unexpected_auth_home_content(home: Path) -> bool:
    forbidden = ('AGENTS.md', 'AGENTS.override.md', 'config.toml', 'rules', 'memories')
    if home.is_symlink() or any((home/n).exists() or (home/n).is_symlink() for n in forbidden):
        return True
    skills = home/'skills'
    # The pinned app-server bootstraps its bundled skills even in a new home.
    # No user skills or links are accepted; the wire audit still verifies no
    # skill content was injected into a generation request.
    return skills.is_symlink() or (skills.exists() and (not skills.is_dir() or any(
        p.name != '.system' or not p.is_dir() or p.is_symlink() for p in skills.iterdir())))


def _error_status(error) -> str:
    s = json.dumps(error, ensure_ascii=False).lower()
    if any(x in s for x in ("rate_limit", "rate limit", "usage limit", "usage_limit", "quota exceeded", "limit reached")):
        return "rate_limited"
    if any(x in s for x in ("max_output_tokens", "max_tokens", "output_token_limit", "output budget")):
        return "output_budget_exhausted"
    if "refus" in s:
        return "refused"
    return "error"


class CodexCLIGateway(Gateway):
    provider = "codex-cli"

    def __init__(self, binary: str | Path | None = None, *, allow_live: bool = False,
                 runtime: dict | None = None, cli_version: str | None = None):
        self.binary = str(binary) if binary else "codex"
        self.allow_live = allow_live
        self.runtime = runtime or {}
        self.cli_version = cli_version

    def parse_stream(self, lines):
        text_by_id, usage = {}, Usage()
        tool_items, problems, completed, status, error, stop = set(), [], False, None, None, None
        anonymous = 0
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                if not isinstance(ev, dict):
                    raise ValueError("event must be an object")
            except (ValueError, TypeError):
                problems.append(f"malformed_json_line:{number}")
                continue
            typ = ev.get("type")
            if not isinstance(typ, str):
                problems.append("invalid_event_type")
                continue
            if typ in {"item.started", "item.updated", "item.completed"}:
                item = ev.get("item")
                if not isinstance(item, dict):
                    problems.append("missing_item")
                    continue
                kind = item.get("type")
                if not isinstance(kind, str):
                    problems.append("invalid_item_type")
                    continue
                if kind not in SAFE_ITEMS:
                    tool_items.add(str(item.get("id", f"line-{number}")))
                # Raw assistant output is not scanned for identity words: it may
                # independently hypothesize a label. Only supplied context is.
                if typ == "item.completed" and kind == "agent_message":
                    key = item.get("id")
                    if key is None:
                        anonymous += 1
                        key = f"anonymous-{anonymous}"
                    elif not isinstance(key, str):
                        problems.append("invalid_item_id")
                        continue
                    val = item.get("text", "")
                    if not isinstance(val, str):
                        problems.append("nontext_agent_message")
                    else:
                        text_by_id[key] = val
                continue
            if typ == "turn.completed":
                if completed:
                    problems.append("multiple_turn_completions")
                completed = True
                u = ev.get("usage")
                if not isinstance(u, dict):
                    problems.append("invalid_usage")
                    continue
                if not {"input_tokens", "output_tokens"}.issubset(u):
                    problems.append("missing_usage_counts")
                mapping = {"input_tokens": "input_tokens", "output_tokens": "output_tokens",
                           "cached_input_tokens": "cache_read_tokens", "reasoning_output_tokens": "thinking_tokens"}
                for src, dest in mapping.items():
                    v = u.get(src)
                    if src in {"input_tokens", "output_tokens"} and v is None:
                        problems.append(f"missing_usage_count:{src}")
                    if v is not None:
                        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                            problems.append(f"invalid_usage:{src}")
                        else:
                            setattr(usage, dest, v)
                stop = ev.get("stop_reason")
                if stop is not None and not isinstance(stop, str):
                    problems.append("invalid_stop_reason")
                    continue
                if stop in {"max_tokens", "max_output_tokens", "length"}:
                    status = "output_budget_exhausted"
                continue
            if typ in {"turn.failed", "error"}:
                e = ev.get("error", ev.get("message", "provider error"))
                status, error = _error_status(e), json.dumps(e, ensure_ascii=False)[:1000]
                continue
            if typ not in SAFE_EVENTS:
                # Includes unexpected input/tool/context events. Do not assume
                # an unrecognized provider event is harmless.
                problems.append(f"unsupported_event:{typ}")
        if problems:
            status, error = "error", "; ".join(problems)
        elif not completed and status is None:
            status, error = "error", "missing_turn_completed"
        text = "\n".join(text_by_id.values())
        if completed and not text.strip() and status is None:
            status = "no_output"
        audit = Audit(tool_free=not tool_items and not problems, n_tool_events=len(tool_items),
                      isolation_ok=None, notes="offline structural audit; live isolation requires a separate receipt" + ("; " + "; ".join(problems) if problems else ""))
        return text, usage, audit, stop, status, error

    def build_cmd(self, req: TurnRequest, instructions: Path):
        if req.model not in MODELS or req.effort not in EFFORTS:
            raise ValueError("model/effort outside the four registered GPT arms")
        configs = {
            "model_reasoning_effort": req.effort,
            "model_instructions_file": str(instructions.resolve()),
            "project_doc_max_bytes": 0,
            "developer_instructions": "",
            "web_search": "disabled",
            "agents.enabled": False,
            "model_catalog_json": str(Path(__file__).with_name("model_catalog.json").resolve()),
            "tools.update_plan.enabled": False,
            "tools.experimental_request_user_input.enabled": False,
            "memories.use_memories": False,
            "memories.generate_memories": False,
            "features.memories": False,
            "include_environment_context": False,
            "include_apps_instructions": False,
            "include_collaboration_mode_instructions": False,
            "skills.include_instructions": False,
            **{f"features.{k}": False for k in DISABLED_FEATURES},
        }
        cmd = [self.binary, "exec", "--strict-config", "--ignore-user-config", "--ignore-rules",
               "--ephemeral", "--skip-git-repo-check", "--json", "--color", "never",
               "--sandbox", "read-only", "--model", req.model]
        for k, v in configs.items():
            cmd += ["-c", f"{k}={json.dumps(v)}"]
        # No resume, approval bypass, config profile, user prompt file or run id.
        return cmd + ["-"]

    def readiness_errors(self, req):
        errors = []
        if not self.allow_live:
            errors.append("offline_only: live inference has not been enabled")
        if not Path(self.binary).is_absolute() or not Path(self.binary).is_file():
            errors.append("native CLI binary must be an existing absolute file")
        if self.runtime.get("binary_sha256") != (sha256_file(self.binary) if Path(self.binary).is_file() else None):
            errors.append("CLI binary identity has not been verified")
        if self.runtime.get("isolation_verified") is not True:
            errors.append("live isolation has not been verified")
        auth_home = self.runtime.get("isolated_codex_home")
        if not auth_home or not Path(auth_home).is_absolute():
            errors.append("an isolated CODEX_HOME must be configured; the host authentication store is not reused")
        elif unexpected_auth_home_content(Path(auth_home)):
            errors.append("isolated CODEX_HOME contains user instructions or executable configuration")
        neutral_root = self.runtime.get("neutral_tmp_root")
        output_root = self.runtime.get("allowed_runs_root")
        if not neutral_root or not Path(req.workdir).resolve().is_relative_to(Path(neutral_root).resolve()):
            errors.append("workdir outside the neutral runtime root")
        if not req.raw_dir or not output_root or not Path(req.raw_dir).resolve().is_relative_to(Path(output_root).resolve()):
            errors.append("raw_dir outside the authorized run root")
        if req.max_output_tokens is not None:
            errors.append("requested output-token cap unsupported; U14 requires CLI default (null)")
        if self.runtime.get("output_cap_policy") != "U14":
            errors.append("output cap policy U14 has not been recorded")
        if self.runtime.get("live_enabled") is not True:
            errors.append("live execution is not enabled")
        if self.runtime.get("tools_table_verified") is not True:
            errors.append("actual tool table has not been verified")
        if req.model not in MODELS or req.effort not in EFFORTS:
            errors.append("unregistered model/effort")
        # Previously generated assistant replies are legitimate context, but an
        # arbitrary assistant role must not provide an identity-leak bypass.
        trusted_replies = set()
        if req.raw_dir:
            for p in (Path(req.raw_dir).parent/"turns").glob("*.reply.txt"):
                if p.stem.split(".")[0].isdigit() and int(p.stem.split(".")[0]) < req.turn_idx:
                    trusted_replies.add(p.read_text())
        provided = [m.content for m in req.messages if m.role == "user" or m.content not in trusted_replies | {"(no usable reply)"}]
        if identity_hits(req.system) or any(identity_hits(t) for t in provided):
            errors.append("identity leak in supplied context")
        return errors

    def complete(self, req):
        errors = self.readiness_errors(req)
        if errors:
            return TurnResult(status="error", audit=Audit(tool_free=False, notes="preflight refused before process launch"),
                              provider=self.provider, model=req.model, effort=req.effort,
                              system_sha256=sha256_text(req.system), cli_version=self.cli_version, error="; ".join(errors))
        wd = Path(req.workdir).resolve()
        wd.mkdir(parents=True, exist_ok=True)
        instructions = wd / "instructions.txt"
        instructions.write_text(req.system, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "ANTHROPIC_", "CLAUDE_", "CODEX_"))}
        # Official login must provision this dedicated runtime in a later task.
        # Reusing the host CODEX_HOME also reuses its global AGENTS.md even with
        # --ignore-user-config; do not copy or symlink credential material.
        env["CODEX_HOME"] = self.runtime["isolated_codex_home"]
        env["TMPDIR"] = str(wd)
        t0, status, err = time.monotonic(), None, None
        try:
            cp = self.invoke(req, self.build_cmd(req, instructions), env, wd)
            raw = cp.stdout
            if cp.returncode:
                status, err = "error", (cp.stderr or "CLI process failed")[:1000]
        except subprocess.TimeoutExpired as ex:
            raw = ex.stdout or ""
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            status, err = "timeout", "CLI process timed out"
        except OSError as ex:
            raw, status, err = "", "error", str(ex)
        rd = Path(req.raw_dir or req.workdir)
        rd.mkdir(parents=True, exist_ok=True)
        sequence = 1
        path = rd / f"turn{req.turn_idx:02d}.try{sequence:04d}.events.jsonl"
        while path.exists():
            sequence += 1
            path = rd / f"turn{req.turn_idx:02d}.try{sequence:04d}.events.jsonl"
        with path.open("x", encoding="utf-8") as f:
            f.write(raw)
        try:
            text, usage, audit, stop, override, perr = self.parse_stream(raw.splitlines())
        except Exception as ex:
            text, usage, stop, override, perr = "", Usage(), None, "error", f"parser failure: {type(ex).__name__}"
            audit = Audit(tool_free=False, notes=perr)
        # The transport status wins over an incomplete stream on timeout.
        final_status = "timeout" if status == "timeout" else (override or status or "ok")
        if override and perr and (err is None or raw.strip()):
            err = perr
        if not audit.tool_free and status != "timeout":
            final_status, err = "error", "audit_violation: " + (perr or "tool or unknown context event")
        return TurnResult(status=final_status, text=text, usage=usage, audit=audit,
                          stop_reason=stop, latency_s=time.monotonic()-t0, provider=self.provider,
                          model=req.model, effort=req.effort, cli_version=self.cli_version,
                          system_sha256=sha256_text(req.system), raw_path=str(path), error=err or perr)

    def invoke(self, req, cmd, env, cwd):
        return subprocess.run(cmd, input=render_transcript(req.messages), cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=req.timeout_s)

    def fixture_streams(self):
        j = lambda *events: [json.dumps(e) for e in events]
        message = {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "hello"}}
        end = {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5}}
        return {
            "ok": (j({"type": "thread.started", "thread_id": "synthetic"}, {"type": "turn.started"}, message, end), {"text": "hello", "status": "ok", "tool_free": True}),
            "tool_use": (j({"type": "item.completed", "item": {"id": "t", "type": "command_execution", "command": "id"}}, message, end), {"tool_free": False}),
            "rate_limited": (j({"type": "turn.failed", "error": {"message": "usage limit reached"}}), {"status": "rate_limited"}),
            "max_tokens": (j(message, {**end, "stop_reason": "max_tokens"}), {"status": "output_budget_exhausted"}),
        }


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
