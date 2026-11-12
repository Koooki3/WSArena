"""TEMPLATE for the GPT-side gateway. Copy to gpt/codex_cli.py and implement; do NOT edit wsarena/llm/base.py.

Reference harness (read-only, already cloned): external/lm_understand-logs/harness/{codex_config.toml,instructions.md}
and exp5 run.json (`codex exec --strict-config --skip-git-repo-check --json -m <model> -c model_reasoning_effort=... -`).
Requirements = wsarena/llm/base.py docstring. Specifically for Codex:
  * every built-in capability off (see codex_config.toml; shell, files, web, sub-agents, plugins, ...),
  * isolation from ~/.codex user instructions/AGENTS.md (use an isolated CODEX_HOME with only the neutral config),
  * parse_stream(): `codex exec --json` events -> text (agent_message), usage (input/cached/output/reasoning tokens),
    audit.tool_free = no tool items in the stream; map rate/usage limits and context errors onto TurnResult.status,
  * stateless: render with wsarena.llm.base.render_transcript (identical text to the Claude gateway),
  * effort is recorded verbatim (native names); never treat 'high' on GPT as equal compute to 'high' on Claude.
"""
from wsarena.llm.base import Gateway


class CodexCLIGateway(Gateway):          # pragma: no cover - template
    provider = "codex-cli"

    def complete(self, req):
        raise NotImplementedError

    def parse_stream(self, lines):
        raise NotImplementedError

    def fixture_streams(self):
        raise NotImplementedError
