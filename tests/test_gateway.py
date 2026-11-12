from wsarena.llm.claude_cli import ClaudeCLIGateway
from wsarena.llm.conformance import check_gateway
from wsarena.llm.base import TurnRequest, Message

def test_claude_gateway_conformance():
    assert check_gateway(ClaudeCLIGateway()) == []

def test_claude_command_is_tool_free_and_isolated():
    g = ClaudeCLIGateway()
    req = TurnRequest(run_id="r", turn_idx=1, model="claude-opus-5-5", effort="high", system="S",
                      messages=[Message(role="user", content="hi")], workdir="/tmp/x")
    cmd = g.build_cmd(req)
    i = cmd.index("--tools"); assert cmd[i + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5" and cmd[cmd.index("--effort") + 1] == "high"
    assert "--bare" not in cmd and "--dangerously-skip-permissions" not in cmd
