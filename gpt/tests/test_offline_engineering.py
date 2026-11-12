import copy
import json
from pathlib import Path
import time
import subprocess

import numpy as np
import pytest
import yaml

from gpt.codex_cli import CodexCLIGateway, identity_hits
from gpt import identity, quota
from gpt.run_gpt import execute_batch, plan
from gpt.parity import check_pairs
from gpt.analysis import collect
from gpt.checkpoints import OverlayStore
from wsarena.llm.base import Audit, Message, TurnRequest, TurnResult, Usage, sha256_text
from wsarena.llm.conformance import check_gateway
from wsarena.llm.scripted import ScriptedGateway
from wsarena.mlp import init_weights
from wsarena.store import CheckpointStore


def request(tmp_path):
    return TurnRequest(run_id="private-run", turn_idx=1, model="gpt-6.1-sol", effort="medium",
                       system="Follow the request.", messages=[Message(role="user", content="Return a patch.")],
                       max_output_tokens=64000, workdir=str(tmp_path/"w000000"), raw_dir=str(tmp_path/"raw"))


def events(*ev):
    return [json.dumps(e) for e in ev]


def test_gateway_conformance():
    assert check_gateway(CodexCLIGateway()) == []


@pytest.mark.parametrize("kind", ["command_execution", "file_change", "mcp_tool_call", "web_search", "collab_tool_call", "unknown_tool"])
def test_every_tool_item_invalidates_even_without_success(kind):
    stream = events({"type": "item.completed", "item": {"id": "t", "type": kind}}, {"type": "turn.completed", "usage": {}})
    _, _, audit, _, _, _ = CodexCLIGateway().parse_stream(stream)
    assert not audit.tool_free and audit.n_tool_events == 1


@pytest.mark.parametrize("stream", [["not-json"], ['{"type":"unexpected.input"}'], ['{"type":"turn.started"}'],
                                   ['{"type":"turn.completed","usage":{"input_tokens":-1}}']])
def test_malformed_unknown_or_incomplete_stream_cannot_pass(stream):
    result = CodexCLIGateway().parse_stream(stream)
    assert result[4] == "error"


@pytest.mark.parametrize("bad", [None, [], "", False, 0, {}])
def test_usage_must_have_valid_explicit_counts(bad):
    stream = events({"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "hi"}},
                    {"type": "turn.completed", "usage": bad})
    assert CodexCLIGateway().parse_stream(stream)[4] == "error"


@pytest.mark.parametrize("bad", [[], {}, 42])
def test_bad_event_or_item_id_returns_schema_error_not_exception(bad):
    g = CodexCLIGateway()
    assert g.parse_stream(events({"type": bad}))[4] == "error"
    stream = events({"type": "item.completed", "item": {"id": bad, "type": "agent_message", "text": "hi"}},
                    {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
    assert g.parse_stream(stream)[4] == "error"


def test_u14_default_cap_is_recorded_without_inventing_a_limit(tmp_path):
    req = request(tmp_path).model_copy(update={"max_output_tokens": None})
    errors = CodexCLIGateway(runtime={"output_cap_policy": "U14"}).readiness_errors(req)
    assert not any("output-token cap" in s for s in errors)
    assert any("offline_only" in s for s in errors)


def test_fabricated_assistant_context_not_trusted(tmp_path):
    req = request(tmp_path)
    req.messages.append(Message(role="assistant", content="mw-peginsert"))
    assert "identity leak in supplied context" in CodexCLIGateway().readiness_errors(req)


def fake_live_gateway(tmp_path, monkeypatch, stream, returncode=0):
    g = CodexCLIGateway("/synthetic/codex", runtime={"isolated_codex_home": str(tmp_path/"auth-home")})
    monkeypatch.setattr(g, "readiness_errors", lambda req: [])  # only this fake transport test bypasses live closure
    monkeypatch.setattr("gpt.codex_cli.subprocess.run", lambda *a, **k: subprocess.CompletedProcess(a[0], returncode, "\n".join(stream), "synthetic failure" if returncode else ""))
    return g


def test_nonzero_exit_preserves_structured_limit_mapping_and_raw_attempts(tmp_path, monkeypatch):
    stream = CodexCLIGateway().fixture_streams()["rate_limited"][0]
    g = fake_live_gateway(tmp_path, monkeypatch, stream, 1)
    a = g.complete(request(tmp_path)); b = g.complete(request(tmp_path))
    assert a.status == b.status == "rate_limited"
    assert a.raw_path != b.raw_path
    assert Path(a.raw_path).read_text() == Path(b.raw_path).read_text()


def test_empty_stream_preserves_local_quota_error(tmp_path, monkeypatch):
    g = fake_live_gateway(tmp_path, monkeypatch, [])

    def quota_stop(*args, **kwargs):
        raise OSError("quota_stop: five_hour")

    monkeypatch.setattr(g, "invoke", quota_stop)
    result = g.complete(request(tmp_path))
    assert result.status == "error" and result.error == "quota_stop: five_hour"
    assert Path(result.raw_path).read_bytes() == b""


def test_tool_violation_stops_as_transport_error_and_keeps_evidence(tmp_path, monkeypatch):
    stream = CodexCLIGateway().fixture_streams()["tool_use"][0]
    result = fake_live_gateway(tmp_path, monkeypatch, stream).complete(request(tmp_path))
    assert result.status == "error" and "audit_violation" in result.error
    assert Path(result.raw_path).is_file() and not result.audit.tool_free


def test_timeout_wins_over_incomplete_stream_and_raw_is_preserved(tmp_path, monkeypatch):
    g = fake_live_gateway(tmp_path, monkeypatch, [])
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("synthetic", 1, output=b'{"type":"turn.started"}\n')
    monkeypatch.setattr("gpt.codex_cli.subprocess.run", timeout)
    result = g.complete(request(tmp_path))
    assert result.status == "timeout" and Path(result.raw_path).read_bytes()


def test_token_usage_preserves_unknown_reasoning_not_estimated():
    g = CodexCLIGateway()
    text, usage, audit, _, _, _ = g.parse_stream(g.fixture_streams()["ok"][0])
    assert text == "hello" and usage.input_tokens == 10 and usage.output_tokens == 5
    assert usage.cache_read_tokens == 2 and usage.thinking_tokens is None and usage.cost_usd is None
    assert audit.isolation_ok is None


def test_multiple_messages_are_preserved_and_updates_not_duplicated():
    stream = events({"type": "item.updated", "item": {"id": "a", "type": "agent_message", "text": "par"}},
                    {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "part1"}},
                    {"type": "item.completed", "item": {"id": "b", "type": "agent_message", "text": "part2"}},
                    {"type": "turn.completed", "usage": {}})
    assert CodexCLIGateway().parse_stream(stream)[0] == "part1\npart2"


@pytest.mark.parametrize("message,status", [("usage limit reached", "rate_limited"), ("context window exceeded", "error"),
                                           ("authentication failed", "error"), ("max_output_tokens exhausted", "output_budget_exhausted"),
                                           ("refused", "refused")])
def test_failure_mapping(message, status):
    g = CodexCLIGateway()
    assert g.parse_stream(events({"type": "turn.failed", "error": {"message": message}}))[4] == status


def test_complete_offline_never_starts_process(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("live process must not be started")
    monkeypatch.setattr("gpt.codex_cli.subprocess.run", forbidden)
    result = CodexCLIGateway().complete(request(tmp_path))
    assert result.status == "error" and "offline_only" in result.error and "output-token cap" in result.error
    assert not (tmp_path/"raw").exists()


def test_command_is_stateless_neutral_and_has_no_approval_bypass(tmp_path):
    req = request(tmp_path)
    cmd = CodexCLIGateway("/native/codex").build_cmd(req, tmp_path/"resume/w000000/instructions.txt")
    for flag in ("--strict-config", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--json"):
        assert flag in cmd
    assert "resume" not in cmd
    assert not any(arg.startswith("--dangerously") for arg in cmd)
    assert not any("private-run" in arg for arg in cmd if not arg.startswith("model_instructions_file="))
    assert "project_doc_max_bytes=0" in cmd and "web_search=\"disabled\"" in cmd
    assert "features.shell_tool=false" in cmd and "features.multi_agent=false" in cmd
    assert cmd[-1] == "-"


@pytest.mark.parametrize("label", ["mw-peginsert", "gpt-sol61-high", "gpt-astra6-medium", "hid_ood", "checkpoint class S2"])
def test_identity_labels_in_supplied_context_are_blocked(label):
    assert identity_hits(label)


def reading(now=None):
    now = time.time() if now is None else now
    return quota.QuotaReading.from_dict(dict(observed_at=now, five_hour=0.1, week=0.2,
                                            five_hour_resets_at=now+1000, week_resets_at=now+10000, source="synthetic"))


POLICY = dict(stop_five_hour=0.8, stop_week=0.7, authorization_source="synthetic test only", max_age_s=300)


def test_quota_unknown_stale_future_reset_and_thresholds():
    assert not quota.check(None, POLICY, now=100)[0]
    assert not quota.check(reading(100), None, now=100)[0]
    assert not quota.check(reading(100), POLICY, now=401)[0]
    assert not quota.check(reading(101), POLICY, now=100)[0]
    assert "quota_reset" in quota.check(reading(100), {**POLICY, "max_age_s": 2000}, now=1101)[1]
    assert quota.check(reading(100), POLICY, now=100)[0]
    q = quota.QuotaReading.from_dict({**reading(100).__dict__, "five_hour": 0.8})
    assert quota.check(q, POLICY, now=100) == (False, "quota_stop")


@pytest.mark.parametrize("value", [-1, 1.1, float("nan"), True])
def test_invalid_quota_values_rejected(value):
    with pytest.raises(ValueError):
        quota.QuotaReading.from_dict({**reading().__dict__, "week": value})


def test_saved_codex_quota_windows_are_explicit_not_guessed():
    event = {"result": {"rateLimits": {"primary": {"windowDurationMins": 300, "usedPercent": 12, "resetsAt": 1200},
                                       "secondary": {"windowDurationMins": 10080, "usedPercent": 34, "resetsAt": 12000}}}}
    q = quota.from_account_rate_limits(event, observed_at=100)
    assert q.five_hour == 0.12 and q.week == 0.34
    event["result"]["rateLimits"]["primary"]["windowDurationMins"] = 60
    with pytest.raises(ValueError, match="quota windows"):
        quota.from_account_rate_limits(event, observed_at=100)


def test_quota_logger_rejects_claude_log(tmp_path):
    with pytest.raises(ValueError):
        quota.log(tmp_path/"QUOTA_LOG.tsv", "exp", "a1", "start")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "300"])
def test_quota_age_policy_cannot_disable_freshness(bad):
    assert not quota.check(reading(1), {**POLICY, "max_age_s": bad}, now=100)[0]


@pytest.fixture
def project(tmp_path):
    root = tmp_path/"project"
    for name in ("gpt", "docs/handoff", "configs/tasks", "artifacts/checkpoints"):
        (root/name).mkdir(parents=True)
    (root/"gpt/code.py").write_text("# synthetic implementation identity\n")
    store = CheckpointStore(root/"artifacts/checkpoints")
    h = store.put(init_weights(4, 6, 2, zeros=True), {"class": "S0"})
    entry = dict(ckpt_id="c01-reach/s0/seed1", sha256=h, task_id="c01-reach", **{"class": "S0"}, seed=1, status="built")
    (root/"configs/tasks/ckpt_manifest.json").write_text(json.dumps({"entries": [entry]}))
    (root/"docs/handoff/FROZEN_CONTRACT.json").write_text(json.dumps({"version": 12, "files": {}}))
    cfg = dict(exp_id="gpt-synthetic", protocol="WSP-Native", rounds=2, eps_per_round=2, initial_report_episodes=2,
               hidden_n=8, max_output_tokens=None, output_cap_policy="U14", reps=1, cells=[{"ckpt_ids": [entry["ckpt_id"]]}],
               arms=[{"key": "gpt-sol61-medium", "provider": "codex-cli", "model": "gpt-6.1-sol", "effort": "medium"}])
    config = root/"gpt/config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return root, config, cfg, entry


def test_plan_rejects_missing_checkpoint_path_traversal_and_wrong_model(project):
    _, _, cfg, entry = project
    manifest = {entry["ckpt_id"]: entry}
    assert len(plan(cfg, manifest)) == 1
    for mutate in (lambda c: c.update(exp_id="../escape"), lambda c: c["arms"][0].update(key="claude"),
                   lambda c: c["arms"][0].update(model="gpt-6-luna"), lambda c: c["cells"][0].update(ckpt_ids=["missing"])):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError):
            plan(bad, manifest)


def test_resume_identity_detects_changed_gpt_code_and_contract(project):
    root, config, _, _ = project
    first = identity.capture(root, config)
    target = root/"gpt/identity.json"
    identity.bind(target, first)
    identity.bind(target, first)
    (root/"gpt/code.py").write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="resume identity"):
        identity.bind(target, identity.capture(root, config))


def test_shared_checkpoint_store_is_read_only(project):
    root, _, _, entry = project
    shared = root/"artifacts/checkpoints"
    before = sorted(str(p.relative_to(shared)) for p in shared.rglob("*"))
    overlay = OverlayStore(shared, root/"runs/gpt-test/a1/r/checkpoints")
    weights = overlay.get(entry["sha256"])
    weights["W3"][0, 0] = 1.0
    new = overlay.put(weights, {"round": 1})
    assert overlay.has(new) and overlay.get(new)["W3"][0, 0] == 1.0
    assert overlay.meta(new) == {"round": 1}
    assert sorted(str(p.relative_to(shared)) for p in shared.rglob("*")) == before


class FakeGateway(ScriptedGateway):
    def __init__(self, fail_round=None):
        super().__init__("null")
        self.calls = 0
        self.fail_round = fail_round

    def complete(self, req):
        self.calls += 1
        if req.turn_idx == self.fail_round:
            return TurnResult(status="rate_limited", provider="codex-cli", model=req.model, effort=req.effort,
                              audit=Audit(tool_free=True), system_sha256=sha256_text(req.system), error="synthetic usage limit")
        result = super().complete(req)
        result.usage.input_tokens = 10
        result.usage.output_tokens = 5
        return result


def test_quota_stop_before_gateway_or_evaluation(project):
    root, config, _, _ = project
    def forbidden(arm):
        raise AssertionError("gateway not constructed when quota unknown")
    result = execute_batch(root, config, "a1", gateway_factory=forbidden, quota_reader=lambda: None, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert result["status"] == "quota_blocked"
    assert not list((root/"runs").glob("*/*/*/run.json"))
    assert not (root/"runs/QUOTA_LOG.tsv").exists()


def test_quota_reader_error_is_recorded_as_unknown_with_batch_end(project):
    root, config, _, _ = project
    def broken():
        raise ValueError("synthetic broken receipt")
    out = execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=broken, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert out["status"] == "quota_blocked"
    assert "batch_end" in (root/"runs/QUOTA_LOG_gpt.tsv").read_text()


def test_resume_replays_saved_turns_completed_runs_not_rerun(project):
    root, config, _, _ = project
    first, second = FakeGateway(2), FakeGateway()
    result = execute_batch(root, config, "a1", gateway_factory=lambda arm: first, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert result["status"] == "interrupted" and first.calls == 2
    result = execute_batch(root, config, "a1", gateway_factory=lambda arm: second, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert result["status"] == "batch_complete" and second.calls == 1
    execute_batch(root, config, "a1", gateway_factory=lambda arm: second, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert second.calls == 1
    data = collect([root/"runs/gpt-synthetic/a1"])
    assert data["checkpoint_means"][0]["n_runs"] == 1
    assert not (root/"runs/QUOTA_LOG.tsv").exists()
    summary = json.loads(next((root/"runs").glob("*/*/*/run.json")).read_text())
    assert summary["usage_total"]["thinking_tokens"] is None


def test_replay_refuses_altered_spec(project):
    root, config, _, _ = project
    execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    spec_path = next((root/"runs").glob("*/*/*/spec.json"))
    sp = json.loads(spec_path.read_text()); sp["effort"] = "high"; spec_path.write_text(json.dumps(sp))
    with pytest.raises(ValueError, match="saved spec|evidence changed"):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")


def test_resume_rejects_tampered_saved_reply_before_new_call(project):
    root, config, _, _ = project
    execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(2), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    reply = next((root/"runs").glob("*/*/*/turns/01.reply.txt"))
    reply.write_text("synthetic corruption")
    gw = FakeGateway()
    with pytest.raises(ValueError, match="evidence changed"):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: gw, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert gw.calls == 0


def test_analysis_rejects_altered_final_score(project):
    root, config, _, _ = project
    execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    summary = next((root/"runs").glob("*/*/*/run.json"))
    data = json.loads(summary.read_text())
    data["final"]["hidden"]["id"]["delta_sr"] = 0.987
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evidence changed"):
        collect([root/"runs/gpt-synthetic/a1"])


def test_aggregation_includes_explicit_repeat_directories(project):
    root, config, cfg, _ = project
    for exp in ("gpt-main", "gpt-repeat"):
        cfg["exp_id"] = exp; config.write_text(yaml.safe_dump(cfg))
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    data = collect([root/"runs/gpt-main/a1", root/"runs/gpt-repeat/a1"])
    assert len(data["checkpoint_means"]) == 1 and data["checkpoint_means"][0]["n_runs"] == 2
    with pytest.raises(ValueError, match="more than once"):
        collect([root/"runs/gpt-main/a1", root/"runs/gpt-main/a1"])


def pair_dirs(tmp_path):
    fields = dict(task_id="t", ckpt_sha256="a"*64, protocol="WSP-Native", rounds=4,
                  eps_per_round=16, initial_report_episodes=16, hidden_n=200, max_output_tokens=64000)
    result = []
    for name, provider in (("claude", "claude-cli"), ("gpt", "codex-cli")):
        d = tmp_path/name
        (d/"turns").mkdir(parents=True)
        (d/"spec.json").write_text(json.dumps({**fields, "provider": provider}))
        (d/"turns/01.prompt.txt").write_bytes(b"same exact bytes\n")
        result.append(d)
    return result


def test_parity_requires_both_sides_and_matching_budget_and_bytes(tmp_path):
    left, right = pair_dirs(tmp_path)
    assert check_pairs([[left, right]])["ok"]
    assert not check_pairs([])["ok"]
    assert not check_pairs([[left, tmp_path/"missing"]])["ok"]
    assert not check_pairs([[left, left]])["ok"]
    (right/"turns/01.prompt.txt").write_bytes(b"same exact bytes\r\n")
    assert not check_pairs([[left, right]])["ok"]
    (right/"turns/01.prompt.txt").write_bytes(b"same exact bytes\n")
    sp = json.loads((right/"spec.json").read_text()); sp["rounds"] = 8
    (right/"spec.json").write_text(json.dumps(sp))
    assert not check_pairs([[left, right]])["ok"]


def test_u14_cross_provider_cap_difference_requires_disclosure(tmp_path):
    left, right = pair_dirs(tmp_path)
    sp = json.loads((right/"spec.json").read_text()); sp["max_output_tokens"] = None
    (right/"spec.json").write_text(json.dumps(sp))
    assert not check_pairs([[left, right]])["ok"]
    result = check_pairs([[left, right]], output_cap_policy="U14")
    assert result["ok"] and not result["cap_parity_established"]
    assert result["output_cap_disclosures"][0]["requested_caps"] == [64000, None]


def test_policy_observed_max_is_not_an_effective_ceiling():
    from gpt.output_caps import summarize, pause_reason
    data = {"spec": {"max_output_tokens": None}, "rounds": [
        {"usage": {"output_tokens": 100}, "patch_status": "empty_block"},
        {"usage": {"output_tokens": 250}, "patch_status": "turn_failed:output_budget_exhausted"},
    ]}
    r = summarize(data)
    assert r["observed_max_output_tokens"] == 250 and r["output_cap_effective"] == "unverified"
    assert not r["observed_max_is_output_ceiling"] and r["truncated_rounds"] == 1
    assert pause_reason(100, 1, 0) is None
    assert pause_reason(99, 1, 0) == "truncation_rate_exceeds_1_percent"
    assert pause_reason(100, 0, 1) == "actual_output_tokens_unavailable"


def test_missing_output_counters_cannot_be_fabricated_as_zero():
    g = CodexCLIGateway()
    stream = events({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": None}})
    assert g.parse_stream(stream)[4] == "error"


def test_probe_track_is_not_silently_treated_as_native(project):
    _, _, cfg, entry = project
    cfg["arms"][0].update(track="probe", prompt_variant="blind")
    with pytest.raises(ValueError, match="Probe adapter not yet validated"):
        plan(cfg, {entry["ckpt_id"]: entry})


def test_u14_policy_hash_is_bound_on_resume(project):
    root, config, _, _ = project
    policy = root/"docs/handoff/OUTPUT_CAP_POLICY_ZH.md"
    policy.write_text("synthetic U14 policy")
    first = identity.capture(root, config)
    receipt = root/"gpt/id.json"; identity.bind(receipt, first)
    policy.write_text("synthetic changed policy")
    with pytest.raises(ValueError, match="resume identity"):
        identity.bind(receipt, identity.capture(root, config))


class TruncatingGateway(FakeGateway):
    def complete(self, req):
        self.calls += 1
        return TurnResult(status="output_budget_exhausted", provider="codex-cli", model=req.model, effort=req.effort,
                          usage=Usage(input_tokens=10, output_tokens=42),
                          audit=Audit(tool_free=True), system_sha256=sha256_text(req.system))


def test_truncation_pauses_only_affected_arm_and_survives_resume(project):
    root, config, cfg, entry = project
    second = {**entry, "ckpt_id": "c01-reach/s0/seed2", "seed": 2}
    (root/"configs/tasks/ckpt_manifest.json").write_text(json.dumps({"entries": [entry, second]}))
    cfg["cells"][0]["ckpt_ids"].append(second["ckpt_id"])
    cfg["arms"].append({**cfg["arms"][0], "key": "gpt-sol61-high", "effort": "high"})
    config.write_text(yaml.safe_dump(cfg))
    bad, good = TruncatingGateway(), FakeGateway()
    factory = lambda arm: bad if arm["effort"] == "medium" else good
    out = execute_batch(root, config, "a1", gateway_factory=factory, quota_reader=reading, policy=POLICY, max_runs=8, batch_id="synthetic-batch")
    assert out["status"] == "arms_paused" and out["completed"] == 3
    assert out["paused_arms"] == {"gpt-sol61-medium": "truncation_rate_exceeds_1_percent"}
    assert bad.calls == 2 and good.calls == 4
    resumed = execute_batch(root, config, "a1", gateway_factory=factory, quota_reader=reading, policy=POLICY, max_runs=8, batch_id="synthetic-batch")
    assert resumed["status"] == "arms_paused" and resumed["completed"] == 0
    assert bad.calls == 2 and good.calls == 4


def test_missing_tokens_pause_arm_before_next_run(project):
    root, config, _, _ = project
    class Missing(FakeGateway):
        def complete(self, req):
            result = super().complete(req); result.usage.output_tokens = None
            return result
    out = execute_batch(root, config, "a1", gateway_factory=lambda arm: Missing(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert out["status"] == "arms_paused" and out["paused_arms"]["gpt-sol61-medium"] == "actual_output_tokens_unavailable"


def test_truncated_partial_run_pause_is_not_lost_on_interruption(project):
    root, config, _, _ = project
    class TruncateThenLimit(TruncatingGateway):
        def complete(self, req):
            if req.turn_idx == 1:
                return super().complete(req)
            self.calls += 1
            return TurnResult(status="rate_limited", provider="codex-cli", model=req.model, effort=req.effort,
                              audit=Audit(tool_free=True), system_sha256=sha256_text(req.system), error="synthetic limit")
    first = TruncateThenLimit()
    out = execute_batch(root, config, "a1", gateway_factory=lambda arm: first, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert out["status"] == "interrupted" and "gpt-sol61-medium" in out["paused_arms"]
    second = FakeGateway()
    resumed = execute_batch(root, config, "a1", gateway_factory=lambda arm: second, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    assert resumed["status"] == "arms_paused" and second.calls == 0


def test_saved_round_statistics_cannot_be_changed_to_bypass_pause(project):
    root, config, _, _ = project
    execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(2), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    path = next((root/"runs").glob("*/*/*/rounds.json"))
    rounds = json.loads(path.read_text()); rounds[0]["usage"]["output_tokens"] = 100000
    path.write_text(json.dumps(rounds))
    with pytest.raises(ValueError, match="evidence changed"):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")


def test_ledger_reconstruction_not_dependent_on_directory_order(tmp_path, monkeypatch):
    from gpt.output_caps import BatchLedger
    monkeypatch.setattr("gpt.identity.verify_evidence", lambda path: None)
    for label, records in (("bad-first", [(1, 1), (100, 0)]), ("good-first", [(100, 0), (1, 1)])):
        directory = tmp_path/label
        for i, (rounds, truncated) in enumerate(records):
            target = directory/f"{i}-gpt-run"; target.mkdir(parents=True)
            (target/"output_cap.json").write_text(json.dumps({"batch_id": "b", "arm": "gpt-a", "rounds": rounds,
                "truncated_rounds": truncated, "missing_output_token_rounds": 0, "pause_reason": None}))
        ledger = BatchLedger(directory, "b")
        assert ledger.totals["gpt-a"] == {"rounds": 101, "truncated": 1, "missing": 0}
        assert not ledger.paused


def test_completed_record_replaces_partial_contribution(tmp_path, monkeypatch):
    from gpt.output_caps import BatchLedger
    monkeypatch.setattr("gpt.identity.verify_evidence", lambda path: None)
    directory = tmp_path/"batch"
    common = {"batch_id": "b", "arm": "gpt-a", "missing_output_token_rounds": 0, "pause_reason": None}
    clean = directory/"0-gpt-clean"; clean.mkdir(parents=True)
    (clean/"output_cap.json").write_text(json.dumps({**common, "rounds": 100, "truncated_rounds": 0}))
    resumed = directory/"1-gpt-resumed"; resumed.mkdir(parents=True)
    (resumed/"output_cap_abort.json").write_text(json.dumps({**common, "rounds": 1, "truncated_rounds": 1}))
    (resumed/"output_cap.json").write_text(json.dumps({**common, "rounds": 2, "truncated_rounds": 1}))
    ledger = BatchLedger(directory, "b")
    assert ledger.totals["gpt-a"] == {"rounds": 102, "truncated": 1, "missing": 0} and not ledger.paused


def test_terminal_run_without_cap_after_io_failure_cannot_be_skipped(project, monkeypatch):
    from gpt.output_caps import BatchLedger
    root, config, _, _ = project
    original = BatchLedger.record
    def fail(*args, **kw):
        raise OSError("synthetic cap receipt write failure")
    monkeypatch.setattr(BatchLedger, "record", fail)
    with pytest.raises(OSError):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    monkeypatch.setattr(BatchLedger, "record", original)
    with pytest.raises(ValueError, match="terminal run missing U14"):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")


def test_disjoint_config_cannot_reuse_old_attempt_accounting(project):
    root, config, cfg, entry = project
    second = {**entry, "ckpt_id": "c01-reach/s0/seed2", "seed": 2}
    (root/"configs/tasks/ckpt_manifest.json").write_text(json.dumps({"entries": [entry, second]}))
    execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")
    cfg["cells"][0]["ckpt_ids"] = [second["ckpt_id"]]
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="resume identity"):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="synthetic-batch")


def test_resumed_batch_budget_cannot_be_silently_expanded(project):
    root, config, _, _ = project
    execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(2), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="fixed-batch")
    with pytest.raises(ValueError, match="batch size changed"):
        execute_batch(root, config, "a1", gateway_factory=lambda arm: FakeGateway(), quota_reader=reading, policy=POLICY, max_runs=8, batch_id="fixed-batch")


def test_new_batch_name_cannot_implicitly_unpause_arm(project):
    root, config, cfg, entry = project
    second = {**entry, "ckpt_id": "c01-reach/s0/seed2", "seed": 2}
    (root/"configs/tasks/ckpt_manifest.json").write_text(json.dumps({"entries": [entry, second]}))
    cfg["cells"][0]["ckpt_ids"].append(second["ckpt_id"])
    config.write_text(yaml.safe_dump(cfg))
    execute_batch(root, config, "a1", gateway_factory=lambda arm: TruncatingGateway(), quota_reader=reading, policy=POLICY, max_runs=1, batch_id="first-batch")
    second_gateway = FakeGateway()
    result = execute_batch(root, config, "a1", gateway_factory=lambda arm: second_gateway, quota_reader=reading, policy=POLICY, max_runs=1, batch_id="new-batch")
    assert result["status"] == "arms_paused" and second_gateway.calls == 0
