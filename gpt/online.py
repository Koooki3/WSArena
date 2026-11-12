"""Explicit entry point for the first audited Native smoke; no formal batches."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from gpt import identity, quota
from gpt.account_limits import clean_env, read_subscription_quota
from gpt.codex_cli import CodexCLIGateway, sha256_file, unexpected_auth_home_content
from gpt.run_gpt import execute_batch, token
from gpt.wire_audit import audit_trace
from wsarena.llm.base import Message, TurnRequest, render_transcript

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT/'gpt/configs/smoke-v1.yaml'


def gates(runtime, policy):
    if sha256_file(runtime['native_binary']) != runtime['binary_sha256']:
        raise ValueError('pinned executable hash changed')
    home = Path(runtime['isolated_codex_home'])
    if unexpected_auth_home_content(home):
        raise ValueError('isolated directory contains instructions/configuration or is linked')
    if (home/'auth.json').is_symlink():
        raise ValueError('linked authentication files are not accepted')
    p = subprocess.run([runtime['native_binary'],'login','status'], env=clean_env(runtime), capture_output=True, text=True, timeout=15)
    if p.returncode or 'Logged in using ChatGPT' not in p.stdout+'\n'+p.stderr:
        raise ValueError('isolated ChatGPT subscription sign-in is required')
    reading = read_subscription_quota(runtime)
    ok, why = quota.check(reading, policy)
    if not ok:
        raise ValueError(why)
    return reading


def canary(runtime, policy, dest):
    # Non-scored, harmless live probe. The identical static command profile has
    # already been exercised with a local loopback fixture and no credentials.
    before = gates(runtime, policy)
    ident = identity.capture(ROOT, SMOKE)
    with dest.open('x', encoding='utf-8') as f:
        json.dump({'status':'started','before':asdict(before),'identity':ident}, f)
    with tempfile.TemporaryDirectory(dir=runtime['neutral_tmp_root'], prefix='w') as tmp:
        wd = Path(tmp); trace = wd/'trace'
        req = TurnRequest(run_id='canary', turn_idx=1, model='gpt-6.1-sol', effort='medium',
                          system='Follow the request exactly.', messages=[Message(role='user', content='Reply with the word ready.')],
                          workdir=str(wd), max_output_tokens=None, timeout_s=75)
        ins = wd/'instructions.txt'; ins.write_text(req.system)
        gw = CodexCLIGateway(binary=runtime['native_binary'], runtime=runtime)
        env = clean_env(runtime); env['CODEX_ROLLOUT_TRACE_ROOT'] = str(trace)
        cp = subprocess.run(gw.build_cmd(req, ins), input=render_transcript(req.messages), cwd=wd,
                            env=env, capture_output=True, text=True, timeout=req.timeout_s)
        # Keep evidence even when semantic admission fails. This is only the
        # fixed harmless canary, never account RPC or authentication output.
        with dest.with_suffix('.events.jsonl').open('x', encoding='utf-8') as f:
            f.write(cp.stdout)
        text, usage, audit, stop, status, error = gw.parse_stream(cp.stdout.splitlines())
        evidence = audit_trace(trace, req)
        with dest.with_suffix('.wire.json').open('x', encoding='utf-8') as f:
            json.dump(evidence, f, indent=2)
        if cp.returncode or status not in (None, 'ok') or not audit.tool_free or text.strip().lower().rstrip('.') != 'ready' or not usage.output_tokens:
            raise ValueError(f'canary failed: exit={cp.returncode}, status={status}, tool_free={audit.tool_free}, output_tokens={usage.output_tokens}')
    after = read_subscription_quota(runtime)
    result = {'status':'passed','before':asdict(before),'after':asdict(after),'identity':ident,
              'wire':evidence,'usage':usage.model_dump(),'finished_at':datetime.now(timezone.utc).isoformat()}
    dest.with_suffix('.passed.json').open('x', encoding='utf-8').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'identity'}))


class AuditedGateway(CodexCLIGateway):
    def __init__(self, runtime, policy):
        super().__init__(binary=runtime['native_binary'], cli_version=runtime['cli_version'], allow_live=True,
                         runtime={**runtime,'live_enabled':True,'isolation_verified':True,'tools_table_verified':True})
        self.policy = policy

    def invoke(self, req, cmd, env, cwd):
        try:
            reading = read_subscription_quota(self.runtime)
        except (ValueError, OSError, KeyError, TypeError) as ex:
            raise OSError('GPT quota unavailable; no model request sent') from ex
        ok, why = quota.check(reading, self.policy)
        if not ok:
            raise OSError(why)
        with tempfile.TemporaryDirectory(dir=self.runtime['neutral_tmp_root'], prefix='wire-') as tmp:
            env['CODEX_ROLLOUT_TRACE_ROOT'] = tmp
            cp = super().invoke(req, cmd, env, cwd)
            try:
                evidence = audit_trace(Path(tmp), req)
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as ex:
                # Preserve ordinary provider errors in raw JSONL; a response
                # with an audit failure must never be used as a valid edit.
                return subprocess.CompletedProcess(cp.args, 1, cp.stdout+'\n'+json.dumps({'type':'error','message':'audit_violation: '+str(ex)})+'\n', cp.stderr)
            rd = Path(req.raw_dir); rd.mkdir(parents=True, exist_ok=True)
            with (rd/f'turn{req.turn_idx:02d}.{evidence["request_sha256"][:20]}.wire.json').open('x', encoding='utf-8') as f:
                json.dump({**evidence,'quota_before':asdict(reading)}, f, indent=2)
            return cp


def smoke(runtime, policy, receipt, attempt):
    d = json.loads(receipt.read_text())
    if d['status'] != 'passed' or d['identity'] != identity.capture(ROOT, SMOKE):
        raise ValueError('canary receipt does not match this exact smoke environment')
    if (d['wire']['model'], d['wire']['effort'], d['wire']['tools_count'], d['wire']['input_exact']) != ('gpt-6.1-sol','medium',0,True):
        raise ValueError('canary did not prove the planned arm and request profile')
    gates(runtime, policy)
    os.environ['TMPDIR'] = runtime['neutral_tmp_root']; tempfile.tempdir = runtime['neutral_tmp_root']
    result = execute_batch(ROOT, SMOKE, attempt, gateway_factory=lambda arm: AuditedGateway(runtime, policy),
                           quota_reader=lambda:read_subscription_quota(runtime), policy=policy, max_runs=1, batch_id='first-smoke')
    print(json.dumps(result))
    if result['status'] != 'batch_complete' or result['completed'] != 1 or result['outcomes'][0]['status'] != 'verified':
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['canary','smoke'])
    ap.add_argument('--attempt', required=True)
    args = ap.parse_args(); token(args.attempt, 'attempt')
    rt = json.loads((ROOT/'gpt/runtime.json').read_text())
    limits = json.loads((ROOT/'configs/quota_limits.json').read_text())
    policy = {'stop_five_hour':limits['five_hour'],'stop_week':limits['seven_day'],
              'authorization_source':'2026-11-11 user: 和claude端一样，请注意额度的5小时限制和周额度限制','max_age_s':300}
    dest = ROOT/'gpt'/f'online-canary-{args.attempt}.json'
    if args.phase == 'canary':
        canary(rt, policy, dest)
    else:
        smoke(rt, policy, dest.with_suffix('.passed.json'), args.attempt)


if __name__ == '__main__':
    main()
