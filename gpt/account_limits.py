"""Read subscription quota through a short-lived Codex stdio connection."""
from __future__ import annotations

import json
from dataclasses import replace
import os
from queue import Empty, Queue
import subprocess
from threading import Thread
import time

from gpt.quota import from_account_rate_limits


def clean_env(runtime):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('CODEX_', 'OPENAI_', 'ANTHROPIC_', 'CLAUDE_'))}
    env.update(CODEX_HOME=runtime['isolated_codex_home'], TMPDIR=runtime['neutral_tmp_root'])
    return env


def read_subscription_quota(runtime):
    cmd = [runtime['native_binary'], 'app-server', '--listen', 'stdio://']
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          cwd=runtime['neutral_tmp_root'], env=clean_env(runtime), text=True) as p:
        lines = Queue()
        def collect():
            for line in p.stdout:
                lines.put(line)
            lines.put(None)
        t = Thread(target=collect, daemon=True)
        t.start()
        end = time.monotonic() + 25
        def send(message):
            p.stdin.write(json.dumps(message)+'\n'); p.stdin.flush()
        def reply(request_id):
            while True:
                try:
                    line = lines.get(timeout=max(0, end-time.monotonic()))
                except Empty as ex:
                    raise ValueError('Codex quota RPC timed out') from ex
                if line is None:
                    raise ValueError('Codex quota RPC ended before replying')
                obj = json.loads(line)
                if obj.get('id') != request_id:
                    continue
                if 'result' not in obj:
                    raise ValueError('Codex quota RPC rejected the read')
                return obj['result']
        try:
            send({'method':'initialize','id':1,'params':{'clientInfo':{'name':'local-benchmark-quota','version':'1.0'}}})
            reply(1)
            send({'method':'initialized'})
            send({'method':'account/rateLimits/read','id':2})
            result = reply(2)
        finally:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=2)
            t.join(timeout=2)
    if result.get('rateLimits', {}).get('rateLimitReachedType') is not None:
        raise ValueError('Codex reports an active quota limit')
    return replace(from_account_rate_limits(result, observed_at=time.time()), source='live Codex account/rateLimits/read')
