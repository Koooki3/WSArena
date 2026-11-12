"""Execute the existing eight-run Native Pilot in a single bounded batch."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import yaml

from gpt import identity
from gpt.account_limits import read_subscription_quota
from gpt.online import ROOT, AuditedGateway, gates
from gpt.run_gpt import execute_batch, plan, token

CONFIG = ROOT/'gpt/configs/pilot-v1.yaml'
SMOKE = ROOT/'runs/gpt-smoke-v1/native-smoke-20261111c/c01-reach__s1-weak_axis-seed1__gpt-sol61-medium__r0'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    token(args.attempt, 'attempt')
    ident = identity.capture(ROOT, CONFIG)
    identity.verify_evidence(SMOKE)
    smoke = json.loads((SMOKE/'run.json').read_text())
    if smoke['status'] != 'verified' or len(smoke['rounds']) != 4:
        raise ValueError('verified four-round smoke required')
    verified = json.loads((SMOKE/'gpt_identity.json').read_text())
    for name in ('online.py', 'account_limits.py', 'codex_cli.py', 'wire_audit.py', 'runtime.json', 'model_catalog.json'):
        key = f'gpt/{name}'
        if verified['files'][key] != ident['files'][key]:
            raise ValueError(f'audited request profile changed: {key}')
    cfg = yaml.safe_load(CONFIG.read_text())
    manifest = json.loads((ROOT/'configs/tasks/ckpt_manifest.json').read_text())['entries']
    runs = plan(cfg, {entry['ckpt_id']: entry for entry in manifest})
    if len(runs) != 8 or cfg['rounds'] != 4:
        raise ValueError('Pilot must match its eight-run, four-round configuration')
    if args.dry_run:
        print(json.dumps({'runs': [run.run_id for run in runs], 'max_model_rounds': 32}, indent=2))
        return
    runtime = json.loads((ROOT/'gpt/runtime.json').read_text())
    limits = json.loads((ROOT/'configs/quota_limits.json').read_text())
    policy = {'stop_five_hour': limits['five_hour'], 'stop_week': limits['seven_day'],
              'authorization_source': '用户要求沿用 Claude 端两项额度停线', 'max_age_s': 300}
    gates(runtime, policy)
    os.environ['TMPDIR'] = runtime['neutral_tmp_root']
    tempfile.tempdir = runtime['neutral_tmp_root']
    result = execute_batch(ROOT, CONFIG, args.attempt, gateway_factory=lambda _: AuditedGateway(runtime, policy),
                           quota_reader=lambda: read_subscription_quota(runtime), policy=policy,
                           max_runs=8, batch_id='native-pilot-v1')
    print(json.dumps(result), flush=True)
    if result['status'] != 'batch_complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
