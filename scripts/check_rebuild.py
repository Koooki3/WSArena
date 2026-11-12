"""Regression: rebuild three Tier-0 checkpoints and require the SAME sha256 as configs/tasks/ckpt_manifest.json (run after any refactor of builders)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from wsarena import tasks
from wsarena.edits import weights_sha256
m={e['ckpt_id']:e for e in json.load(open('configs/tasks/ckpt_manifest.json'))['entries'] if e['status']=='built'}
t=tasks.TASKS['c01-reach']
for cid,fn in (('c01-reach/s2/seed1',lambda:tasks.build_s2(t,1)),('c01-reach/s1-weak_axis/seed1',lambda:tasks.build_s1(t,1,'weak_axis')),('c01-reach/s3/seed1',lambda:tasks.build_s3(t,1))):
    ck=fn(); print(cid, weights_sha256(ck['weights'])==m[cid]['sha256'], flush=True)
