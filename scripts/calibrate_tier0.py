"""Calibration of Tier-0 tasks: expert SR (ID/OOD), healthy BC SR, S2/S3 selection feasibility. Prints a table."""
import sys, time
import numpy as np
from wsarena import tasks, envs, mlp

def expert_sr(task, ood):
    params = task.params_ood if ood else task.params_id
    seeds = [envs.seed_for(task.task_id, "select", i) for i in range(200)]
    r = envs.rollout(params, lambda o: tasks.expert_action(task.params_id, o), seeds)
    return r.success.mean(), r

for tid, task in tasks.TASKS.items():
    t0 = time.time()
    sid, r = expert_sr(task, False); sood, _ = expert_sr(task, True)
    from collections import Counter
    print(f"{tid}: expert SR id={sid:.2f} ood={sood:.2f} fails={dict(Counter(r.failure))}", flush=True)
    w = tasks.build_healthy_good(task, 1)
    print(f"   BC-good SR id={tasks.success_rate(task,w):.2f} ood={tasks.success_rate(task,w,ood=True):.2f}  ({time.time()-t0:.0f}s)", flush=True)
