"""Check S1/S2/S3 builders hit their target SR bands on every Tier-0 task (selection seeds only)."""
import time
from wsarena import tasks
for tid, task in tasks.TASKS.items():
    t0 = time.time()
    s2 = tasks.build_s2(task, 1); s3 = tasks.build_s3(task, 1)
    print(f"{tid}: S2={'-' if s2 is None else (round(s2['meta']['select_sr'],2), s2['meta']['gt']['mode'], s2['meta']['gt'].get('kd'))}"
          f" S3={'-' if s3 is None else (round(s3['meta']['select_sr'],2), s3["meta"]["construction"])}", flush=True)
    for k in tasks.CORRUPTIONS:
        s1 = tasks.build_s1(task, 1, k)
        print(f"   S1 {k}: " + ("-" if s1 is None else f"{s1['meta']['select_sr_before']:.2f} -> {s1['meta']['select_sr_after']:.2f}"), flush=True)
    print(f"   ({time.time()-t0:.0f}s)", flush=True)
