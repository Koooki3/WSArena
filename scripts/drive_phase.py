"""Quota-driven driver for a list of frozen experiment configs (U11). Runs them one after another, serially, in this order, never in parallel.
Each run is guarded inside run_experiment.py (5h < --stop-5h, week < --stop-week, read from the CLI event stream).
When the 5h window is the limit, it waits (session-bound, visible, killable; no cron/daemon) until the window's reset time, takes ONE tiny probe call
to refresh the reading, and continues. When the weekly limit is hit, or a runner fails for any other reason, it STOPS and leaves the state for the owner.
  .venv/bin/python scripts/drive_phase.py --attempt a1 CONFIG [CONFIG ...] [--plan]"""
import argparse, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); import quota
PY = str(ROOT / ".venv/bin/python"); RUN = str(ROOT / "scripts/run_experiment.py")
ap = argparse.ArgumentParser(); ap.add_argument("configs", nargs="+"); ap.add_argument("--attempt", default="a1"); ap.add_argument("--plan", action="store_true"); ap.add_argument("--python", default=None, help="interpreter for the runner (e.g. venvs/metaworld/bin/python for official-tier tasks)")
ap.add_argument("--stop-5h", type=float, default=None); ap.add_argument("--stop-week", type=float, default=None); a = ap.parse_args()
say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

if a.python: PY = str(ROOT / a.python) if not a.python.startswith("/") else a.python
def todo(cfg):
    out = subprocess.run([PY, RUN, cfg, "--attempt", a.attempt, "--dry-run"], capture_output=True, text=True, cwd=ROOT).stdout
    m = re.search(r"plan: (\d+) runs, (\d+) done, (\d+) to do", out); return int(m.group(3)) if m else None
def busy(): return subprocess.run(["pgrep", "-f", "scripts/run_experiment.py"], capture_output=True).returncode == 0

if a.plan:
    for c in a.configs: print(c, "to do:", todo(c))
    print(quota.fmt(quota.latest())); sys.exit(0)
while busy(): say("another run_experiment is active; waiting"); time.sleep(60)
for cfg in a.configs:
    while True:
        n = todo(cfg)
        if n == 0: say(f"{cfg}: complete"); break
        if n is None: say(f"{cfg}: cannot plan; STOP"); sys.exit(2)
        q = quota.latest(); L5, LW = quota.limits(); L5 = a.stop_5h if a.stop_5h is not None else L5; LW = a.stop_week if a.stop_week is not None else LW
        if q and q["seven_day"] >= LW: say(f"weekly {q['seven_day']*100:.0f}% >= {LW*100:.0f}%: STOP"); sys.exit(3)
        if q and q["five_hour"] >= L5:
            wake = (q["five_hour_resets_at"] or time.time()) + 120; say(f"5h {q['five_hour']*100:.0f}% >= {L5*100:.0f}%: waiting for the window reset ({time.strftime('%m-%d %H:%M', time.localtime(wake))})")
            while time.time() < wake: time.sleep(min(300, max(5, wake - time.time())))
            q = quota.probe(); say("after reset (probe): " + (quota.fmt(q) if q else "no reading"));
            if not q or q["five_hour"] >= L5: say("window did not reset as expected: STOP"); sys.exit(4)
            continue
        say(f"{cfg}: {n} runs to do; {quota.fmt(q) if q else 'no reading'}")
        before = n; r = subprocess.run([PY, RUN, cfg, "--attempt", a.attempt, "--quota-note", f"driver auto; guard 5h<{L5*100:.0f} wk<{LW*100:.0f}"], cwd=ROOT)
        after = todo(cfg)
        q = quota.latest(); L5, LW = quota.limits(); L5 = a.stop_5h if a.stop_5h is not None else L5; LW = a.stop_week if a.stop_week is not None else LW; capped = q and (q["five_hour"] >= L5 or q["seven_day"] >= LW)
        if r.returncode != 0 or (after == before and not capped): say(f"{cfg}: runner stopped without progress or with an error (rc={r.returncode}): STOP for the owner"); sys.exit(5)
say("all configs complete")
