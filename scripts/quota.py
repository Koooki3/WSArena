"""Autonomous quota reading from the `claude -p` event stream (no credentials touched, no extra tool).
Every model call's stream-json contains a `rate_limit_event` with account-level 5h and 7d utilization and reset times.
Cross-checked against the user's own readings on 2026-11-07 (e.g. user 53% / 40% / 43% vs event 0.52 / 0.41 / 0.43).
  .venv/bin/python scripts/quota.py            # latest reading found in saved run streams (free)
  .venv/bin/python scripts/quota.py --probe    # one tiny model call (haiku, ~1 reply token) for a fresh reading (costs well under 1 point)
  .venv/bin/python scripts/quota.py --log "note"   # append the reading to runs/QUOTA_LOG.tsv (source=cli_event)
Limit: the reading is as fresh as the last model call; calls from other sessions are included (account-level) only up to that call."""
import argparse, glob, json, os, subprocess, sys, tempfile, time, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; TZ = dt.timezone(dt.timedelta(hours=8))

def _parse(line):
    if '"rate_limit_event"' not in line: return None
    try: j = json.loads(line)["rate_limit_info"]
    except Exception: return None
    w = j.get("unifiedWindows", {}); f, s = w.get("five_hour", {}), w.get("seven_day", {})
    if f.get("utilization") is None or s.get("utilization") is None: return None
    return {"five_hour": f["utilization"], "seven_day": s["utilization"], "five_hour_resets_at": f.get("resetsAt"), "seven_day_resets_at": s.get("resetsAt"),
            "status": j.get("status"), "overage": j.get("isUsingOverage")}

def latest(max_files=12):
    files = sorted(glob.glob(str(ROOT / "runs/*/*/*/raw/*.events.jsonl")), key=os.path.getmtime, reverse=True)[:max_files]
    best = None
    for f in files:
        for line in open(f):
            r = _parse(line)
            if r:
                r["source"] = f"stream:{os.path.relpath(f, ROOT)}"; r["read_at"] = os.path.getmtime(f)
                if best is None or r["read_at"] >= best["read_at"]: best = r
        if best: break          # newest file that has one is enough (the last event in it is the freshest)
    return _expire(best)

def _expire(r, now=None):
    """A window that has already reset since the reading was taken is empty (utilization 0) until new usage shows up."""
    if not r: return r
    now = now or time.time()
    for k, ra in (("five_hour", "five_hour_resets_at"), ("seven_day", "seven_day_resets_at")):
        if r.get(ra) and r[ra] < now: r[k] = 0.0; r[ra] = None; r["stale_reset_" + k] = True
    return r

def probe():
    d = tempfile.mkdtemp(prefix="wq")
    p = subprocess.run(["claude", "-p", "Reply with OK.", "--model", "claude-haiku-5-5", "--effort", "low", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
                        "--disable-slash-commands", "--system-prompt", "Answer the user's request.", "--no-session-persistence", "--output-format", "stream-json", "--verbose"],
                       cwd=d, capture_output=True, text=True, timeout=120)
    best = None
    for line in p.stdout.splitlines():
        r = _parse(line)
        if r: best = r
    if best: best.update(source="probe", read_at=time.time())
    return best

def limits():
    """Stop thresholds (fraction). Read from configs/quota_limits.json on every call so a change by the user applies to running drivers at their next check."""
    try: j = json.loads((ROOT / "configs/quota_limits.json").read_text()); return float(j["five_hour"]), float(j["seven_day"])
    except Exception: return 0.80, 0.70      # conservative fallback = global rule

def fmt(r):
    t = lambda x: dt.datetime.fromtimestamp(x, TZ).strftime("%m-%d %H:%M") if x else "-"
    return (f"5h={r['five_hour']*100:.0f}% (resets {t(r['five_hour_resets_at'])})  week={r['seven_day']*100:.0f}% (resets {t(r['seven_day_resets_at'])})  "
            f"read {t(r['read_at'])} via {r['source'].split(':')[0]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--probe", action="store_true"); ap.add_argument("--log", default=""); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    r = probe() if a.probe else latest()
    if not r: sys.exit("no reading available")
    print(json.dumps(r) if a.json else fmt(r))
    if a.log:
        q = ROOT / "runs/QUOTA_LOG.tsv"
        with q.open("a") as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S+08:00')}\t-\t-\tauto_reading\t5h~{r['five_hour']*100:.0f}% wk~{r['seven_day']*100:.0f}%\tsource=cli_event; {a.log}\n")
