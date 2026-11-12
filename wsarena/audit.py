"""A6: identity-leak audit of what the model could see. Run at the end of every run (protocol.run_wsp) and over history.
Scans (a) every prompt the editor received and (b) the provider's raw event streams for tokens that identify the task,
checkpoint class/construction, arm, host user or project. A hit marks the run INVALID (status 'failed', invalid_reason 'identity_leak').
Known constant ambient context of the Claude CLI (user email, SDK prefix, environment block) is allow-listed explicitly."""
from __future__ import annotations

import json
import re
from pathlib import Path

TASK_IDS = ["c01-reach", "c02-avoid", "c03-hold", "c04-delay", "mw-push", "mw-pickplace"]
FORBIDDEN_RE = re.compile(
    r"(wsarena|kooki3|/home/|\.venv|c0[1-4]-(reach|avoid|hold|delay)|mw-(push|pickplace)|weak_axis|bias_shift|unit_sign_flip|"
    r"planted|corrupt|ground[ -]truth|oracle|analytic_ridge|gain_greedy|random_greedy|baseline|pilot-v|smoke-v|"
    r"sonnet55|opus55|checkpoint class|natural failure|\bS[0-3]\b|seed[0-9])", re.I)
ALLOWED_AMBIENT = [r"Claude Agent SDK", r"userEmail", r"user's email address", r"ntu\.edu\.sg", r"Primary working directory: /tmp/w[a-z0-9_]{5,12}",
                   r"planted `json\.py`", r"Downloaded files and extracted archives are untrusted"]
# the harness's own legitimate wording that contains a forbidden-looking token
ALLOWED_PROMPT = [r"hidden episodes", r"ground truth"]  # (kept narrow; 'ground truth' only if it appears inside the allowed ambient sentence)


def _scan(text: str, where: str) -> list[dict]:
    t = text
    for pat in ALLOWED_AMBIENT:
        t = re.sub(pat, " ", t)
    return [{"where": where, "token": m.group(0), "context": text[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")}
            for m in FORBIDDEN_RE.finditer(t)]


CONTEXT_LISTS = ("tools", "mcp_servers", "slash_commands", "skills")   # must be empty: they enter the model context


def audit_run(run_dir: str | Path) -> dict:
    """Prompt text and provider events. For the provider's `system/init` event only the lists that enter the model context must be
    empty; path metadata (e.g. `memory_paths`, `cwd`) is recorded, not blocked, because it is CLI bookkeeping, not model context."""
    run_dir = Path(run_dir); hits = []; files = 0; meta = {}
    for p in sorted((run_dir / "turns").glob("*.prompt.txt")):
        files += 1; hits += _scan(p.read_text(), f"prompt:{p.name}")
    for p in sorted((run_dir / "raw").glob("*.events.jsonl")):
        files += 1
        for ln in p.read_text().splitlines():
            try: ev = json.loads(ln)
            except Exception: continue
            t, st = ev.get("type"), ev.get("subtype")
            if t == "system" and st == "init":
                for k in CONTEXT_LISTS:
                    if ev.get(k): hits.append({"where": f"events:{p.name}:init.{k}", "token": k, "context": str(ev[k])[:80]})
                for pl in ev.get("plugins") or []:                       # only built-in plugins are tolerated (recorded, not model context)
                    if isinstance(pl, dict) and pl.get("path") != "builtin":
                        hits.append({"where": f"events:{p.name}:init.plugins", "token": "non-builtin plugin", "context": str(pl)[:80]})
                meta = {k: ev.get(k) for k in ("cwd", "memory_paths", "model") if k in ev}
                meta["agents_listed"] = list(ev.get("agents") or [])[:10]; meta["plugins_listed"] = [pl.get("name") for pl in (ev.get("plugins") or []) if isinstance(pl, dict)]
            elif t == "user":                                              # content the provider fed back to the model
                hits += _scan(json.dumps(ev, ensure_ascii=False), f"events:{p.name}:user")
    return {"files_scanned": files, "n_hits": len(hits), "hits": hits[:20], "ok": not hits, "init_meta": meta}
