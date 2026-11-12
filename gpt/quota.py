"""GPT-only quota evidence. Never read Claude streams, credentials or call a model."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


@dataclass(frozen=True)
class QuotaReading:
    observed_at: float
    five_hour: float
    week: float
    five_hour_resets_at: float
    week_resets_at: float
    source: str

    @classmethod
    def from_dict(cls, j):
        q = cls(**j)
        for k in ("five_hour", "week"):
            v = getattr(q, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                raise ValueError(f"invalid utilization: {k}")
        if not isinstance(q.source, str) or not q.source.strip():
            raise ValueError("quota evidence needs its source")
        for k in ("observed_at", "five_hour_resets_at", "week_resets_at"):
            v = getattr(q, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float("inf"):
                raise ValueError(f"invalid timestamp: {k}")
        return q


def check(reading: QuotaReading | None, policy: dict | None, *, now=None):
    now = time.time() if now is None else now
    if reading is None:
        return False, "quota_unknown"
    if not isinstance(policy, dict) or not isinstance(policy.get("authorization_source"), str) or not policy["authorization_source"].strip():
        return False, "GPT quota policy not set; Claude thresholds are not inherited"
    try:
        reading = QuotaReading.from_dict(reading.__dict__)
    except (TypeError, ValueError, AttributeError):
        return False, "quota_reading_invalid"
    age_limit = policy.get("max_age_s", 300)
    if isinstance(age_limit, bool) or not isinstance(age_limit, (int, float)) or not math.isfinite(age_limit) or age_limit <= 0:
        return False, "invalid quota reading max age"
    for k in ("stop_five_hour", "stop_week"):
        v = policy.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
            return False, "invalid GPT quota policy"
    age = now - reading.observed_at
    if not math.isfinite(now) or age < 0 or age > age_limit:
        return False, "quota_reading_stale"
    if min(reading.five_hour_resets_at, reading.week_resets_at) <= now:
        return False, "quota_reset_since_reading: require a fresh reading, never assume zero"
    if reading.five_hour >= policy["stop_five_hour"] or reading.week >= policy["stop_week"]:
        return False, "quota_stop"
    return True, "quota_ok"


def read(path):
    return QuotaReading.from_dict(json.loads(Path(path).read_text()))


def from_account_rate_limits(response: dict, *, observed_at: float):
    """Normalize a saved official account/rateLimits/read response, no RPC here.

    Require the observed window durations; never guess whether primary means
    five hours or whether secondary means one week for a different quota tier.
    """
    result = response.get("result", response)
    windows = result["rateLimits"]
    primary, secondary = windows["primary"], windows["secondary"]
    if primary["windowDurationMins"] != 300 or secondary["windowDurationMins"] != 10080:
        raise ValueError("unexpected GPT quota windows; do not infer units")
    for window in (primary, secondary):
        v = window["usedPercent"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 100:
            raise ValueError("invalid GPT quota percentage")
    return QuotaReading.from_dict({
        "observed_at": observed_at, "five_hour": primary["usedPercent"]/100,
        "week": secondary["usedPercent"]/100, "five_hour_resets_at": primary["resetsAt"],
        "week_resets_at": secondary["resetsAt"], "source": "saved official account/rateLimits/read response",
    })


def log(path: Path, exp: str, attempt: str, event: str, reading=None, note=""):
    if path.name != "QUOTA_LOG_gpt.tsv":
        raise ValueError("GPT quota log must not target Claude's log")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    safe = lambda s: str(s).replace("\t", " ").replace("\n", " ")
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write("time\texp\tattempt\tevent\treading\tnote\n")
        row = [datetime.now(timezone.utc).isoformat(), exp, attempt, event,
               json.dumps(reading.__dict__) if reading else "unknown", note]
        f.write("\t".join(map(safe, row)) + "\n")
