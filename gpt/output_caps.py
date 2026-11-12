"""U14 accounting: observations are not evidence of the actual output ceiling."""
from __future__ import annotations
import json
from pathlib import Path

POLICY_ID = "U14"
TRUNCATION_PATCH_STATUS = "turn_failed:output_budget_exhausted"


def valid_tokens(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def summarize(run: dict):
    rounds = run.get("rounds", [])
    tokens = [r.get("usage", {}).get("output_tokens") for r in rounds]
    observed = [n for n in tokens if valid_tokens(n)]
    truncated = sum(r.get("turn_status") == "output_budget_exhausted" or r.get("patch_status") == TRUNCATION_PATCH_STATUS for r in rounds)
    return {
        "policy": POLICY_ID,
        "output_cap_requested": run["spec"].get("max_output_tokens"),
        "output_cap_requested_label": "none (CLI default)" if run["spec"].get("max_output_tokens") is None else str(run["spec"]["max_output_tokens"]),
        "output_cap_effective": "unverified",
        "observed_max_output_tokens": max(observed) if observed else None,
        "observed_max_is_output_ceiling": False,
        "output_tokens_by_round": tokens,
        "rounds": len(rounds), "truncated_rounds": truncated,
        "missing_output_token_rounds": len(tokens)-len(observed),
        "truncation_rate": truncated/len(rounds) if rounds else None,
        "comparison_disclosure": "Effective output ceilings are unverified and not matched across providers; observed output maxima do not establish those ceilings.",
    }


def pause_reason(rounds: int, truncated: int, missing: int):
    if missing:
        return "actual_output_tokens_unavailable"
    # Exact integer comparison implements >1%, so exactly 1/100 is allowed.
    if rounds and truncated * 100 > rounds:
        return "truncation_rate_exceeds_1_percent"
    return None


class BatchLedger:
    """Explicitly named logical batch, preserved across interrupted invocations.

    A paused arm is never automatically restarted by rerunning the same attempt.
    Its model/effort/variant key is retained independently of other arms.
    """
    def __init__(self, directory: Path, batch_id: str):
        self.batch_id, self.totals, self.paused = batch_id, {}, {}
        self.completed_runs = 0
        from gpt.identity import verify_evidence
        paths = list(directory.glob("*gpt-*/output_cap.json")) + list(directory.glob("*gpt-*/output_cap_abort.json"))
        for parent in sorted({p.parent for p in paths}):
            verify_evidence(parent)
            # Complete observations replace partial contributions from the same
            # run; preserve any actual pause marker without counting twice.
            full, partial = parent/"output_cap.json", parent/"output_cap_abort.json"
            record = json.loads((full if full.exists() else partial).read_text())
            if full.exists() and partial.exists():
                older = json.loads(partial.read_text())
                if older.get("batch_id") == batch_id and older.get("pause_reason"):
                    self.paused[older["arm"]] = older["pause_reason"]
            if record["batch_id"] == batch_id:
                self._add(record, completed=full.exists())
            elif record.get("pause_reason"):
                # Merely choosing a new batch name is not a reviewed unpause.
                self.paused[record["arm"]] = record["pause_reason"]
        # Filesystem order is unrelated to the chronological batch prefix.
        # Evaluate reconstructed totals only once after reading all records.
        for arm, totals in self.totals.items():
            reason = pause_reason(**totals)
            if reason:
                self.paused[arm] = reason

    def _add(self, record, *, completed=True):
        key = record["arm"]
        totals = self.totals.setdefault(key, {"rounds": 0, "truncated": 0, "missing": 0})
        totals["rounds"] += record["rounds"]
        totals["truncated"] += record["truncated_rounds"]
        totals["missing"] += record["missing_output_token_rounds"]
        reason = record.get("pause_reason")
        if reason:
            self.paused[key] = reason
        self.completed_runs += int(completed)

    def record(self, path: Path, run: dict, arm: str, *, aborted=False):
        record = {**summarize(run), "batch_id": self.batch_id, "arm": arm}
        totals = self.totals.get(arm, {"rounds": 0, "truncated": 0, "missing": 0})
        record["pause_reason"] = pause_reason(totals["rounds"] + record["rounds"],
                                             totals["truncated"] + record["truncated_rounds"],
                                             totals["missing"] + record["missing_output_token_rounds"])
        with (path/("output_cap_abort.json" if aborted else "output_cap.json")).open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        self._add(record, completed=not aborted)
        return record
