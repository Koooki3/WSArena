"""Summarise an attempt: runs/SUMMARY/<exp>-<attempt>.{json,md}. Refuses runs with incomplete identity."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.analysis.metrics import load_runs
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("attempt"); a = ap.parse_args()
df = load_runs(ROOT / "runs" / a.exp / a.attempt)
out = ROOT / "runs" / "SUMMARY"; out.mkdir(parents=True, exist_ok=True)
df.drop(columns=["path"]).to_json(out / f"{a.exp}-{a.attempt}.json", orient="records", indent=1)
if "d_sr_id" in df:
    g = df.groupby(["model", "effort", "ckpt_class"]).agg(
        runs=("run_id", "count"), verified=("status", lambda s: (s == "verified").sum()),
        completion=("completion_rate", "mean"), valid_patch=("valid_patch_rate", "mean"),
        dSR_id=("d_sr_id", "mean"), dSR_ood=("d_sr_ood", "mean"), recovery=("recovery_id", "mean"),
        regression=("regression_id", "mean"), frac_changed=("frac_changed", "mean")).round(3)
    md = g.reset_index().to_markdown(index=False)
else:
    md = "(no completed runs)"
(out / f"{a.exp}-{a.attempt}.md").write_text(f"# {a.exp} / {a.attempt}\n\n{md}\n")
print(md)
