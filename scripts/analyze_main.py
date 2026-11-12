"""Primary and secondary analysis of the main study (PREREG §11). Fixed before any main-study output exists.
usage: .venv/bin/python scripts/analyze_main.py [--llm main-v1-t0] [--ckpt-filter seed2,seed3] [--out reports/MAIN_RESULTS.md]
Unit of inference = checkpoint; LLM repeats (if any) are averaged inside a checkpoint; hidden-ID dSR, Final-Round track; failed/invalid runs stay in as dSR=0
is NOT assumed: a run that produced no valid patch has its measured dSR (an unedited policy gives 0 by construction).
Primary family (Holm, 12 tests = 4 arms x 3):  C1 L2(S1+S2) dSR vs `random`;  C2 L2 dSR vs `analytic_ridge`;  C3 dSR(S3) - dSR(L2).
Secondary family (Holm, 12 tests): L2 dSR vs `gain_greedy`, `tensor_gain_greedy`, `random_greedy` for each arm.  Everything else is exploratory and labelled so."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.analysis.metrics import load_runs
from wsarena.analysis.stats import holm, sign_flip_pvalue
from wsarena.analysis.mechanism import classify
from wsarena.store import CheckpointStore

ap = argparse.ArgumentParser(); ap.add_argument("--llm", default="main-v1-t0"); ap.add_argument("--seeds", default="2,3")
ap.add_argument("--out", default="reports/MAIN_RESULTS.md"); a = ap.parse_args()
# Names of the scripted controls in the PREREG vs the arm keys on disk. random_greedy = greedy_f02_s01: chosen BEFORE any LLM output because it has the higher
# mean dSR of the two greedy variants on the non-LLM baselines (L2 0.007 vs -0.010; L3 -0.031 vs -0.117; seeds 2,3).
ALIAS = {"random5": "random", "greedy_f02_s01": "random_greedy"}
LLM = ["sonnet55-medium", "sonnet55-high", "opus55-medium", "opus55-high"]
man = {e["sha256"][:12]: e for e in json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"] if e["status"] == "built"}
seeds = {int(s) for s in a.seeds.split(",")}

frames = []
for d in sorted((ROOT / "runs").glob("*")):
    if d.name == a.llm or d.name.startswith("baselines-") or d.name in ("pilot-v0", "pilot-v1"):
        if d.name.startswith("baselines-v1-regress"): continue
        df = load_runs(d)
        if len(df): df["exp"] = d.name; frames.append(df)
df = pd.concat(frames, ignore_index=True)
df = df[df["status"].isin(["verified", "failed"]) & df["ckpt"].isin(man)]
df["arm"] = df["run_id"].str.split("__").str[2].replace(ALIAS); df["seed"] = df["ckpt"].map(lambda c: man[c]["seed"]); df["ckpt_id"] = df["ckpt"].map(lambda c: man[c]["ckpt_id"])
df = df[df["seed"].isin(seeds)]
df["level"] = df["ckpt_class"].map({"S1": "L2", "S2": "L2", "S3": "L3"})
use = df[(df["exp"] == a.llm) | (df["arm"].isin(["random", "analytic_ridge", "gain_greedy", "random_greedy", "tensor_gain_greedy", "null", "oracle_restore"]))]
per = use.groupby(["arm", "task_id", "ckpt_id", "level"])["d_sr_id"].mean().reset_index()
rng = np.random.default_rng(0)

def strat_boot(diffs_by_task, n=5000):
    g = [np.asarray(v) for v in diffs_by_task.values() if len(v)]
    b = np.array([np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in g])) for _ in range(n)])
    return float(np.mean(np.concatenate(g))), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

def paired(arm, ref, level):
    p = per[per["level"] == level].pivot_table(index=["task_id", "ckpt_id"], columns="arm", values="d_sr_id")
    if arm not in p or ref not in p: return None
    p = p.dropna(subset=[arm, ref]); diff = (p[arm] - p[ref]);
    if diff.empty: return None
    m, ci = strat_boot({t: g.to_numpy() for t, g in diff.groupby(level=0)})
    return {"n": len(diff), "diff": m, "ci": ci, "p": sign_flip_pvalue(diff.to_numpy())}

def c3(arm):
    p = per[per["arm"] == arm]; l2 = p[p["level"] == "L2"]; l3 = p[p["level"] == "L3"]
    if l2.empty or l3.empty: return None
    obs = l3["d_sr_id"].mean() - l2["d_sr_id"].mean()
    bs = []
    for _ in range(5000):
        x = l3.groupby("task_id")["d_sr_id"].apply(lambda s: s.to_numpy()[rng.integers(0, len(s), len(s))].mean()).mean()
        y = l2.groupby("task_id")["d_sr_id"].apply(lambda s: s.to_numpy()[rng.integers(0, len(s), len(s))].mean()).mean()
        bs.append(x - y)
    bs = np.array(bs)
    both = pd.concat([l2, l3]); lab = np.r_[np.zeros(len(l2)), np.ones(len(l3))]; tk = both["task_id"].to_numpy(); v = both["d_sr_id"].to_numpy(); cnt = 0; NP = 5000
    for _ in range(NP):      # permute level labels among the checkpoints of each task (null: no L3/L2 difference)
        pl = lab.copy()
        for t_ in set(tk): ix = np.where(tk == t_)[0]; pl[ix] = rng.permutation(pl[ix])
        st_ = np.mean([v[(tk == t_) & (pl == 1)].mean() - v[(tk == t_) & (pl == 0)].mean() for t_ in set(tk) if ((tk == t_) & (pl == 1)).any() and ((tk == t_) & (pl == 0)).any()])
        cnt += abs(st_) >= abs(obs) - 1e-12
    return {"n": f"{len(l3)} vs {len(l2)}", "diff": float(obs), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": (cnt + 1) / (NP + 1)}

out = [f"# Main study results (generated by scripts/analyze_main.py; seeds {sorted(seeds)})", ""]
arms = [x for x in LLM if x in set(per["arm"])]
rows = {}
for arm in arms:
    for ref, tag in (("random", "C1"), ("analytic_ridge", "C2")):
        r = paired(arm, ref, "L2");
        if r: rows[(arm, tag, f"L2 vs {ref}")] = r
    r = c3(arm)
    if r: rows[(arm, "C3", "dSR(S3) - dSR(L2)")] = r
hp = holm({f"{k[0]}|{k[1]}": v["p"] for k, v in rows.items()})
out += ["## Primary comparisons (Holm over the family)", "", "| arm | id | contrast | n clusters | mean diff | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
for k, v in rows.items():
    out.append(f"| {k[0]} | {k[1]} | {k[2]} | {v['n']} | {v['diff']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['p']:.4f} | {hp[f'{k[0]}|{k[1]}']:.4f} |")
sec = {}
for arm in arms:
    for ref in ("gain_greedy", "tensor_gain_greedy", "random_greedy"):
        r = paired(arm, ref, "L2")
        if r: sec[(arm, ref)] = r
hs = holm({f"{k[0]}|{k[1]}": v["p"] for k, v in sec.items()})
out += ["", "## Secondary comparisons (strongest simple alternatives; Holm over this family)", "", "| arm | vs | n | mean diff | 95% CI | p (Holm) |", "|---|---|---|---|---|---|"]
for k, v in sec.items():
    out.append(f"| {k[0]} | {k[1]} | {v['n']} | {v['diff']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {hs[f'{k[0]}|{k[1]}']:.4f} |")
t = per.groupby(["arm", "level"])["d_sr_id"].agg(["mean", "count"]).unstack("level")
out += ["", "## Mean hidden-ID dSR by arm and level (exploratory descriptive)", "", t.round(3).to_markdown()]
llm = use[use["exp"] == a.llm]
if len(llm):
    g = llm.groupby("arm").agg(runs=("run_id", "count"), completion=("completion_rate", "mean"), valid_patch=("valid_patch_rate", "mean"),
                              regress=("regression_id", "mean"), recov=("recovery_id", "mean"), tok_out=("tok_out", "mean"), frac_changed=("frac_changed", "mean"))
    out += ["", "## Run health and cost proxies (exploratory)", "", g.round(3).to_markdown()]
    st = CheckpointStore(ROOT / "artifacts/checkpoints"); mech = []
    for _, r in llm.iterrows():
        j = json.loads(Path(r["path"]).read_text()); mech.append(classify(st.get(j["identity"]["base_sha256"]), st.get(j["identity"]["final_sha256"]))["mechanism"])
    llm = llm.assign(mechanism=mech)
    m = llm.groupby("mechanism").agg(runs=("run_id", "count"), mean_dsr=("d_sr_id", "mean")).round(3)
    out += ["", "## Edit mechanism strata (exploratory; PREREG: report dSR by mechanism, not only pooled)", "", m.to_markdown()]
    rr = llm.groupby(["task_id", "ckpt_id", "arm"])["d_sr_id"].agg(["std", "count"]).dropna()
    rr = rr[rr["count"] >= 2]
    if len(rr): out += ["", f"sigma_run (pooled SD of repeats): {float(np.sqrt((rr['std']**2).mean())):.3f} from {len(rr)} groups"]
Path(ROOT / a.out).write_text("\n".join(out) + "\n"); print("\n".join(out))
