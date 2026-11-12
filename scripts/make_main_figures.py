"""Main-study figures (MAIN-01..05) from run.json files. Same style module as the earlier figures; colour follows the entity, effort is fill/marker.
usage: .venv/bin/python scripts/make_main_figures.py   -> reports/figures/{light,dark}/MAIN-0x.{png,pdf,svg} and reports/figures/data/MAIN-0x.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.analysis.metrics import load_runs
from wsarena.viz import style as S
OUT = ROOT / "reports/figures"; (OUT / "data").mkdir(parents=True, exist_ok=True)
man = {e["sha256"][:12]: e for e in json.loads((ROOT / "configs/tasks/ckpt_manifest.json").read_text())["entries"] if e["status"] == "built"}
def get(exp):
    d = load_runs(ROOT / "runs" / exp); d = d[d["status"].isin(["verified", "failed"])].copy()
    d["arm"] = d["run_id"].str.split("__").str[2]; d["ckpt_id"] = d["ckpt"].map(lambda c: man[c]["ckpt_id"]); d["level"] = d["ckpt_class"].map({"S1": "L2", "S2": "L2", "S3": "L3"}); return d
t0, mw, t0rep, mwrep, blind = get("main-v1-t0"), get("main-v1-mw"), get("main-v1-t0-rep"), get("main-v1-mw-rep"), get("main-v1-t0-blind")
base = pd.concat([get(e) for e in ("baselines-v1", "baselines-gain-v1", "baselines-mw-v1", "baselines-mwpeg-v1")])
base["arm"] = base["arm"].replace({"random5": "random"})
def per(df, ids): return df[df.ckpt_id.isin(ids)].groupby(["arm", "ckpt_id", "level"])["d_sr_id"].mean().reset_index()
def model(a): return "Sonnet 5.5" if a.startswith("sonnet") else "Opus 5.5"
def label(a): return {"sonnet55-medium": "Sonnet·med", "sonnet55-high": "Sonnet·high", "opus55-medium": "Opus·med", "opus55-high": "Opus·high", "random": "random", "gain_greedy": "gain search", "analytic_ridge": "closed-form*", "oracle_restore": "oracle†"}.get(a, a)
def col(a, mode): return S.color(model(a) if a[:3] in ("son", "opu") else a, mode)

def fig_main01(mode):
    S.apply_theme(mode); C = S.CHROME[mode]; fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True); rows = []
    for ax, (name, df, arms) in zip(axs, (("Tier-0 (c02, c03, c04) — 16 checkpoints", t0, ["random", "gain_greedy", "analytic_ridge", "sonnet55-medium", "sonnet55-high", "opus55-medium", "opus55-high"]),
                                          ("Meta-World (push, pick-place, peg-insert) — 17 checkpoints", mw, ["random", "gain_greedy", "analytic_ridge", "sonnet55-medium", "opus55-medium"]))):
        ids = set(df.ckpt_id); p = pd.concat([per(df, ids), per(base[base.arm.isin(["random", "gain_greedy", "analytic_ridge"])], ids)])
        rng = np.random.default_rng(0); ax.axhline(0, color=C["axis"], lw=0.8)
        for i, a in enumerate(arms):
            q = p[p.arm == a]; x = i + rng.uniform(-0.18, 0.18, len(q)); k = col(a, mode)
            for lv, mk, fc in (("L2", "o", k), ("L3", "o", "none")):
                m = q.level == lv; ax.scatter(x[m.to_numpy()], q[m].d_sr_id, s=26, marker=mk, facecolors=fc, edgecolors=k, linewidths=1.2, zorder=3, alpha=0.95)
            ax.plot([i - 0.3, i + 0.3], [q.d_sr_id.mean()] * 2, color=C["ink"], lw=2, zorder=4)
            for _, r in q.iterrows(): rows.append([name.split(" —")[0], a, r.ckpt_id, r.level, r.d_sr_id])
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([label(a) for a in arms], rotation=25, ha="right", fontsize=8.5); ax.set_title(name, fontsize=9.5, loc="left", color=C["ink"]); ax.grid(axis="y", lw=0.4, alpha=0.5)
    axs[0].set_ylabel("change in hidden success rate (edited − unedited)"); axs[0].annotate("filled = fault/failure checkpoint (L2)    hollow = healthy checkpoint (S3)    bar = mean", (0.0, -0.33), xycoords="axes fraction", fontsize=8, color=C["ink2"])
    fig.suptitle("Every dot is one checkpoint: how much each editor raises hidden success", x=0.01, ha="left", fontsize=11, color=C["ink"]); S.footnote(fig, S.FOOT, mode); fig.tight_layout(rect=(0, 0.04, 1, 0.95)); return fig, rows

def fig_main02(mode):
    S.apply_theme(mode); C = S.CHROME[mode]; fig, ax = plt.subplots(figsize=(7.6, 4.6)); rows = []
    full = pd.concat([t0t[t0t.arm.isin(["sonnet55-medium", "opus55-medium"])], t0rep]).groupby(["ckpt_id", "arm"])["d_sr_id"].mean(); bl = blind.assign(arm=blind.arm.str.replace("-blind", "")).groupby(["ckpt_id", "arm"])["d_sr_id"].mean()
    j = pd.concat([full.rename("full"), bl.rename("blind")], axis=1).dropna().reset_index()
    for _, r in j.iterrows():
        k = col(r.arm, mode); ax.plot([0, 1], [r.full, r.blind], color=k, lw=1.3, alpha=0.8, marker="o", ms=4); rows.append([r.ckpt_id, r.arm, r.full, r.blind])
    for a in ("sonnet55-medium", "opus55-medium"):
        q = j[j.arm == a]; ax.plot([0, 1], [q.full.mean(), q.blind.mean()], color=col(a, mode), lw=3.6, zorder=5); S.end_label(ax, 1, q.blind.mean(), f"{label(a)} mean", col(a, mode), mode, dx=8)
    ax.set_xlim(-0.15, 1.45); ax.set_xticks([0, 1]); ax.set_xticklabels(["weights shown", "weights hidden (blind)"]); ax.axhline(0, color=C["axis"], lw=0.8); ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_ylabel("change in hidden success rate"); ax.set_title("Hiding the weights changes little on average\n(8 checkpoints × 2 models; each thin line is one checkpoint, thick = model mean)", fontsize=10, loc="left", color=C["ink"])
    fig.tight_layout(); return fig, rows

def fig_main03(mode):
    S.apply_theme(mode); C = S.CHROME[mode]; fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4), sharex=True, sharey=True); rows = []
    for ax, (name, m, r) in zip(axs, (("Tier-0 (c02, c03, c04)", t0, t0rep), ("Meta-World", mw, mwrep))):
        j = pd.concat([m.set_index(["ckpt_id", "arm"]).d_sr_id.rename("a"), r.set_index(["ckpt_id", "arm"]).d_sr_id.rename("b")], axis=1).dropna().reset_index()
        ax.fill_between([-0.5, 1], [-0.6, 0.9], [-0.4, 1.1], color=C["grid"], alpha=0.6, lw=0); ax.plot([-0.5, 1], [-0.5, 1], color=C["axis"], lw=0.9)
        for a in ("sonnet55-medium", "opus55-medium"):
            q = j[j.arm == a]; ax.scatter(q.a, q.b, s=34, color=col(a, mode), edgecolors=C["bg"] if "bg" in C else "white", linewidths=0.6, label=label(a), zorder=3)
            for _, x in q.iterrows(): rows.append([name, x.ckpt_id, a, x.a, x.b])
        s = np.sqrt(((j.a - j.b) ** 2).mean() / 2); ax.set_title(f"{name}: run-to-run spread σ = {s:.3f}", fontsize=9.5, loc="left", color=C["ink"]); ax.set_xlabel("gain, first run"); ax.grid(lw=0.4, alpha=0.5); ax.set_xlim(-0.2, 0.85); ax.set_ylim(-0.2, 0.85)
    axs[0].set_ylabel("gain, independent second run"); axs[0].legend(fontsize=8, frameon=False, loc="upper left"); fig.suptitle("Same checkpoint, same prompt, two independent runs (line = identical, band = ±0.10)", x=0.01, ha="left", fontsize=10.5, color=C["ink"]); fig.tight_layout(rect=(0, 0, 1, 0.94)); return fig, rows

def fig_main04(mode):
    S.apply_theme(mode); C = S.CHROME[mode]; fig = plt.figure(figsize=(12.5, 6.2)); rows = []
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 2.4, 0.16], left=0.14, right=0.93, top=0.88, bottom=0.17, wspace=0.62); axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]; cax = fig.add_subplot(gs[2])
    short = lambda i: i.replace("-avoid", "").replace("-hold", "").replace("-delay", "").replace("mw-", "").replace("-weak_axis", "-weak").replace("-bias_shift", "-bias").replace("/seed", " seed")
    for ax, (name, df, arms) in zip(axs, (("Tier-0", t0, ["sonnet55-medium", "sonnet55-high", "opus55-medium", "opus55-high"]), ("Meta-World", mw, ["sonnet55-medium", "opus55-medium"]))):
        ids = sorted(set(df.ckpt_id)); p = per(df, ids).pivot_table(index="ckpt_id", columns="arm", values="d_sr_id"); g = per(base[base.arm == "gain_greedy"], ids).set_index("ckpt_id")["d_sr_id"]
        M = pd.DataFrame({a: p[a] - g.reindex(p.index) for a in arms}).loc[ids]; im = ax.imshow(M.to_numpy(), cmap=S.diverging_cmap(mode), vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([label(a) for a in arms], rotation=25, ha="right", fontsize=8.5); ax.set_yticks(range(len(ids))); ax.set_yticklabels([short(i) for i in ids], fontsize=7.5); ax.grid(False)
        for (i, jx), v in np.ndenumerate(M.to_numpy()): ax.text(jx, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.8, color=C["ink"])
        ax.set_title(name, fontsize=9.5, loc="left", color=C["ink"]); [rows.append([name, ids[i], arms[jx], v]) for (i, jx), v in np.ndenumerate(M.to_numpy())]
    cb = fig.colorbar(im, cax=cax); cb.set_label("editor gain − output-gain-search gain (blue: editor better)", fontsize=8); cax.tick_params(labelsize=7.5)
    fig.suptitle("Where the editors beat or lose to a one-knob search, checkpoint by checkpoint", x=0.01, ha="left", fontsize=10.5, color=C["ink"]); return fig, rows

def fig_main05(mode):
    from wsarena import tasks, mlp
    S.apply_theme(mode); C = S.CHROME[mode]; fig, ax = plt.subplots(figsize=(7.4, 4.4)); rows = []
    for w in (24, 64, 128, 256, 512):
        a = tasks.get_task(f"c02-avoid@w{w}").arch(); wt = mlp.init_weights(a[0], a[1], a[2], np.random.default_rng(0)); tok = len(mlp.format_weights_text(wt)) / 2.6; rows.append([w, mlp.n_params(wt), tok])
    d = pd.DataFrame(rows, columns=["width", "params", "tokens"]); k = S.color("Opus 5.5", mode)
    ax.plot(d.width, d.tokens, color=k, lw=2, marker="o"); ax.set_xscale("log", base=2); ax.set_yscale("log")
    for lim, t in ((200_000, "200k-token context"), (1_000_000, "1M-token context")): ax.axhline(lim, color=C["muted"], lw=1, ls="--"); ax.text(24, lim * 1.08, t, fontsize=8, color=C["ink2"])
    for _, r in d.iterrows(): ax.annotate(f"{int(r.params):,} params\n≈{r.tokens/1000:.0f}k tokens", (r.width, r.tokens), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, color=C["ink2"])
    ax.axvspan(17, 140, color=C["grid"], alpha=0.5, lw=0); ax.text(19, 400, "Native track:\nweights in the prompt", fontsize=8.5, color=C["ink"]); ax.text(200, 400, "Probe track:\nread-only probes", fontsize=8.5, color=C["ink"])
    ax.set_xlim(17, 700); ax.set_xticks(d.width); ax.set_xticklabels([str(w) for w in d.width]); ax.set_xlabel("hidden width of the c02 policy"); ax.set_ylabel("tokens needed to print the weights"); ax.set_ylim(300, 3e6); ax.grid(lw=0.4, alpha=0.5)
    ax.set_title("Why a probe track: the weights stop fitting in the prompt at width ≈ 256", fontsize=10, loc="left", color=C["ink"]); fig.tight_layout(); return fig, rows

t0t = t0
FIGS = {"MAIN-01": fig_main01, "MAIN-02": fig_main02, "MAIN-03": fig_main03, "MAIN-04": fig_main04, "MAIN-05": fig_main05}
for fid, fn in FIGS.items():
    for mode in ("light", "dark"):
        fig, rows = fn(mode); S.provenance(fig, fid, "runs/main-v1-*, baselines-*; scripts/make_main_figures.py", mode); S.save_all(fig, fid, OUT, mode)
    pd.DataFrame(rows).to_csv(OUT / "data" / f"{fid}.csv", index=False); print("wrote", fid, len(rows), "rows", flush=True)
