"""LLM-era figure templates (rendered only when real LLM runs exist; otherwise 'pending data', never a placeholder chart).

Entities and colours are fixed in style.ENTITY_SLOT: model = hue (Sonnet 5.5 blue, Opus 5.5 orange); effort = fill (high = solid,
medium = hollow ring). Scripted controls appear only as neutral reference marks/lines so that no hue changes meaning.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D
from . import style as S
from .figures import PendingData, figure, _canvas  # noqa: F401
from .style import CHROME, CLASS_LABEL, color

ROOT = S.ROOT
MODEL_LABEL = {"claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5"}
ARMS = [("Sonnet 5.5", "medium"), ("Sonnet 5.5", "high"), ("Opus 5.5", "medium"), ("Opus 5.5", "high")]
PATCH_STATES = ["valid_changed", "valid_noop", "empty_block", "no_block", "all_skipped"]


def _llm_runs() -> pd.DataFrame:
    """All completed non-scripted runs under runs/<exp>/<attempt>/ (baseline experiments excluded)."""
    parts = []
    for exp in sorted((ROOT / "runs").glob("*")):
        if not exp.is_dir() or exp.name in ("SUMMARY",) or exp.name.startswith(("baselines", "smoke")):
            continue
        for att in sorted(exp.glob("*")):
            if att.is_dir() and list(att.glob("*/run.json")):
                d = D.load_runs(att)
                if len(d): parts.append(d)
    if not parts: raise PendingData("no LLM runs under runs/ yet")
    d = pd.concat(parts, ignore_index=True)
    d = d[d.provider != "scripted"] if "provider" in d else d.iloc[0:0]
    d = d[d.status.isin(["verified", "failed"])]
    if d.empty: raise PendingData("no completed LLM runs yet")
    d["entity"] = d.model.map(MODEL_LABEL).fillna(d.model)
    return d


def _note(d) -> str:
    return f"{len(d)} LLM runs from " + ", ".join(sorted(set(d.exp_id)))


def _ref_lines():
    try:
        b = D.ckpt_level(D.load_baselines())
    except Exception:
        return {}
    return {k: b[(b.arm == "analytic_ridge") & (b.ckpt_class == k)].d_sr_id.mean() for k in ("S1", "S2", "S3")}


def _arm_style(ent, eff, mode):
    c = CHROME[mode]; col = color(ent, mode)
    return dict(color=col, mfc=col if eff == "high" else c["surface"], mec=col)


def _arm_key(ent, eff): return f"{ent} · {eff}"


def _legend(ax, mode):
    c = CHROME[mode]; h = []
    for ent, eff in ARMS:
        st = _arm_style(ent, eff, mode); h.append(plt.Line2D([], [], marker="o", ls="", ms=6, mfc=st["mfc"], mec=st["mec"], mew=1.4, label=_arm_key(ent, eff)))
    return h


@figure("LLM-01", "Claude arms: hidden ID gain by checkpoint class",
        "【待数据】Claude 四个臂（Sonnet/Opus × medium/high）在隐藏 ID 上的 ΔSR（点=检查点，粗线=聚类 bootstrap 95% CI）；虚线=闭式解（特权）均值，灰带=判定带宽。",
        "llm")
def llm01(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = _llm_runs(); ref = _ref_lines()
    ck = d.groupby(["ckpt_class", "ckpt", "entity", "effort"], as_index=False).d_sr_id.mean()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True); rng = np.random.default_rng(5); rows = []
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        ax.axhspan(-0.10, 0.10, color=c["grid"], alpha=0.9, lw=0, zorder=0); ax.axhline(0, color=c["axis"], lw=0.9)
        if k in ref and ref[k] == ref[k]: ax.axhline(ref[k], color=color("analytic_ridge", mode), lw=1.2, ls=(0, (4, 3)), zorder=1)
        for i, (ent, eff) in enumerate(ARMS):
            q = ck[(ck.ckpt_class == k) & (ck.entity == ent) & (ck.effort == eff)].d_sr_id
            if q.empty: continue
            st = _arm_style(ent, eff, mode)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(q)), q, s=11, color=st["color"], alpha=0.45, lw=0)
            m, lo, hi = D.boot_ci(q.values, seed=i); ax.plot([i, i], [lo, hi], color=st["color"], lw=2.2, solid_capstyle="round")
            ax.scatter([i], [m], s=52, facecolor=st["mfc"], edgecolor=st["mec"], lw=1.6, zorder=5)
            ax.annotate(f"{m:+.2f}", (i, hi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, color=c["ink2"])
            rows.append({"class": k, "arm": _arm_key(ent, eff), "n_ckpt": len(q), "mean": m, "ci_lo": lo, "ci_hi": hi})
        ax.set_xticks(range(4)); ax.set_xticklabels(["Sonnet\nmedium", "Sonnet\nhigh", "Opus\nmedium", "Opus\nhigh"], fontsize=6.8); ax.set_title(CLASS_LABEL[k], fontsize=8.6)
        ax.set_xlim(-0.6, 3.6)
    axs[0].set_ylabel("$\\Delta$ hidden success rate (ID)", fontsize=8); axs[0].set_ylim(-0.85, 0.85)
    S.footnote(fig, "dashed = closed-form refit* mean (privileged reference) · gray band = decision band ±0.10 · solid = high effort, ring = medium", mode)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.3, wspace=0.06)
    S.provenance(fig, "LLM-01", _note(d), mode)
    return fig, pd.DataFrame(rows), "LLM runs"


@figure("LLM-02", "Decomposing the gain: valid patch × improvement × size",
        "【待数据】把 ΔSR 分解为 P(提交合法补丁)、P(改善 | 合法)、E[ΔSR | 合法]，区分“没写出补丁”“补丁无用”“补丁有用”；推理强度是否单调看这三项而不是只看总分。",
        "llm")
def llm02(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = _llm_runs(); fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=False); rows = []
    for ax, (key, ttl) in zip(axs, [("p_valid", "P(valid patch)"), ("p_improve", "P(improve | valid)"), ("e_gain", "E[ΔSR | valid]")]):
        for i, (ent, eff) in enumerate(ARMS):
            q = d[(d.entity == ent) & (d.effort == eff)]
            v = q[q.valid_patch_rate > 0]
            val = {"p_valid": q.valid_patch_rate.values, "p_improve": (v.d_sr_id > 0).astype(float).values, "e_gain": v.d_sr_id.values}[key]
            if len(val) == 0: continue
            st = _arm_style(ent, eff, mode); m, lo, hi = D.boot_ci(val, seed=i)
            ax.plot([m, m], [i - 0.0, i], color=st["color"]); ax.plot([lo, hi], [i, i], color=st["color"], lw=2.2, solid_capstyle="round")
            ax.scatter([m], [i], s=52, facecolor=st["mfc"], edgecolor=st["mec"], lw=1.6, zorder=5)
            ax.annotate(f"{m:.2f}", (hi, i), xytext=(5, 0), textcoords="offset points", fontsize=7, va="center", color=c["ink2"])
            rows.append({"metric": ttl, "arm": _arm_key(ent, eff), "n": len(val), "mean": m, "ci_lo": lo, "ci_hi": hi})
        ax.set_yticks(range(4)); ax.set_yticklabels([_arm_key(*a) for a in ARMS] if key == "p_valid" else [], fontsize=7); ax.invert_yaxis(); ax.set_title(ttl, fontsize=8.4)
        if key != "e_gain": ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.88, bottom=0.2, wspace=0.12); S.provenance(fig, "LLM-02", _note(d), mode)
    return fig, pd.DataFrame(rows), "LLM runs"


@figure("LLM-03", "Cost versus gain",
        "【待数据】每个 run 的输出 token（对数）与 ΔSR：同一增益下谁更省；高推理强度是否把预算花在了思考而不是补丁上。",
        "llm")
def llm03(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = _llm_runs(); d = d[d.tok_out.notna() & (d.tok_out > 0)]
    if d.empty: raise PendingData("no token accounting in LLM runs")
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax.axhspan(-0.10, 0.10, color=c["grid"], alpha=0.9, lw=0, zorder=0); ax.axhline(0, color=c["axis"], lw=0.9); rows = []
    for ent, eff in ARMS:
        q = d[(d.entity == ent) & (d.effort == eff)]
        if q.empty: continue
        st = _arm_style(ent, eff, mode)
        ax.scatter(q.tok_out, q.d_sr_id, s=22, facecolor=st["mfc"], edgecolor=st["mec"], lw=1.2, alpha=0.8)
        rows += [{"arm": _arm_key(ent, eff), "tok_out": a, "d_sr_id": b} for a, b in zip(q.tok_out, q.d_sr_id)]
    ax.set_xscale("log"); ax.set_xlabel("output tokens per run (log)"); ax.set_ylabel("$\\Delta$ hidden success rate (ID)")
    ax.legend(handles=_legend(ax, mode), loc="upper left", fontsize=7, ncol=2)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.2); S.provenance(fig, "LLM-03", _note(d) + "; provider-reported output tokens", mode)
    return fig, pd.DataFrame(rows), "LLM runs"


@figure("LLM-04", "Gain across editing rounds",
        "【待数据】开发集成功率随编辑轮次的变化（0=未编辑的初始报告；均值与 95% CI）：对应 VABAR 的“零样本 vs 两轮反馈后”，这里给出每一轮的曲线。",
        "llm")
def llm04(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = _llm_runs(); recs = []
    for r in d.itertuples():
        j = json.loads(Path(r.path).read_text())
        sr = [j["initial_report"]["success_rate"]] + [x["dev_report"]["success_rate"] for x in j["rounds"]]
        for t, v in enumerate(sr): recs.append({"entity": r.entity, "effort": r.effort, "ckpt": r.ckpt, "round": t, "dev_sr": v})
    t = pd.DataFrame(recs); fig, ax = plt.subplots(figsize=(6.6, 3.1)); rows = []; ends = []
    for i, (ent, eff) in enumerate(ARMS):
        q = t[(t.entity == ent) & (t.effort == eff)]
        if q.empty: continue
        st = _arm_style(ent, eff, mode); g = q.groupby("round").dev_sr; m = g.mean(); x = m.index.values
        ci = np.array([D.boot_ci(g.get_group(k).values, n=1000, seed=i)[1:] for k in x])
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=st["color"], alpha=0.14, lw=0); ax.plot(x, m.values, color=st["color"], lw=1.8, marker="o", ms=4, mfc=st["mfc"], mec=st["mec"])
        ends.append((m.values[-1], _arm_key(ent, eff)))
        rows += [{"arm": _arm_key(ent, eff), "round": int(k), "mean_dev_sr": float(v)} for k, v in m.items()]
    last = None
    for y, lab in sorted(ends):                      # nudge end labels apart so none overlap
        yy = y if last is None or y - last >= 0.05 else last + 0.05; last = yy
        ax.annotate(lab, (t["round"].max(), yy), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center", color=c["ink2"])
    ax.set_xticks(sorted(t["round"].unique()))
    ax.set_xlabel("editing round (0 = unedited)"); ax.set_ylabel("development success rate"); ax.set_ylim(-0.02, 1.02)
    fig.subplots_adjust(left=0.1, right=0.8, top=0.95, bottom=0.2); S.provenance(fig, "LLM-04", _note(d), mode)
    return fig, pd.DataFrame(rows), "LLM runs"


@figure("LLM-05", "Where editing fails: patch status per round",
        "【待数据】每个臂各轮次的补丁状态构成（合法且改变权重 / 合法但未改 / 空块 / 无块 / 全部被跳过）：失败留在分母里，并按“最早的解释层”归类。",
        "llm")
def llm05(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = _llm_runs(); recs = []
    for r in d.itertuples():
        for x in json.loads(Path(r.path).read_text())["rounds"]:
            st = x["patch_status"] if x["patch_status"] in PATCH_STATES else "turn_failed"
            recs.append({"arm": _arm_key(r.entity, r.effort), "status": st})
    t = pd.DataFrame(recs); states = PATCH_STATES + ["turn_failed"]
    cols = {"valid_changed": S.STATUS["good"], "valid_noop": c["muted"], "empty_block": c["axis"], "no_block": S.STATUS["warning"], "all_skipped": S.STATUS["serious"], "turn_failed": S.STATUS["critical"]}
    fig, ax = plt.subplots(figsize=(7.0, 2.6)); rows = []
    for i, (ent, eff) in enumerate(ARMS):
        q = t[t.arm == _arm_key(ent, eff)]
        if q.empty: continue
        left = 0.0
        for s_ in states:
            v = (q.status == s_).mean()
            if v <= 0: continue
            ax.barh(i, v, left=left, color=cols[s_], edgecolor=c["surface"], lw=1.5, height=0.6)
            if v > 0.07: ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=6.8, color="#0b0b0b" if s_ in ("no_block", "empty_block") else "#ffffff")
            rows.append({"arm": _arm_key(ent, eff), "status": s_, "share": v}); left += v
    ax.set_yticks(range(4)); ax.set_yticklabels([_arm_key(*a) for a in ARMS], fontsize=7.4); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=cols[s_], label=s_.replace("_", " ")) for s_ in states], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=6, fontsize=6.6, handlelength=1.2)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.32); S.provenance(fig, "LLM-05", _note(d), mode)
    return fig, pd.DataFrame(rows), "LLM runs"
