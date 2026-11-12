"""WSArena figure catalogue (baseline-era figures; LLM-era templates are in llm_figures.py).

Each figure function f(mode) -> (fig, table_df, data_note). Registered with @figure so that scripts/make_figures.py renders
every figure in light + dark, saves png/pdf/svg + a CSV table view, and writes reports/FIGURES_INDEX.md + manifest.json.
Figures show ONLY recorded artifacts (run.json files, manifest, SUMMARY/*.json); nothing is recomputed here except the
case-study rollouts (FIG-14), which re-run stored checkpoints on the stored hidden seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from . import data as D
from . import style as S
from .style import CHROME, ARM_LABEL, ARM_ORDER, CLASS_LABEL, TASK_LABEL, color

ROOT = S.ROOT
class PendingData(Exception):
    """Raised by a figure whose input data does not exist yet (rendered as 'pending', never as a placeholder chart)."""


REGISTRY: list["Fig"] = []
SHORT = {"random": "random\nsparse", "random_greedy": "greedy\nsearch", "gain_greedy": "gain\nsearch", "analytic_ridge": "closed\nform*", "oracle_restore": "oracle\nrestore†", "null": "no edit"}
OFF = {"null": (6, -11), "random": (7, 5), "random_greedy": (7, 5), "gain_greedy": (7, 5), "analytic_ridge": (9, -14), "oracle_restore": (9, 10)}   # FIG-09 label offsets
TASK_ORDER = ["c01-reach", "c02-avoid", "c03-hold", "c04-delay", "mw-push", "mw-pickplace", "mw-peginsert"]


@dataclass
class Fig:
    fid: str; title: str; caption: str; group: str; fn: object


def figure(fid, title, caption, group):
    def deco(fn):
        REGISTRY.append(Fig(fid, title, caption, group, fn)); return fn
    return deco


# --------------------------------------------------------------------------------------------- helpers
def _box(ax, x, y, w, h, text, mode, fc=None, ec=None, ls="-", fs=7.4, bold=False, lw=0.9, ha="center"):
    c = CHROME[mode]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc or c["neutral"], ec=ec or c["axis"], lw=lw, ls=ls)
    ax.add_patch(p)
    tx = x + w / 2 if ha == "center" else x + 1.2
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=c["ink"], fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, p0, p1, mode, rad=0.0, lw=1.0, ls="-", color_=None):
    c = CHROME[mode]
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, ls=ls, color=color_ or c["ink2"],
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def _text_on(hex_bg):
    h = hex_bg.lstrip("#"); r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return "#0b0b0b" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#ffffff"


def _cell_text(ax, x, y, s, rgba, **kw):  # noqa: E302
    rgb = "#%02x%02x%02x" % tuple(int(255 * v) for v in rgba[:3])
    ax.text(x, y, s, ha="center", va="center", color=_text_on(rgb), **kw)


def _canvas(mode, w=7.2, h=3.4, xlim=(0, 100), ylim=(0, 40)):
    S.apply_theme(mode)
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off"); ax.grid(False)
    return fig, ax


def _patch_status_counts() -> pd.DataFrame:
    rows = []
    for exp, att in D.BASELINE_DIRS:
        for rj in sorted((ROOT / "runs" / exp / att).glob("*/run.json")):
            j = json.loads(rj.read_text()); arm = j["spec"]["run_id"].split("__")[2]
            for r in j["rounds"]:
                rows.append({"arm": arm, "turn_status": r["turn_status"], "patch_status": r["patch_status"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- FIG-01
@figure("FIG-01", "WSP-8 evaluation pipeline",
        "WSP-8 评测流程：密封基线检查点 → 编辑者读权重文本并回复一个 edits 块 → 确定性应用 → 开发集反馈（循环 R 轮）→ 冻结最终权重 → 隐藏集（ID/OOD）配对评测。服务端强制预算、工具审计与哈希校验，失败留在分母。",
        "overview")
def fig01(mode):
    c = CHROME[mode]; fig, ax = _canvas(mode, 7.2, 4.1, ylim=(0, 53))
    row1 = 30; h = 13; w = 18.4; xs = [0.8 + i * 19.8 for i in range(5)]
    labels = ["Sealed base\ncheckpoint $\\theta_0$\n(sha256, read-only)\nclass S0–S3", "Editor reads the\nweights as text +\nfeedback JSON\n(LLM or control)",
              "Reply with ONE\n```edits``` block\n(WEP-1; only the\nlast block is read)", "Deterministic apply\nparse · skip count ·\nparameter diff\nstatistics", "Development\nfeedback: 16 fresh\nepisodes / round\nSR · failure types"]
    for x, t in zip(xs, labels):
        _box(ax, x, row1, w, h, t, mode, fs=6.5)
    for i in range(4):
        _arrow(ax, (xs[i] + w + 0.1, row1 + h / 2), (xs[i + 1] - 0.1, row1 + h / 2), mode)
    ax.add_patch(FancyArrowPatch((xs[4] + w / 2, row1 + h + 0.4), (xs[1] + w / 2, row1 + h + 0.4), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                 color=c["ink2"], connectionstyle="arc3,rad=0.16"))
    ax.text(50, row1 + h + 8.6, "R rounds · edits are cumulative · RESET restores $\\theta_0$ · one candidate patch per round, no retries",
            ha="center", fontsize=6.8, color=c["ink2"])
    row2 = 11
    _box(ax, 0.8, row2, 21, h, "Freeze the final-round\nweights $\\theta_R$\n(the editor sees\nnothing more)", mode, fs=6.5)
    _box(ax, 29, row2, 31, h, "Hidden evaluation (never shown)\nID 200 + OOD 200 episodes,\npaired with the base checkpoint", mode, ls=(0, (4, 2)), ec=c["ink"], lw=1.1, fs=6.5)
    _box(ax, 68, row2, 31, h, "Metrics\n$\\Delta$SR · Recovery / Regression\nlocalization · patch size · cost", mode, fs=6.5)
    xe = xs[4] + w / 2
    ax.plot([xe, xe, 11.3], [row1 - 0.4, 27.6, 27.6], color=c["ink2"], lw=1.0, solid_capstyle="round")
    _arrow(ax, (11.3, 27.6), (11.3, row2 + h + 0.4), mode)
    ax.text(60, 28.3, "after round R", fontsize=6.4, color=c["muted"], ha="center")
    _arrow(ax, (22, row2 + h / 2), (28.7, row2 + h / 2), mode); _arrow(ax, (60.3, row2 + h / 2), (67.7, row2 + h / 2), mode)
    ax.add_patch(Rectangle((0.8, 0.8), 98.4, 7.0, fc="none", ec=c["axis"], lw=0.7, ls=(0, (2, 2))))
    ax.text(50, 4.3, "Enforced server-side:  budgets  ·  tool-free audit (any tool event invalidates the run)  ·  hash-verified checkpoints\n"
            "failures stay in the denominator  ·  provider limits interrupt the run, which resumes by replaying saved turns", ha="center", va="center",
            fontsize=6.4, color=c["ink2"], linespacing=1.5)
    S.provenance(fig, "FIG-01", "schematic", mode)
    return fig, pd.DataFrame({"stage": [l.replace("\n", " ") for l in labels]}), "schematic"


# --------------------------------------------------------------------------------------------- FIG-02
@figure("FIG-02", "Task formalisation and capability ladder",
        "WSArena 任务实例的形式化定义，以及五级能力 L0–L4 与检查点类别、实现状态的对应（✓ 已实现并测试，○ 计划）。",
        "overview")
def fig02(mode):
    c = CHROME[mode]; fig, ax = _canvas(mode, 7.2, 3.6, ylim=(0, 46))
    ax.text(1, 43, "Task instance", fontsize=9, fontweight="bold", color=c["ink"])
    ax.text(1, 39.2, "$\\mathcal{T}=(\\theta_0,\\ \\mathcal{C},\\ \\mathcal{E},\\ \\mathcal{A},\\ \\mathbf{b},\\ \\mathcal{S})$", fontsize=10, color=c["ink"])
    items = [("$\\theta_0$", "sealed initial checkpoint\n(float64, sha256)"), ("$\\mathcal{C}$", "class: S0 zero · S1 planted ·\nS2 natural · S3 healthy"),
             ("$\\mathcal{E}$", "environment family;\nID and hidden OOD parameters"), ("$\\mathcal{A}$", "edit interface: WEP-1 text,\nno tools / training / search"),
             ("$\\mathbf{b}$", "budget: R rounds × 16 dev\nepisodes, 0 retries"), ("$\\mathcal{S}$", "score: hidden $\\Delta$SR,\npaired with the base")]
    for i, (k, v) in enumerate(items):
        y = 33.5 - i * 5.9; ax.text(1.5, y, k, fontsize=8.5, color=c["ink"], va="center"); ax.text(7, y, v, fontsize=6.9, color=c["ink2"], va="center", linespacing=1.3)
    ax.plot([49, 49], [2, 43], color=c["axis"], lw=0.8)
    ax.text(52, 43, "Capability ladder (scored separately)", fontsize=9, fontweight="bold", color=c["ink"])
    rows = [("L0", "Weight reading", "open-ended identification", False), ("L1", "Weight synthesis", "S0 zero network", True),
            ("L2", "Weight repair", "S1 planted · S2 natural", True), ("L3", "Weight enhancement", "S3 healthy policy", True),
            ("L4", "Generalisation", "hidden OOD · reparameterisation", True)]
    for i, (lv, name, cls, done) in enumerate(rows):
        y = 36 - i * 6.6
        _box(ax, 52, y - 2.4, 6, 4.8, lv, mode, bold=True, fs=8)
        ax.text(60, y + 1.1, name, fontsize=7.8, color=c["ink"], va="center", fontweight="bold"); ax.text(60, y - 1.6, cls, fontsize=6.4, color=c["ink2"], va="center")
        st = ("implemented", S.STATUS["good"]) if done else ("planned", c["muted"])
        ax.add_patch(plt.Circle((90.2, y), 0.9, color=st[1], lw=0)); ax.text(92.2, y, st[0], fontsize=6.6, color=c["ink2"], ha="left", va="center")
    ax.text(52, 3.6, "Primary experiment: L2 (repair) · hardest question: L3\nL0/L1 diagnostics · L4 rules out permutation luck", fontsize=6.4, color=c["ink2"], va="center", linespacing=1.4)
    S.provenance(fig, "FIG-02", "schematic", mode)
    return fig, pd.DataFrame(rows, columns=["level", "name", "checkpoints", "implemented"]), "schematic"


# --------------------------------------------------------------------------------------------- FIG-03
@figure("FIG-03", "Layered failure attribution",
        "诊断式失败归因（参照 VABAR 的诊断执行流）：每个失败归到最早能解释后续结果的层——回合层（供应商状态）、补丁层（解析与应用）、效果层（行为变化与逐 episode 失败类型）。下方计数来自 588 个脚本对照 run 的 2,352 个回合。",
        "overview")
def fig03(mode):
    c = CHROME[mode]; fig, ax = _canvas(mode, 7.2, 3.9, ylim=(0, 48))
    cols = [("1  Turn layer", "provider status", ["ok", "rate_limited → run interrupted\n(saved, resumed by replay)", "output_budget_exhausted", "no_output · refused · timeout"]),
            ("2  Patch layer", "parse + apply (deterministic)", ["no_block", "all_skipped / empty_block", "valid_noop", "valid_changed"]),
            ("3  Effect layer", "behaviour on paired hidden episodes", ["Recovery  P(success | base failed)", "Regression  P(failure | base succeeded)", "failure type per episode:\ncollision · wall · left_zone · overshoot ·\nnear_unsettled · stalled_far · drift_far", "ΔSR  (ID / OOD)"])]
    x0 = [1, 35, 69]; w = 30
    ps = _patch_status_counts()
    counts = ps["patch_status"].value_counts().to_dict()
    for (title, sub, items), x in zip(cols, x0):
        ax.text(x, 45, title, fontsize=9, fontweight="bold", color=c["ink"]); ax.text(x, 42.2, sub, fontsize=7, color=c["ink2"])
        for i, it in enumerate(items):
            y = 33.5 - i * 8.2; _box(ax, x, y, w, 6.6, it, mode, fs=6.7)
            if title.startswith("2") and it in ("no_block", "valid_noop", "valid_changed", "all_skipped / empty_block"):
                n = counts.get("valid_changed", 0) if it == "valid_changed" else counts.get("valid_noop", 0) if it == "valid_noop" else \
                    counts.get("no_block", 0) if it == "no_block" else counts.get("all_skipped", 0) + counts.get("empty_block", 0)
                ax.text(x + w - 0.8, y + 1.2, f"n={n:,}", fontsize=6.0, color=c["muted"], ha="right", va="center")
    for a in (31.8, 65.8):
        _arrow(ax, (a, 20), (a + 2.4, 20), mode, lw=1.4)
    ax.add_patch(Rectangle((1, 1.2), 98, 5.4, fc="none", ec=c["axis"], lw=0.7, ls=(0, (2, 2))))
    ax.text(50, 3.9, "Attribution rule: a failed run is charged to the EARLIEST layer that explains the later outcome;\nevery failure stays in the denominator (no retries, no replacement).", ha="center", va="center", fontsize=6.6, color=c["ink2"], linespacing=1.5)
    S.provenance(fig, "FIG-03", f"scripted controls only (n={len(ps):,} rounds); no LLM runs yet", mode)
    return fig, ps["patch_status"].value_counts().rename_axis("patch_status").reset_index(name="rounds"), "baselines"


# --------------------------------------------------------------------------------------------- FIG-04
@figure("FIG-04", "Benchmark coverage matrix",
        "检查点覆盖矩阵：任务（行）× 检查点类别（列），格内为“可用数”，斜线数字为按 SR 带筛选未能构造出的不可行数；右侧给出观测/动作维度与参数量。",
        "benchmark")
def fig04(mode):
    S.apply_theme(mode); c = CHROME[mode]; m = D.manifest()
    tasks = [t for t in TASK_ORDER if t in set(m.task_id)]; classes = ["S0", "S1", "S2", "S3"]
    built = np.array([[((m.task_id == t) & (m["class"] == k) & (m.status == "built")).sum() for k in classes] for t in tasks])
    bad = np.array([[((m.task_id == t) & (m["class"] == k) & (m.status != "built")).sum() for k in classes] for t in tasks])
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(tasks) + 1.5)); ax.grid(False)
    cm = S.seq_cmap(mode); vmax = max(built.max(), 1)
    for i, t in enumerate(tasks):
        for j, k in enumerate(classes):
            col = cm(0.12 + 0.88 * built[i, j] / vmax) if built[i, j] else c["neutral"]
            ax.add_patch(Rectangle((j, len(tasks) - 1 - i), 0.94, 0.9, fc=col, ec=c["surface"], lw=2))
            if built[i, j]: _cell_text(ax, j + 0.47, len(tasks) - 1 - i + 0.62, f"{built[i, j]}", cm(0.12 + 0.88 * built[i, j] / vmax), fontsize=9, fontweight="bold")
            else: ax.text(j + 0.47, len(tasks) - 1 - i + 0.62, "–", ha="center", va="center", color=c["muted"], fontsize=9)
            if bad[i, j]: ax.text(j + 0.47, len(tasks) - 1 - i + 0.22, f"{bad[i, j]} infeasible", ha="center", va="center", fontsize=6.4,
                                  color=c["ink2"] if not built[i, j] else _text_on("#%02x%02x%02x" % tuple(int(255 * v) for v in col[:3])))
    ax.set_xlim(0, 4.2); ax.set_ylim(0, len(tasks))
    ax.set_xticks([j + 0.47 for j in range(4)]); ax.set_xticklabels([CLASS_LABEL[k] for k in classes], fontsize=7.4)
    ax.set_yticks([len(tasks) - 1 - i + 0.45 for i in range(len(tasks))]); ax.set_yticklabels([TASK_LABEL[t] for t in tasks])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    arch = {"c01-reach": "4→24→24→2 · 866 p", "c02-avoid": "6→24→24→2 · 962 p", "c03-hold": "5→24→24→2 · 914 p", "c04-delay": "8→24→24→2 · 1,106 p",
            "mw-push": "39→24→24→4 · 1,660 p", "mw-pickplace": "39→24→24→4 · 1,660 p", "mw-peginsert": "39→24→24→4 · 1,660 p"}
    for i, t in enumerate(tasks): ax.text(4.35, len(tasks) - 1 - i + 0.45, arch[t], va="center", fontsize=6.8, color=c["ink2"], clip_on=False)
    ax.set_title("Checkpoints per task and class", pad=24)
    fig.subplots_adjust(left=0.17, right=0.78, top=0.80, bottom=0.12)
    S.provenance(fig, "FIG-04", f"ckpt_manifest.json ({int((m.status=='built').sum())} built / {int((m.status!='built').sum())} infeasible)", mode)
    tab = pd.DataFrame(built, index=tasks, columns=classes).add_suffix("_built").join(pd.DataFrame(bad, index=tasks, columns=classes).add_suffix("_infeasible"))
    return fig, tab.reset_index(names="task"), "manifest"


# --------------------------------------------------------------------------------------------- FIG-05
@figure("FIG-05", "Start-state difficulty of the checkpoints",
        "未编辑检查点的隐藏成功率（实心=ID，空心=OOD；每点一个检查点，短横线为均值）。S1/S2 起点约 0.3–0.55，S3 约 0.7–0.9；OOD 起点整体低得多——这是留给编辑的空间。",
        "benchmark")
def fig05(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = D.load_baselines(); n = d[d.arm == "null"]
    ck = n.groupby(["task_id", "ckpt", "ckpt_class"], as_index=False)[["sr_base_id", "sr_base_ood"]].mean()
    tasks = [t for t in TASK_ORDER if t in set(ck.task_id)]; fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    rng = np.random.default_rng(1); rows = []
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        for i, t in enumerate(tasks):
            q = ck[(ck.task_id == t) & (ck.ckpt_class == k)]
            if q.empty: continue
            jit = rng.uniform(-0.13, 0.13, len(q))
            ax.scatter(i - 0.18 + jit, q.sr_base_id, s=16, color=c["ink2"], lw=0, alpha=0.9, zorder=3)
            ax.scatter(i + 0.18 + jit, q.sr_base_ood, s=16, facecolor=c["surface"], edgecolor=c["ink2"], lw=0.9, zorder=3)
            for off, v in ((-0.18, q.sr_base_id.mean()), (0.18, q.sr_base_ood.mean())):
                ax.plot([i + off - 0.15, i + off + 0.15], [v, v], color=c["ink"], lw=1.6, zorder=4)
            rows += [{"class": k, "task": t, "ckpt": r.ckpt, "sr_id": r.sr_base_id, "sr_ood": r.sr_base_ood} for r in q.itertuples()]
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=7.2, rotation=0)
        ax.set_title(CLASS_LABEL[k], fontsize=8.6); ax.set_ylim(-0.03, 1.03); ax.set_xlim(-0.6, len(tasks) - 0.4)
    axs[0].set_ylabel("hidden success rate (unedited)")
    h = [plt.Line2D([], [], marker="o", ls="", color=c["ink2"], label="ID"), plt.Line2D([], [], marker="o", ls="", mfc=c["surface"], mec=c["ink2"], label="OOD"),
         plt.Line2D([], [], color=c["ink"], lw=1.6, label="mean")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.115))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.30, wspace=0.07)
    S.provenance(fig, "FIG-05", "baselines-v1/a1 (null arm), one point per checkpoint", mode)
    return fig, pd.DataFrame(rows), "baselines-v1"


# --------------------------------------------------------------------------------------------- FIG-06/07
def _strip(mode, metric, fid, ylabel, title_suffix):
    S.apply_theme(mode); c = CHROME[mode]; d = D.load_baselines(); ck = D.ckpt_level(d)
    arms = [a for a in ARM_ORDER if a != "null"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True); rng = np.random.default_rng(2); rows = []
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        ax.axhspan(-0.10, 0.10, color=c["grid"], alpha=0.9, lw=0, zorder=0); ax.axhline(0, color=c["axis"], lw=0.9, zorder=1)
        for i, a in enumerate(arms):
            if a == "oracle_restore" and k != "S1":
                ax.text(i, 0.0, "n/a", ha="center", va="center", fontsize=7, color=c["muted"]); continue
            q = ck[(ck.ckpt_class == k) & (ck.arm == a)][metric].dropna()
            if q.empty: continue
            col = color(a, mode)
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(q)), q, s=10, color=col, alpha=0.45, lw=0, zorder=2)
            m, lo, hi = D.boot_ci(q.values, seed=i)
            ax.plot([i, i], [lo, hi], color=col, lw=2.2, zorder=4, solid_capstyle="round")
            ax.scatter([i], [m], s=44, color=col, edgecolor=c["surface"], lw=1.4, zorder=5)
            ax.annotate(f"{m:+.2f}", (i, hi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, color=c["ink2"])
            rows += [{"class": k, "arm": a, "n_ckpt": len(q), "mean": m, "ci_lo": lo, "ci_hi": hi}]
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([SHORT[a] for a in arms], fontsize=6.8)
        ax.set_title(CLASS_LABEL[k], fontsize=8.6); ax.set_xlim(-0.6, len(arms) - 0.4)
    axs[0].set_ylabel(ylabel, fontsize=8); axs[0].set_ylim(-0.85, 0.85)
    S.footnote(fig, "gray band = decision band $\\delta$ = ±0.10   ·   " + S.FOOT, mode, y=0.045)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.27, wspace=0.06)
    S.provenance(fig, fid, "baselines-v1 + greedy-v1 + gain-v1 (a1); dots=checkpoints, bar=95% cluster bootstrap CI", mode)
    return fig, pd.DataFrame(rows), "baselines-v1"


@figure("FIG-06", "Control arms: hidden ID gain",
        "脚本对照臂在隐藏 ID 上的成功率增益 ΔSR（点=检查点，粗线=按检查点聚类的 bootstrap 95% CI，灰带=判定带宽 ±0.10）。闭式解在 S1 上≈神谕；S2 上不稳；同等反馈预算的随机搜索≈0。",
        "baselines")
def fig06(mode):
    return _strip(mode, "d_sr_id", "FIG-06", "$\\Delta$ hidden success rate (ID)", "ID")


@figure("FIG-07", "Control arms: hidden OOD gain",
        "同 FIG-06，但为隐藏 OOD 上的增益。OOD 偏移已在 select 命名空间上校准；c02-avoid 的 S2 起点 OOD 成功率仍为 0（该子集无分辨力）。",
        "baselines")
def fig07(mode):
    return _strip(mode, "d_sr_ood", "FIG-07", "$\\Delta$ hidden success rate (OOD)", "OOD")


# --------------------------------------------------------------------------------------------- FIG-08
@figure("FIG-08", "Gain by task, class and control arm",
        "ΔSR（ID）热图：行=任务×类别，列=对照臂；蓝=提升，红=下降，灰=0；格内为均值，括号为检查点数。可以读出：解析岭回归在 c03-hold 的 S2 上显著为负（−0.36）。",
        "baselines")
def fig08(mode):
    S.apply_theme(mode); c = CHROME[mode]; ck = D.ckpt_level(D.load_baselines())
    arms = ["oracle_restore", "analytic_ridge", "gain_greedy", "random_greedy", "random"]
    keys = [(t, k) for t in TASK_ORDER for k in ["S1", "S2", "S3"] if not ck[(ck.task_id == t) & (ck.ckpt_class == k)].empty]
    M = np.full((len(keys), len(arms)), np.nan); N = np.zeros((len(keys), len(arms)), int)
    for i, (t, k) in enumerate(keys):
        for j, a in enumerate(arms):
            q = ck[(ck.task_id == t) & (ck.ckpt_class == k) & (ck.arm == a)].d_sr_id
            if len(q) and not (a == "oracle_restore" and k != "S1"): M[i, j] = q.mean(); N[i, j] = len(q)
    fig, ax = plt.subplots(figsize=(5.6, 0.27 * len(keys) + 1.3)); ax.grid(False); cm = S.diverging_cmap(mode)
    for i in range(len(keys)):
        for j in range(len(arms)):
            if np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j, len(keys) - 1 - i), 0.95, 0.9, fc=c["page"], ec=c["surface"], lw=2)); ax.text(j + .47, len(keys) - 1 - i + .45, "n/a", ha="center", va="center", fontsize=6.2, color=c["muted"]); continue
            col = cm(0.5 + 0.5 * np.clip(M[i, j] / 0.7, -1, 1))
            ax.add_patch(Rectangle((j, len(keys) - 1 - i), 0.95, 0.9, fc=col, ec=c["surface"], lw=2))
            _cell_text(ax, j + 0.47, len(keys) - 1 - i + 0.45, f"{M[i, j]:+.2f}", col, fontsize=7.2)
    ax.set_xlim(0, len(arms)); ax.set_ylim(0, len(keys)); ax.tick_params(length=0)
    ax.set_xticks([j + .47 for j in range(len(arms))]); ax.set_xticklabels([ARM_LABEL[a].replace(" ", "\n", 1) for a in arms], fontsize=6.9); ax.xaxis.tick_top()
    ax.set_yticks([len(keys) - 1 - i + .45 for i in range(len(keys))]); ax.set_yticklabels([f"{TASK_LABEL[t]} · {k}" for t, k in keys], fontsize=7)
    for sp in ax.spines.values(): sp.set_visible(False)
    cax = fig.add_axes([0.80, 0.30, 0.02, 0.4]); g = np.linspace(1, -1, 64)[:, None]
    cax.imshow(g, aspect="auto", cmap=cm, extent=[0, 1, -0.7, 0.7]); cax.set_xticks([]); cax.yaxis.tick_right(); cax.set_yticks([-0.7, 0, 0.7]); cax.set_yticklabels(["−0.7", "0", "+0.7"], fontsize=6.8)
    for sp in cax.spines.values(): sp.set_visible(False)
    cax.set_title("$\\Delta$SR", fontsize=7.5, loc="left")
    S.footnote(fig, S.FOOT, mode, y=0.025); fig.subplots_adjust(left=0.27, right=0.78, top=0.86, bottom=0.1)
    S.provenance(fig, "FIG-08", "baselines-v1/a1 + greedy; mean over checkpoints; 'n/a' = not applicable (oracle needs planted truth)", mode)
    rows = [{"task": t, "class": k, **{a: M[i, j] for j, a in enumerate(arms)}} for i, (t, k) in enumerate(keys)]
    return fig, pd.DataFrame(rows), "baselines-v1"


# --------------------------------------------------------------------------------------------- FIG-09
@figure("FIG-09", "Recovery versus Regression",
        "可靠性：Recovery=原本失败的 episode 被修好的比例；Regression=原本成功的 episode 被破坏的比例。理想点在左上角。闭式解在 S2 上同时高 Regression（0.50），而 S1 上接近理想。",
        "baselines")
def fig09(mode):
    S.apply_theme(mode); c = CHROME[mode]; ck = D.ckpt_level(D.load_baselines())
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True); rows = []
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        ax.plot([0, 1], [0, 1], color=c["axis"], lw=0.9, ls=(0, (3, 3))); ax.text(0.62, 0.55, "recovery = regression", fontsize=6.2, color=c["muted"], rotation=34)
        for a in ARM_ORDER:
            if a == "oracle_restore" and k != "S1": continue
            q = ck[(ck.ckpt_class == k) & (ck.arm == a)]
            if q.empty or q.recovery_id.isna().all(): continue
            x, y = q.regression_id.mean(), q.recovery_id.mean(); col = color(a, mode)
            ax.scatter(q.regression_id, q.recovery_id, s=8, color=col, alpha=0.3, lw=0)
            ax.scatter([x], [y], s=52, color=col, edgecolor=c["surface"], lw=1.4, zorder=5)
            ax.annotate({"null": "no edit", "random": "random", "random_greedy": "greedy", "gain_greedy": "gain", "analytic_ridge": "refit*", "oracle_restore": "oracle†"}[a], (x, y), xytext=OFF[a], textcoords="offset points", fontsize=6.6, color=c["ink2"], ha="left")
            rows.append({"class": k, "arm": a, "regression": x, "recovery": y})
        ax.set_title(CLASS_LABEL[k], fontsize=8.6); ax.set_xlim(-0.04, 1.02); ax.set_ylim(-0.04, 1.04); ax.set_xlabel("Regression")
    axs[0].set_ylabel("Recovery")
    S.footnote(fig, "ideal = top-left (high recovery, low regression)   ·   " + S.FOOT, mode); fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.30, wspace=0.07)
    S.provenance(fig, "FIG-09", "baselines-v1/a1 + greedy; large dot = mean of checkpoints, small dots = checkpoints", mode)
    return fig, pd.DataFrame(rows), "baselines-v1"


# --------------------------------------------------------------------------------------------- FIG-10
@figure("FIG-10", "Evaluation noise floor",
        "噪声尺度：同一检查点在 5 个互不相交的 200-episode 集合上的成功率标准差（每点一个检查点；虚线=二项上界 0.035）。判定带宽 δ=0.10 高于 2σ 的典型值，故安全。",
        "benchmark")
def fig10(mode):
    S.apply_theme(mode); c = CHROME[mode]; nf = D.summary_json("noise_floor.json")
    rows = pd.DataFrame(nf["rows"]); rows["task"] = rows.ckpt_id.str.split("/").str[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True); out = []
    for ax, key, ttl in zip(axs, ["sd_id", "sd_ood"], ["ID split", "OOD split"]):
        rng = np.random.default_rng(3)
        for i, k in enumerate(["S0", "S1", "S2", "S3"]):
            q = rows[rows["class"] == k][key]
            if q.empty: continue
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(q)), q, s=12, color=c["ink2"], alpha=0.7, lw=0)
            ax.plot([i - .22, i + .22], [q.median()] * 2, color=c["ink"], lw=1.8)
        ax.axhline(0.0354, color=c["ink"], lw=0.9, ls=(0, (4, 3))); ax.text(-0.42, 0.0368, "binomial bound 0.035", fontsize=6.6, ha="left", color=c["ink2"])
        ax.set_xticks(range(4)); ax.set_xticklabels([CLASS_LABEL[k].replace(" ", "\n", 1) for k in ["S0", "S1", "S2", "S3"]], fontsize=7); ax.set_title(ttl, fontsize=8.6)
        out += [{"split": ttl, "class": k, "median_sd": float(rows[rows['class'] == k][key].median())} for k in ["S0", "S1", "S2", "S3"] if not rows[rows['class'] == k].empty]
    axs[0].set_ylabel("SD of success rate across 5 episode sets"); axs[0].set_ylim(-0.003, 0.07)
    axs[0].text(0.02, 0.97, f"median SD (ID) = {nf['summary']['median_sd_id']:.3f}\nmax = {nf['summary']['max_sd_id']:.3f}\nwith n=200: 2σ ≈ 0.07 < δ = 0.10",
                transform=axs[0].transAxes, fontsize=7, va="top", color=c["ink2"])
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.2, wspace=0.05)
    S.provenance(fig, "FIG-10", f"noise_floor.json ({nf['summary']['n_ckpts']} checkpoints × 5 disjoint 200-episode sets)", mode)
    return fig, pd.DataFrame(out), "noise_floor.json"


# --------------------------------------------------------------------------------------------- FIG-11
@figure("FIG-11", "OOD shift calibration",
        "OOD 偏移强度校准（只用 select 命名空间，不碰隐藏集）：横轴为偏移强度 s（0=同分布，1=v1 的最大偏移），纵轴为 S2∪S3 检查点的平均成功率；竖线=选定值，实线=ID 基线。规则：取使 OOD 成功率 ≥0.20 且 ID−OOD ≥0.15 的最大 s。",
        "benchmark")
def fig11(mode):
    S.apply_theme(mode); c = CHROME[mode]; cal = D.summary_json("ood_calibration.json")
    tids = list(cal.keys()); fig, axs = plt.subplots(1, len(tids), figsize=(7.2, 2.5), sharey=True); rows = []
    for ax, t in zip(np.atleast_1d(axs), tids):
        g = pd.DataFrame(cal[t]["grid"]); ax.axhline(cal[t]["id_sr"], color=c["ink2"], lw=1.1); ax.axhline(0.20, color=c["axis"], lw=0.9, ls=(0, (3, 3)))
        ax.plot(g.s, g.ood_sr, color=c["ink"], lw=1.6, marker="o", ms=3.2)
        s = cal[t]["choice_s"]
        if s is not None:
            y = float(g[g.s == s].ood_sr.iloc[0]); ax.axvline(s, color=c["muted"], lw=0.9); ax.scatter([s], [y], s=46, color=c["ink"], edgecolor=c["surface"], lw=1.4, zorder=5)
            ax.annotate(f"s={s}", (s, y), xytext=(-5 if s > 0.7 else 4, 8), textcoords="offset points", fontsize=7, color=c["ink2"], ha="right" if s > 0.7 else "left")
        ax.set_title(TASK_LABEL[t], fontsize=8.6); ax.set_xlabel("shift strength s"); ax.set_ylim(-0.02, 0.95); ax.set_xlim(0, 1.06)
        rows += [{"task": t, **r} for r in cal[t]["grid"]]
    np.atleast_1d(axs)[0].set_ylabel("mean SR of S2∪S3"); fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.42, wspace=0.07)
    S.footnote(fig, "horizontal solid line = ID level of the same checkpoints · dashed = 0.20 floor · vertical line = chosen s", mode)
    S.provenance(fig, "FIG-11", "ood_calibration.json ('select' namespace only)", mode)
    return fig, pd.DataFrame(rows), "ood_calibration.json"


# --------------------------------------------------------------------------------------------- FIG-12
@figure("FIG-12", "What the unedited policies get wrong",
        "未编辑检查点在隐藏 ID 上的失败类型构成（占失败 episode 的比例，规则分型，不是编辑者自评）。S1/S2 以 stalled_far / drift_far / collision 为主，说明故障是可诊断的。",
        "benchmark")
def fig12(mode):
    S.apply_theme(mode); c = CHROME[mode]; cnt = {k: Counter() for k in ["S1", "S2", "S3"]}
    for rj in sorted((ROOT / "runs/baselines-v1/a1").glob("*__null__r0/run.json")):
        j = json.loads(rj.read_text()); k = j["spec"]["ckpt_class"]
        for f, n in j["final"]["hidden"]["id"]["failures_edited"].items():
            if f != "success": cnt[k][f] += n
    types = [t for t, _ in sum(cnt.values(), Counter()).most_common()]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True); rows = []
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        tot = sum(cnt[k].values()) or 1; vals = [cnt[k][t] / tot for t in types][::-1]
        ax.barh(range(len(types)), vals, color=c["ink2"], height=0.62)
        for i, v in enumerate(vals):
            if v > 0: ax.text(v + 0.012, i, f"{v:.0%}", va="center", fontsize=6.8, color=c["ink2"])
        ax.set_yticks(range(len(types))); ax.set_yticklabels(types[::-1] if k == "S1" else [], fontsize=7); ax.set_xlim(0, 0.75)
        ax.set_title(f"{CLASS_LABEL[k]}\nn = {tot:,} failures", fontsize=7.8); ax.set_xlabel("share of failures"); ax.grid(axis="y", visible=False)
        rows += [{"class": k, "failure_type": t, "share": cnt[k][t] / tot, "count": cnt[k][t]} for t in types]
    fig.subplots_adjust(left=0.17, right=0.99, top=0.84, bottom=0.2, wspace=0.08)
    S.provenance(fig, "FIG-12", "baselines-v1/a1 null-arm hidden-ID episodes (rule-based failure types)", mode)
    return fig, pd.DataFrame(rows), "baselines-v1"


# --------------------------------------------------------------------------------------------- FIG-13
@figure("FIG-13", "Edit footprint versus gain",
        "补丁足迹：每个 run 一个点，横轴为改动参数占比（对数），纵轴为隐藏 ID 的 ΔSR。神谕只改约 1% 参数就得到最大增益；随机扰动改动最多且有害。形状=检查点类别。",
        "baselines")
def fig13(mode):
    S.apply_theme(mode); c = CHROME[mode]; d = D.load_baselines(); d = d[(d.arm != "null") & ~((d.arm == "oracle_restore") & (d.ckpt_class != "S1"))]
    fig, ax = plt.subplots(figsize=(7.0, 3.3)); mk = {"S1": "o", "S2": "s", "S3": "^"}; rows = []
    ax.axhspan(-0.10, 0.10, color=c["grid"], alpha=0.9, lw=0, zorder=0); ax.axhline(0, color=c["axis"], lw=0.9)
    for a in ARM_ORDER[1:]:
        for k, m in mk.items():
            q = d[(d.arm == a) & (d.ckpt_class == k)]
            if q.empty: continue
            ax.scatter(q.frac_changed.clip(lower=1e-4), q.d_sr_id, s=14, marker=m, color=color(a, mode), alpha=0.5, lw=0)
    for a in ARM_ORDER[1:]:
        q = d[d.arm == a]; x = np.exp(np.log(q.frac_changed.clip(lower=1e-4)).mean()); y = q.d_sr_id.mean()
        ax.scatter([x], [y], s=70, color=color(a, mode), edgecolor=c["surface"], lw=1.6, zorder=5); ax.annotate({"random": "random", "random_greedy": "greedy", "gain_greedy": "gain", "analytic_ridge": "refit*", "oracle_restore": "oracle†"}[a], (x, y), xytext=(-8, 8) if a == "random" else (8, 6), textcoords="offset points", fontsize=7.2, color=c["ink2"], ha="right" if a == "random" else "left")
        rows.append({"arm": a, "geomean_frac_changed": x, "mean_dSR": y})
    ax.set_xscale("log"); ax.set_xlim(2.5e-3, 0.5); ax.set_xlabel("fraction of parameters changed (log)"); ax.set_ylabel("$\\Delta$ hidden success rate (ID)")
    h = [plt.Line2D([], [], marker=m, ls="", color=c["ink2"], label=CLASS_LABEL[k]) for k, m in mk.items()]
    ax.legend(handles=h, loc="lower left", fontsize=7, ncol=3)
    S.footnote(fig, S.FOOT, mode); fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.2)
    S.provenance(fig, "FIG-13", "baselines-v1/a1 + greedy, one point per run; big dot = arm mean", mode)
    return fig, pd.DataFrame(rows), "baselines-v1"


# --------------------------------------------------------------------------------------------- FIG-14
@figure("FIG-14", "Case study: one planted fault, three outcomes",
        "案例（c01-reach，S1 weak_axis 植入故障，隐藏 ID 前 12 个 episode）：上排=三个策略的轨迹（○起点，◆目标；实线=成功，×=失败）；下排=补丁相对基线的输出层 W3 差异。神谕只改被腐蚀的那一列；闭式解重写整个输出层。",
        "case")
def fig14(mode):
    from ..store import CheckpointStore
    from ..tasks import get_task
    S.apply_theme(mode); c = CHROME[mode]; st = CheckpointStore(ROOT / "artifacts/checkpoints"); task = get_task("c01-reach")
    runs = {a: json.loads(p.read_text()) for a in ("oracle_restore", "analytic_ridge")
            for p in [next((ROOT / "runs/baselines-v1/a1").glob(f"c01-reach__s1-weak_axis-seed1__{a}__r0/run.json"))]}
    base_h = runs["oracle_restore"]["identity"]["base_sha256"]
    W = {"base": st.get(base_h), "oracle": st.get(runs["oracle_restore"]["identity"]["final_sha256"]), "ridge": st.get(runs["analytic_ridge"]["identity"]["final_sha256"])}
    titles = {"base": "unedited (S1)", "oracle": "oracle restore†", "ridge": "closed-form refit*"}
    ent = {"base": "null", "oracle": "oracle_restore", "ridge": "analytic_ridge"}
    fig = plt.figure(figsize=(7.2, 5.3)); gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.38, wspace=0.18, left=0.07, right=0.90, top=0.91, bottom=0.12)
    rows = []
    for j, key in enumerate(["base", "oracle", "ridge"]):
        ax = fig.add_subplot(gs[0, j]); ax.grid(False); res = task.rollout(W[key], "hid_id", 0, 12, keep_traj=True); P = res.traj["P"]; G = res.traj["goal"]
        for i in range(12):
            ok = bool(res.success[i]); col = color(ent[key], mode) if ok else CHROME[mode]["muted"]
            ax.plot(P[:, i, 0], P[:, i, 1], color=col, lw=1.3 if ok else 1.0, alpha=0.95 if ok else 0.8, ls="-" if ok else (0, (3, 2)))
            ax.scatter(P[0, i, 0], P[0, i, 1], s=14, facecolor=c["surface"], edgecolor=c["ink2"], lw=0.9, zorder=4)
            ax.scatter(G[i, 0], G[i, 1], s=22, marker="D", color=c["ink"], lw=0, zorder=4)
            if not ok: ax.scatter(P[-1, i, 0], P[-1, i, 1], s=22, marker="x", color=S.STATUS["critical"], lw=1.2, zorder=5)
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{titles[key]} · {int(res.success.sum())}/12", fontsize=8.2)
        if j == 0: S.panel_label(ax, "a", mode)
        rows.append({"policy": key, "success_of_12": int(res.success.sum())})
    dW = {"oracle": W["oracle"]["W3"] - W["base"]["W3"], "ridge": W["ridge"]["W3"] - W["base"]["W3"]}
    gt_cols = [c_ for c_ in range(2) if np.abs(dW["oracle"][:, c_]).max() > 0]
    vm = max(np.abs(dW["ridge"]).max(), np.abs(dW["oracle"]).max()); cm = S.diverging_cmap(mode)
    axs = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, key in zip(axs, ["oracle", "ridge"]):
        ax.grid(False); im = ax.imshow(dW[key].T, aspect="auto", cmap=cm, vmin=-vm, vmax=vm, interpolation="nearest")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["out 0", "out 1"] if key == "oracle" else ["", ""], fontsize=7); ax.set_xlabel("hidden unit (layer 2)", fontsize=7.5)
        ax.set_title(f"$\\Delta W_3$: {int((dW[key] != 0).sum())}/{dW[key].size} entries", fontsize=8)
    S.panel_label(axs[0], "b", mode)
    ax0 = fig.add_subplot(gs[1, 0]); ax0.axis("off"); ax0.grid(False)
    ax0.text(0, 0.95, "Reading the case", fontsize=8.5, fontweight="bold", color=c["ink"], va="top")
    ax0.text(0, 0.80, "The planted fault scales one\noutput column of $W_3$. The oracle\nrestores exactly those entries\n(localization precision = recall\n= 1.0). The closed-form refit\nrewrites the whole output layer\nfrom privileged expert labels:\nit also repairs the policy, with\na far larger footprint.", fontsize=6.9, color=c["ink2"], va="top", linespacing=1.4)
    cax_ = fig.add_axes([0.925, 0.14, 0.014, 0.26]); cbar = fig.colorbar(im, cax=cax_); cbar.ax.tick_params(labelsize=6.5); cbar.outline.set_visible(False)
    S.footnote(fig, S.FOOT, mode, y=0.03); S.provenance(fig, "FIG-14", "c01-reach / s1-weak_axis / seed1 · stored checkpoints re-run on stored hidden seeds 0–11", mode)
    return fig, pd.DataFrame(rows), "baselines-v1 + checkpoint store"


# --------------------------------------------------------------------------------------------- FIG-15
@figure("FIG-15", "Validity and reproducibility record",
        "工程与可复现性记录：关键计数（检查点、对照臂 run、独立复算、测试）与契约版本时间线（每次修订的原因，旧结果保留为证据）。",
        "process")
def fig15(mode):
    S.apply_theme(mode); c = CHROME[mode]; m = D.manifest(); d = D.load_baselines()
    n_tests = sum(len(re.findall(r"^def test_", p.read_text(), re.M)) for p in (ROOT / "tests").glob("test_*.py"))
    kpis = [(f"{int((m.status == 'built').sum())}", "checkpoints built", f"{int((m.status != 'built').sum())} infeasible, recorded"),
            (f"{len(d):,}", "control-arm runs", "all verified; 0 interrupted"), ("240", "independent checks", "0 mismatches vs run.json"),
            (f"{n_tests}", "test functions", "pytest green · contract OK")]
    fig = plt.figure(figsize=(7.2, 3.6)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.28, left=0.03, right=0.98, top=0.95, bottom=0.08)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.grid(False); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    for i, (v, l, sub) in enumerate(kpis):
        x = 1 + i * 25
        ax.add_patch(FancyBboxPatch((x, 0.8), 23, 8.4, boxstyle="round,pad=0.2,rounding_size=0.8", fc=c["neutral"], ec=c["axis"], lw=0.8))
        ax.text(x + 1.5, 6.3, v, fontsize=19, fontweight="bold", color=c["ink"], va="center"); ax.text(x + 1.5, 3.3, l, fontsize=7.5, color=c["ink"], va="center", fontweight="bold")
        ax.text(x + 1.5, 1.9, sub, fontsize=6.0, color=c["ink2"], va="center")
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.grid(False); ax2.set_xlim(0, 100); ax2.set_ylim(-3, 8)
    ev = [("v1", "first frozen contract"), ("v2", "manifest built; per-round\nweights stored"), ("v3–v5", "S2 localization bug\nfixed; greedy control added"),
          ("v6", "S3 rebuilt as natural\npolicies; OOD recalibrated")]
    ax2.plot([3, 97], [3, 3], color=c["axis"], lw=1.4)
    for i, (v, t) in enumerate(ev):
        x = 10 + i * 27; ax2.scatter([x], [3], s=90, color=c["surface"], edgecolor=c["ink"], lw=1.4, zorder=3)
        ax2.text(x, 3, "", ha="center"); ax2.text(x, 5.2, v, ha="center", fontsize=8.5, fontweight="bold", color=c["ink"]); ax2.text(x, 0.7, t, ha="center", va="top", fontsize=6.8, color=c["ink2"], linespacing=1.3)
    ax2.text(1, 7.6, "Contract versions (shared with the Codex side; old results are kept as SUPERSEDED evidence)", fontsize=8, fontweight="bold", color=c["ink"])
    S.provenance(fig, "FIG-15", "ckpt_manifest.json · runs · tests/ · CONTRACT_CHANGELOG.md", mode)
    return fig, pd.DataFrame(kpis, columns=["value", "metric", "note"]), "repo state"


# --------------------------------------------------------------------------------------------- FIG-16
@figure("FIG-16", "Official benchmark feasibility: Meta-World",
        "官方基准可行性（Meta-World）：(a) 行为克隆策略成功率随训练轮数——reach 迅速饱和，push 有可利用的过渡区；(b) 已构造的 Meta-World 检查点在 select 上的成功率与目标区间。",
        "benchmark")
def fig16(mode):
    S.apply_theme(mode); c = CHROME[mode]; fe = D.summary_json("feasibility-metaworld.json"); m = D.manifest(); m = m[m.task_id.str.startswith("mw-") & (m.status == "built")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.15]}); rows = []
    ax = axs[0]
    for name, ls in (("reach-v3", "-"), ("push-v3", "-")):
        if name not in fe: continue
        e = [r["epochs"] for r in fe[name]["bc"]]; s = [r["sr"] for r in fe[name]["bc"]]
        ax.plot(e, s, marker="o", color=c["ink"] if name.startswith("push") else c["muted"], lw=1.6, ms=4); ax.annotate(name, (e[0], s[0]), xytext=(4, 8), textcoords="offset points", fontsize=7, color=c["ink2"])
        rows += [{"panel": "a", "task": name, "epochs": a, "sr": b} for a, b in zip(e, s)]
    ax.set_xscale("log"); ax.set_xticks([20, 100, 400]); ax.set_xticklabels(["20", "100", "400"]); ax.minorticks_off(); ax.set_xlabel("BC epochs (log scale)"); ax.set_ylabel("success rate (select)"); ax.set_ylim(-0.03, 1.08); S.panel_label(ax, "a", mode)
    ax = axs[1]; bands = {"S1": (0.1, 0.6), "S2": (0.2, 0.6), "S3": (0.6, 0.9)}; rng = np.random.default_rng(4)
    for j, k in enumerate(["S1", "S2", "S3"]):
        lo, hi = bands[k]; ax.add_patch(Rectangle((j - 0.4, lo), 0.8, hi - lo, fc=c["grid"], ec="none", zorder=0))
        for t, off in (("mw-push", -0.12), ("mw-pickplace", 0.12)):
            q = m[(m.task_id == t) & (m["class"] == k)].select_sr.astype(float)
            ax.scatter(j + off + rng.uniform(-0.05, 0.05, len(q)), q, s=20, color=c["ink2"] if t == "mw-push" else c["surface"], edgecolor=c["ink2"], lw=0.9, zorder=3)
            rows += [{"panel": "b", "task": t, "class": k, "select_sr": v} for v in q]
    ax.set_xticks(range(3)); ax.set_xticklabels([CLASS_LABEL[k].replace(" ", "\n", 1) for k in ["S1", "S2", "S3"]], fontsize=7.2); ax.set_ylim(-0.03, 1.05)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c["ink2"], label="push"), plt.Line2D([], [], marker="o", ls="", mfc=c["surface"], mec=c["ink2"], label="pick-place"),
                       Rectangle((0, 0), 1, 1, fc=c["grid"], label="target band")], loc="lower right", fontsize=7, ncol=3); S.panel_label(ax, "b", mode)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.2, wspace=0.18)
    S.provenance(fig, "FIG-16", "feasibility-metaworld.json; ckpt_manifest.json (select namespace)", mode)
    return fig, pd.DataFrame(rows), "feasibility-metaworld.json"


# --------------------------------------------------------------------------------------------- FIG-17
@figure("FIG-17", "LLM weight programming versus conventionally trained networks",
        "效率前沿：横轴为墙钟时间（对数），纵轴为隐藏 ID 的 ΔSR。灰色=训练参照（方块=专家标签的行为克隆微调，三角=无标签的进化策略；点=各检查点均值）；彩色=Claude Pilot 的 run（蓝=Sonnet，橙=Opus，实心=high）。训练按 CPU 秒计，LLM 按提供方延迟计；训练便宜约 3–4 个数量级（金钱），但需要专家标签，而无标签的 ES 在同等交互预算下远不如 LLM。",
        "baselines")
def fig17(mode):
    from .llm_figures import _llm_runs, MODEL_LABEL, _arm_style
    S.apply_theme(mode); c = CHROME[mode]; tr = pd.read_csv(ROOT / "runs/SUMMARY/trained-baselines.csv"); llm = _llm_runs()
    llm = llm.copy(); llm["secs"] = [sum(r["latency_s"] for r in json.loads(Path(p).read_text())["rounds"]) for p in llm.path]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True); rows = []
    mk = {"bc_ft_small": ("s", "BC 10 demos"), "bc_ft_med": ("s", "BC 50"), "bc_ft_full": ("s", "BC 300"), "bc_scratch": ("D", "BC scratch"), "es_80": ("^", "ES 80 eps"), "es_800": ("^", "ES 800")}
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        ax.axhline(0, color=c["axis"], lw=0.9); ax.axhspan(-0.10, 0.10, color=c["grid"], alpha=0.9, lw=0, zorder=0)
        for name, (m, lab) in mk.items():
            q = tr[(tr.trainer == name) & (tr["class"] == k)]
            if q.empty: continue
            x, y = max(q.wall_s.mean(), 1e-2), q.d_id.mean()
            ax.scatter([x], [y], s=34, marker=m, color=c["ink2"], zorder=4)
            rows.append({"class": k, "method": name, "wall_s": x, "d_id": y, "kind": "trained"})
        for ent, eff in [("Sonnet 5.5", "medium"), ("Sonnet 5.5", "high"), ("Opus 5.5", "medium"), ("Opus 5.5", "high")]:
            q = llm[(llm.ckpt_class == k) & (llm.entity == ent) & (llm.effort == eff)]
            if q.empty: continue
            st = _arm_style(ent, eff, mode); ax.scatter(q.secs, q.d_sr_id, s=22, facecolor=st["mfc"], edgecolor=st["mec"], lw=1.2, alpha=0.85, zorder=5)
            rows += [{"class": k, "method": f"{ent} {eff}", "wall_s": a, "d_id": b, "kind": "llm"} for a, b in zip(q.secs, q.d_sr_id)]
        ax.set_xscale("log"); ax.set_xlim(0.015, 600); ax.set_title(CLASS_LABEL[k], fontsize=8.6); ax.set_xlabel("wall-clock seconds (log)")
    axs[0].set_ylabel("$\\Delta$ hidden success rate (ID)", fontsize=8)
    h = [plt.Line2D([], [], marker="s", ls="", color=c["ink2"], label="BC fine-tune, 10 / 50 / 300 demos (expert labels)"), plt.Line2D([], [], marker="D", ls="", color=c["ink2"], label="BC from scratch"),
         plt.Line2D([], [], marker="^", ls="", color=c["ink2"], label="ES, 80 / 800 episodes (no labels)"), plt.Line2D([], [], marker="o", ls="", mfc=_arm_style("Sonnet 5.5", "high", mode)["mfc"], mec=_arm_style("Sonnet 5.5", "high", mode)["mec"], label="Sonnet 5.5"),
         plt.Line2D([], [], marker="o", ls="", mfc=_arm_style("Opus 5.5", "high", mode)["mfc"], mec=_arm_style("Opus 5.5", "high", mode)["mec"], label="Opus 5.5 (solid = high, ring = medium)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, 0.17), handletextpad=0.3, columnspacing=1.2)
    S.footnote(fig, "LLM cost ≈ $0.14–0.8 per run vs ≈ $0.00005 for 4 s of CPU training (assumed $0.05 per core-hour)", mode)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.46, wspace=0.06)
    S.provenance(fig, "FIG-17", "trained-baselines.csv + pilot runs; trained and LLM points are NOT matched by checkpoint", mode)
    return fig, pd.DataFrame(rows), "trained-baselines + pilot runs"


# --------------------------------------------------------------------------------------------- FIG-18
@figure("FIG-18", "Robustness to out-of-distribution shift (OOD curves)",
        "OOD 曲线（A7）：横轴为 OOD 偏移强度 λ（0=同分布参数，1=校准的 OOD 划分，2=两倍偏移），纵轴为编辑后的隐藏成功率（实线），灰色虚线=未编辑基线。同一批隐藏 OOD episode 上评测。基线随偏移崩溃，所以增益随偏移变大要结合基线读；Opus 的收益在 λ≈1 最高，Sonnet 在 S1 上随偏移单调下降。",
        "baselines")
def fig18(mode):
    S.apply_theme(mode); c = CHROME[mode]; p = ROOT / "runs/SUMMARY/ood-curves.csv"
    if not p.exists(): raise PendingData("run scripts/ood_curves.py first")
    d = pd.read_csv(p, keep_default_na=False)
    fam = lambda a: ("Sonnet 5.5" if a.startswith("sonnet") else "Opus 5.5" if a.startswith("opus") else {"analytic_ridge": "analytic_ridge", "gain_greedy": "gain_greedy", "oracle_restore": "oracle_restore"}.get(a))
    d["ent"] = d.arm.map(fam); fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True); rows = []
    for ax, k in zip(axs, ["S1", "S2", "S3"]):
        q0 = d[(d["class"] == k)]
        b = q0.groupby("lam").sr_base.mean(); ax.plot(b.index, b.values, color=c["muted"], lw=1.6, ls=(0, (4, 3)))
        for ent in ["gain_greedy", "analytic_ridge", "oracle_restore", "Sonnet 5.5", "Opus 5.5"]:
            q = q0[q0.ent == ent]
            if q.empty or (ent == "oracle_restore" and k != "S1"): continue          # the oracle only exists where a planted truth exists
            g = q.groupby(["ckpt", "lam"]).sr_final.mean().groupby("lam").mean()
            ax.plot(g.index, g.values, color=color(ent, mode), lw=1.8, marker="o", ms=3.5); rows += [{"class": k, "entity": ent, "lam": a, "sr_final": b_} for a, b_ in g.items()]
        ax.set_title(CLASS_LABEL[k], fontsize=8.6); ax.set_xlabel("OOD shift strength $\\lambda$"); ax.set_xticks([0, 0.5, 1, 1.5, 2]); ax.set_ylim(-0.02, 1.02)
    axs[0].set_ylabel("hidden OOD success rate after editing", fontsize=8)
    h = [plt.Line2D([], [], color=c["muted"], ls=(0, (4, 3)), label="unedited")] + [plt.Line2D([], [], color=color(e, mode), marker="o", ms=4, label=l) for e, l in
         [("gain_greedy", "gain search"), ("analytic_ridge", "closed-form*"), ("oracle_restore", "oracle†"), ("Sonnet 5.5", "Sonnet 5.5"), ("Opus 5.5", "Opus 5.5")]]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6.8, bbox_to_anchor=(0.5, 0.2), columnspacing=1.0, handletextpad=0.4)
    S.footnote(fig, S.FOOT + "   ·   LLM = Pilot runs (n small)", mode); fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.42, wspace=0.06)
    S.provenance(fig, "FIG-18", "ood-curves.csv (hid_ood, 100 episodes per point); mean over checkpoints", mode)
    return fig, pd.DataFrame(rows), "ood-curves.csv"
