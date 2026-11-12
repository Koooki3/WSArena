"""Unified figure style (dataviz method; palette = validated reference instance, see reports/figures/PALETTE_VALIDATION.txt).

Rules enforced here, not left to each figure:
  * colour follows the ENTITY, never its rank (ENTITY_SLOT is global and fixed; never cycled);
  * categorical hues come from the 8-slot validated order; no-op / reference entities are neutral gray;
  * one y-axis per axes (no dual axes); thin marks; hairline grid; direct labels + legend for >= 2 series;
  * every figure carries a provenance footer (figure id, contract version, source hash, data) and a CSV table view;
  * light and dark variants are separate selected steps (not an automatic inversion).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]

CHROME = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  neutral="#f0efec", on_fill="#ffffff"),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 neutral="#383835", on_fill="#ffffff"),
}
CAT = {  # slots 1..8 (validated, fixed order)
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# entity -> categorical slot (1-based). GLOBAL and fixed: a colour means the same thing in every figure.
ENTITY_SLOT = {
    "Sonnet 5.5": 1, "Opus 5.5": 2,                         # LLM editors (effort is encoded by fill, not hue)
    "analytic_ridge": 3, "random_greedy": 4, "random": 5, "oracle_restore": 6,   # scripted controls
    "GPT (Codex side)": 7,                                  # reserved for the Codex arms (effort = fill, not hue)
    "gain_greedy": 8,                                       # one-parameter output-gain search (strongest simple control found so far)
}
NEUTRAL_ENTITIES = {"null"}                                  # reference / no-op: gray, never a hue
ARM_LABEL = {"null": "no edit", "random": "random sparse", "gain_greedy": "output-gain search", "random_greedy": "greedy random search",
             "analytic_ridge": "closed-form refit*", "oracle_restore": "oracle restore†"}
ARM_ORDER = ["null", "random", "random_greedy", "gain_greedy", "analytic_ridge", "oracle_restore"]
CLASS_LABEL = {"S0": "S0 zero", "S1": "S1 planted fault", "S2": "S2 natural failure", "S3": "S3 healthy"}
TASK_LABEL = {"c01-reach": "reach", "c02-avoid": "avoid", "c03-hold": "hold", "c04-delay": "delay",
              "mw-push": "MW push", "mw-pickplace": "MW pick-place", "mw-peginsert": "MW peg-insert"}
FOOT = "*uses privileged expert labels   †uses planted-fault ground truth"


def color(entity: str, mode: str = "light") -> str:
    if entity in NEUTRAL_ENTITIES:
        return CHROME[mode]["muted"]
    return CAT[mode][ENTITY_SLOT[entity] - 1]


def diverging_cmap(mode: str = "light"):
    mid = CHROME[mode]["neutral"]
    return LinearSegmentedColormap.from_list("wsa_div", [CAT[mode][7], mid, CAT[mode][0]])   # red - neutral - blue


def seq_cmap(mode: str = "light"):
    ramp = SEQ_BLUE if mode == "light" else SEQ_BLUE[::-1][:11]
    if mode == "dark":                       # low = dark (near surface), high = light
        ramp = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#b7d3f6"]
    return LinearSegmentedColormap.from_list("wsa_seq", ramp)


def apply_theme(mode: str = "light"):
    c = CHROME[mode]
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "savefig.facecolor": c["surface"],
        "text.color": c["ink"], "axes.labelcolor": c["ink2"], "xtick.color": c["muted"], "ytick.color": c["muted"],
        "axes.edgecolor": c["axis"], "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": c["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 5, "pdf.fonttype": 42, "svg.fonttype": "path",
        "figure.dpi": 110, "savefig.dpi": 220,
    })


def panel_label(ax, text, mode="light"):
    ax.text(-0.02, 1.08, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=CHROME[mode]["ink"], va="bottom", ha="right")


def contract_version() -> int | None:
    p = ROOT / "docs/handoff/FROZEN_CONTRACT.json"
    return json.loads(p.read_text())["version"] if p.exists() else None


def _bottom_room(fig, inches):
    """Guarantee `inches` of free space under the axes for the footnote + provenance lines."""
    h = fig.get_figheight()
    try:
        sp = fig.subplotpars
        if sp.bottom < inches / h and not getattr(fig, "_wsa_fixed_layout", False):
            fig.subplots_adjust(bottom=inches / h)
    except Exception:
        pass


def footnote(fig, text, mode="light", y=None):
    """Footnote line (0.36 in above the bottom edge); always clear of the provenance line (0.10 in)."""
    _bottom_room(fig, 1.0)
    fig.text(0.01, 0.36 / fig.get_figheight(), text, fontsize=6.8, color=CHROME[mode]["ink2"], ha="left", va="bottom")


def provenance(fig, fig_id: str, data_note: str, mode="light"):
    from ..version import source_hash
    _bottom_room(fig, 0.72)
    fig.text(0.01, 0.10 / fig.get_figheight(), f"WSArena · {fig_id} · contract v{contract_version()} · src {source_hash()[:8]} · {data_note}",
             fontsize=6.3, color=CHROME[mode]["muted"], ha="left", va="bottom")


def end_label(ax, x, y, text, entity_color, mode="light", dx=4, **kw):
    """Direct label: the swatch carries identity (colour), the text stays in ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", fontsize=7.5, va="center",
                color=CHROME[mode]["ink2"], **kw)


def save_all(fig, fig_id: str, outdir: Path, mode: str, formats=("png", "pdf", "svg")) -> list[str]:
    d = outdir / mode; d.mkdir(parents=True, exist_ok=True); paths = []
    for f in formats:
        p = d / f"{fig_id}.{f}"; fig.savefig(p); paths.append(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
    plt.close(fig); return paths
