"""Dynamic figures: mp4 animations of stored checkpoints behaving on the stored HIDDEN seeds (no re-training, no new model calls).
  V1 case: one planted-fault checkpoint -> unedited / oracle / closed-form / Sonnet / Opus edits (12 hidden episodes each)
  V2 avoid: c02 natural failure -> unedited / gain search / Sonnet / Opus (obstacle drawn; shows how each edit handles it)
  V3 rounds: one real LLM run -> behaviour after round 0,1,2,3,4 (weights from the immutable per-round checkpoints)
usage: .venv/bin/python scripts/make_videos.py [v1 v2 v3]"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks
from wsarena.store import CheckpointStore
from wsarena.viz import style as S
st = CheckpointStore(ROOT / "artifacts/checkpoints"); OUT = ROOT / "reports/videos"; OUT.mkdir(exist_ok=True)
S.apply_theme("light"); C = S.CHROME["light"]

def run_json(exp, att, rid): return json.loads((ROOT / "runs" / exp / att / rid / "run.json").read_text())
def final_w(j): return st.get(j["identity"]["final_sha256"])
def base_w(j): return st.get(j["identity"]["base_sha256"])

def animate(task, panels, fname, title, n_ep=12, fps=14, hold=18):
    """panels: list of (label, weights, colour). Same hidden episodes in every panel."""
    res = []
    for lab, w, col in panels:
        r = task.rollout(w, "hid_id", 0, n_ep, keep_traj=True); res.append((lab, col, r))
    T = res[0][2].traj["P"].shape[0]; k = len(panels)
    fig, axs = plt.subplots(1, k, figsize=(3.0 * k, 3.5)); axs = np.atleast_1d(axs); fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=C["ink"])
    lines = []; dots = []
    for ax, (lab, col, r) in zip(axs, res):
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        G = r.traj["goal"]; O = r.traj["obstacle"]
        ax.scatter(G[:, 0], G[:, 1], marker="D", s=14, color=C["ink"], zorder=3)
        if task.params_id.kind == "avoid":
            for o in O: ax.add_patch(Circle(o, task.params_id.obstacle_radius, fc=C["grid"], ec=C["axis"], lw=0.6, zorder=1))
        ls = [ax.plot([], [], color=(col if r.success[i] else C["muted"]), lw=1.0, alpha=0.9, zorder=2)[0] for i in range(n_ep)]
        ds = ax.scatter(np.zeros(n_ep), np.zeros(n_ep), s=18, facecolors=[col if r.success[i] else "none" for i in range(n_ep)], edgecolors=[col if r.success[i] else C["ink"] for i in range(n_ep)], linewidths=1.0, zorder=4)
        ax.set_title(f"{lab}  ·  {int(r.success.sum())}/{n_ep} reach", fontsize=8.6, color=C["ink"]); lines.append(ls); dots.append(ds)
    def upd(f):
        t = min(f, T - 1)
        for (lab, col, r), ls, ds in zip(res, lines, dots):
            P = r.traj["P"]
            for i, l in enumerate(ls): l.set_data(P[: t + 1, i, 0], P[: t + 1, i, 1])
            ds.set_offsets(P[t])
        return []
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.06)
    ani = FuncAnimation(fig, upd, frames=T + hold, interval=1000 / fps, blit=False)
    ani.save(OUT / fname, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=110); plt.close(fig); print("wrote", fname, flush=True)

want = set(sys.argv[1:]) or {"v1", "v2", "v3"}
if "v1" in want:
    t = tasks.get_task("c01-reach"); j = lambda a: run_json("baselines-v1", "a1", f"c01-reach__s1-weak_axis-seed1__{a}__r0")
    sm = run_json("pilot-v0", "a1", "c01-reach__s1-weak_axis-seed1__sonnet55-high__r0"); op = run_json("pilot-v0", "a1", "c01-reach__s1-weak_axis-seed1__opus55-high__r0")
    animate(t, [("unedited", base_w(j("oracle_restore")), C["muted"]), ("oracle restore", final_w(j("oracle_restore")), S.color("oracle_restore")),
                ("closed-form refit", final_w(j("analytic_ridge")), S.color("analytic_ridge")), ("Sonnet 5.5", final_w(sm), S.color("Sonnet 5.5")), ("Opus 5.5", final_w(op), S.color("Opus 5.5"))],
            "WSArena-V1.mp4", "One planted fault (reach task), same 12 hidden episodes: how each edit changes the behaviour (hollow ring = not solved)")
if "v2" in want:
    t = tasks.get_task("c02-avoid"); g = run_json("baselines-gain-v1", "a1", "c02-avoid__s2-seed1__gain_greedy__r0")
    sm = run_json("pilot-v0", "a1", "c02-avoid__s2-seed1__sonnet55-medium__r0"); op = run_json("pilot-v0", "a1", "c02-avoid__s2-seed1__opus55-medium__r0")
    animate(t, [("unedited", base_w(g), C["muted"]), ("output-gain search", final_w(g), S.color("gain_greedy")), ("Sonnet 5.5", final_w(sm), S.color("Sonnet 5.5")), ("Opus 5.5", final_w(op), S.color("Opus 5.5"))],
            "WSArena-V2.mp4", "Natural failure with an obstacle (gray discs): one-knob gain search vs LLM edits (hollow ring = episode not solved)", hold=24)
if "v3" in want:
    t = tasks.get_task("c02-avoid"); j = run_json("pilot-v0", "a1", "c02-avoid__s2-seed1__opus55-medium__r0")
    ids = j["identity"]["round_sha256"]; panels = [(f"round {r}", st.get(ids[str(r)] if str(r) in ids else ids[r]), S.color("Opus 5.5") if r else C["muted"]) for r in range(0, 5)]
    animate(t, panels, "WSArena-V3.mp4", "Opus 5.5 editing one checkpoint: behaviour after each round (same hidden episodes)", hold=24)
