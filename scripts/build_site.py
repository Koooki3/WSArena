"""Build the LOCAL project site (reports/site/*.html): Overview / Benchmark details / Results / Case study / Reproducibility.
Modelled on the way VABAR organises its release (docs/PUBLISHING_PLAYBOOK_ZH.md). Numbers come from recorded artifacts only;
anything without data is labelled 'pending'. Nothing is published: this writes files under reports/site/ and stops.
  .venv/bin/python scripts/build_site.py
Prerequisite: scripts/make_figures.py (reads reports/figures/manifest.json)."""
import html, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks as T
from wsarena.viz import data as D, style
from wsarena.version import source_hash

OUT = ROOT / "reports" / "site"; OUT.mkdir(parents=True, exist_ok=True)
fm = json.loads((ROOT / "reports/figures/manifest.json").read_text()); figs = {f["id"]: f for f in fm["figures"]}
m = D.manifest(); base = D.load_baselines(); nf = D.summary_json("noise_floor.json")
CV = style.contract_version(); SH = source_hash()[:8]

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--rule:#e1e0d9;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--rule:#2c2c2a;--accent:#3987e5}}
:root[data-theme=dark]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--rule:#2c2c2a;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
header,main,footer{max-width:980px;margin:0 auto;padding:0 16px}header{padding-top:28px}
nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--rule);padding:10px 0 12px;margin-bottom:8px}nav a{color:var(--ink2);text-decoration:none;font-size:14px}nav a.on{color:var(--ink);font-weight:600;border-bottom:2px solid var(--accent)}
h1{font-size:30px;line-height:1.2;margin:18px 0 6px}h2{font-size:21px;margin:38px 0 8px}h3{font-size:16px;margin:22px 0 4px}p,li{color:var(--ink2)}.lede{font-size:17px;color:var(--ink2)}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}.tag{font-size:12px;border:1px solid var(--rule);border-radius:99px;padding:2px 10px;color:var(--ink2);background:var(--surface)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}.kpi{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}.kpi b{display:block;font-size:30px;color:var(--ink);line-height:1.1}.kpi span{font-size:13px;color:var(--ink2)}
figure{margin:18px 0;background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:10px}figure img{width:100%;height:auto;display:block}figcaption{font-size:13.5px;color:var(--ink2);padding:8px 4px 2px}figcaption b{color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0;background:var(--surface)}th,td{border:1px solid var(--rule);padding:6px 9px;text-align:left;vertical-align:top}th{color:var(--ink);background:var(--bg)}td{color:var(--ink2)}
.note{border-left:3px solid var(--accent);background:var(--surface);padding:8px 14px;margin:16px 0;color:var(--ink2);font-size:14.5px}.pending{border-left-color:var(--muted)}code,pre{font:13px ui-monospace,monospace;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:1px 5px}pre{padding:10px;overflow:auto}
footer{padding:30px 16px 50px;color:var(--muted);font-size:13px}.btn{font-size:13px;border:1px solid var(--rule);background:var(--surface);color:var(--ink2);border-radius:6px;padding:3px 10px;cursor:pointer}
"""
PAGES = [("index.html", "Overview"), ("details.html", "Benchmark details"), ("results.html", "Results"), ("case.html", "Case study"), ("repro.html", "Reproducibility")]


def fig(fid, extra=""):
    f = figs.get(fid)
    if not f or "light" not in f["files"]:
        return f'<div class="note pending">{fid}: pending data (no figure rendered yet).</div>'
    l = f"../figures/light/{fid}.png"; d = f"../figures/dark/{fid}.png"; svg = f"../figures/light/{fid}.svg"; csv = f"../figures/data/{fid}.csv"
    return (f'<figure id="{fid}"><picture><source media="(prefers-color-scheme: dark)" srcset="{d}"><img src="{l}" alt="{html.escape(f["title"])}" loading="lazy"></picture>'
            f'<figcaption><b>{fid} · {html.escape(f["title"])}.</b> {html.escape(f["caption_zh"])} {extra} '
            f'<a href="{svg}">svg</a> · <a href="{csv}">table view (csv)</a></figcaption></figure>')


def page(fname, title, body):
    nav = "".join(f'<a href="{h}" class="{"on" if h == fname else ""}">{t}</a>' for h, t in PAGES)
    (OUT / fname).write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSArena · {title}</title><style>{CSS}</style></head><body><header><nav>{nav}<span style="flex:1"></span>
<button class="btn" onclick="var r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark'">light / dark</button></nav></header>
<main>{body}</main><footer>WSArena · contract v{CV} · source {SH} · built {time.strftime('%Y-%m-%d %H:%M')} (+08) · local build, not published ·
figures: <a href="../FIGURES_INDEX.md">index</a></footer></body></html>""")


built = int((m.status == "built").sum()); infeas = int((m.status != "built").sum())
kp = f'<div class="kpis"><div class="kpi"><b>{built}</b><span>checkpoints (S0–S3), {infeas} infeasible recorded</span></div><div class="kpi"><b>{len(base):,}</b><span>scripted-control runs, all verified</span></div>' \
     f'<div class="kpi"><b>240</b><span>independent recompute checks, 0 mismatches</span></div><div class="kpi"><b>{nf["summary"]["median_sd_id"]:.3f}</b><span>median SD of SR over disjoint 200-episode sets</span></div></div>'

# ------------------------------------------------------------------ Overview
page("index.html", "Overview", f"""
<h1>WSArena: a controlled benchmark for LLM-driven weight-space programming</h1>
<p class="lede">Can a general-purpose LLM read the raw weights of a small policy network and rewrite them — without gradients, optimisers or search — so that the policy behaves better? WSArena measures this under one auditable protocol and scores five capabilities separately (reading, synthesis, repair, enhancement, generalisation).</p>
<div class="tags"><span class="tag">no training</span><span class="tag">hash-sealed checkpoints</span><span class="tag">hidden ID + OOD</span><span class="tag">paired evaluation</span><span class="tag">failures stay in the denominator</span><span class="tag">Claude first · GPT via Codex</span></div>
{kp}
<h2>What is weight-space programming?</h2>
<table><tr><th></th><th>Fine-tuning</th><th>Model editing (ROME-style)</th><th>Steering</th><th>Hyper-networks / LoRA generators</th><th>Weight-space programming (WSArena)</th></tr>
<tr><td>Who sets the weights</td><td>optimiser</td><td>fixed algorithm</td><td>runtime vector, weights untouched</td><td>a trained generator</td><td>a general LLM, by reading and writing numbers</td></tr>
<tr><td>Gradients / data</td><td>yes</td><td>few</td><td>few</td><td>yes (to train the generator)</td><td>none; only a protocol-limited feedback budget</td></tr>
<tr><td>Artifact</td><td>new checkpoint</td><td>new checkpoint</td><td>none (inference-time)</td><td>new checkpoint</td><td>new checkpoint of the <i>same architecture</i>, editor not needed at deployment</td></tr></table>
<div class="note">First public results (MIT CSAIL, 2026-11-02) were single-run and had no control arms; WSArena adds control arms, hidden OOD, selective-change metrics and an audit trail. See <a href="results.html">Results</a>.</div>
<h2>The evaluation pipeline</h2>{fig("FIG-01")}{fig("FIG-02")}
<h2>Where this fits</h2>
<p>The organisation of this record follows the way VABAR (Berkeley / Siemens / NVIDIA, arXiv 2611.06881) is documented: design criteria, a formal task definition, layered failure attribution, difficulty tiers, a case study — and adds what its release lacks: confidence intervals, seed control, versioning and a run-identity audit.</p>
""")

# ------------------------------------------------------------------ Details
rows = "".join(f"<tr><td>{T.TASKS[t].task_id}</td><td>{T.TASKS[t].level}</td><td>{T.TASKS[t].params_id.kind}</td><td>{T.TASKS[t].arch()}</td></tr>" for t in T.TASKS)
page("details.html", "Benchmark details", f"""
<h1>Benchmark details</h1>
<h2>Task ladder</h2><table><tr><th>task</th><th>rung</th><th>environment</th><th>policy architecture</th></tr>{rows}
<tr><td>mw-push, mw-pickplace</td><td>official tier (Meta-World v3)</td><td>Sawyer arm, 500-step episodes</td><td>(39, 24, 4)</td></tr></table>
<h2>Checkpoint classes</h2>
<table><tr><th>class</th><th>construction</th><th>target SR band</th><th>tests</th></tr>
<tr><td>S0 zero</td><td>all-zero weights</td><td>—</td><td>synthesis (L1)</td></tr><tr><td>S1 planted</td><td>healthy policy + localised corruption (ground truth known)</td><td>0.1–0.6 after corruption</td><td>repair (L2), localisation</td></tr>
<tr><td>S2 natural</td><td>trained on flawed demonstrations / data gap</td><td>0.2–0.6</td><td>repair (L2)</td></tr><tr><td>S3 healthy</td><td>naturally trained, no single-point fault</td><td>0.6–0.9</td><td>enhancement (L3)</td></tr></table>
{fig("FIG-04")}{fig("FIG-05")}{fig("FIG-11")}{fig("FIG-10")}
<h2>What the benchmark ships</h2>
<table><tr><th>component</th><th>content</th></tr><tr><td>Checkpoints</td><td>{built} sealed checkpoints (sha256), manifest with infeasible combinations recorded</td></tr>
<tr><td>Edit language</td><td>WEP-1: superset of the MIT numpy-slice edits syntax, deterministic parser, no eval</td></tr>
<tr><td>Harness</td><td>WSP-8 state machine, gateway contract + conformance suite (Claude CLI, Codex), replayable runs</td></tr>
<tr><td>Controls</td><td>null, random, greedy random search, closed-form refit (privileged), oracle restore (privileged)</td></tr>
<tr><td>Analysis</td><td>paired cluster bootstrap, sign-flip test, Holm; independent loop-based recompute</td></tr></table>
<h2>Task execution records</h2>{fig("FIG-03")}
<p>Each run stores <code>spec.json</code>, per-turn prompt/reply/result, per-round patch status and dev report, identity hashes and the hidden-evaluation arrays; see the run schema in <code>docs/FILE_MANAGEMENT_ZH.md §5</code>.</p>
""")

# ------------------------------------------------------------------ Results
page("results.html", "Results", f"""
<h1>Results</h1>
<div class="note pending"><b>LLM results: Pilot only.</b> 16 Claude runs (2 tasks × 2 checkpoints, one run per cell) establish cost, feasibility and failure modes; they carry <b>no inferential power</b> and are not main-study data (see <a href="../PILOT_V0_REPORT_ZH.md">pilot report</a>). The figures in the first block are the <b>scripted control arms</b> — the reference every LLM arm must be read against.</div>
{fig("FIG-06")}{fig("FIG-07")}{fig("FIG-08")}{fig("FIG-09")}{fig("FIG-13")}{fig("FIG-12")}
<h2>Reading the controls</h2>
<ul><li>On planted faults the privileged closed-form refit is almost as good as the oracle: an LLM must be judged against it.</li>
<li>On natural failures the closed-form refit is unstable — this is where an LLM can differ.</li>
<li>Greedy random search with the same feedback budget does nothing; random sparse edits are harmful.</li></ul>
<h2>LLM arms (templates ready)</h2>{fig("LLM-01")}{fig("LLM-04")}
""")

# ------------------------------------------------------------------ Case study
page("case.html", "Case study", f"""
<h1>Case study: one planted fault, three outcomes</h1>
<p>Failure → diagnosis → repair → result, on a single checkpoint. The unedited policy reaches 8 of 12 hidden goals; restoring the corrupted output column (oracle) or refitting the output layer from expert labels (closed form) reaches 12 of 12 — with very different footprints.</p>
{fig("FIG-14")}
<div class="note pending">A natural-failure (S2) case and an LLM case are added when the corresponding runs exist.</div>
""")

# ------------------------------------------------------------------ Reproducibility
page("repro.html", "Reproducibility", f"""
<h1>Reproducibility and release status</h1>{fig("FIG-15")}
<h2>Identity of every run</h2><p>checkpoint sha256 · source-tree hash · config hash · model id · effort · CLI version · system-prompt hash · per-turn usage. A run missing any field is refused by the summariser.</p>
<h2>How to reproduce (local)</h2><pre>cd projects/wsarena
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_frozen.py
.venv/bin/python scripts/run_experiment.py configs/experiments/baselines-v1.yaml --attempt a1
.venv/bin/python checks/recompute.py runs/baselines-v1/a1</pre>
<h2>Release checklist</h2>
<table><tr><th>item</th><th>status</th></tr><tr><td>Code, tests, contract versioning</td><td>local git (not pushed)</td></tr><tr><td>License</td><td><b>not chosen</b> — needs the authors' decision</td></tr><tr><td>Version number / changelog</td><td>contract v{CV}; <code>docs/handoff/CONTRACT_CHANGELOG.md</code></td></tr>
<tr><td>Leaderboard and submission rules</td><td>not started</td></tr><tr><td>Citation / BibTeX</td><td>not available until a report exists</td></tr><tr><td>Public release (site, repo, preprint)</td><td><b>not authorised</b>; this site is a local build</td></tr></table>
""")
print("site written to", OUT.relative_to(ROOT), [p.name for p in OUT.glob("*.html")])
