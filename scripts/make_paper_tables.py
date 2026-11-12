"""Generate paper/gen/*.tex from the result reports (no numbers are typed by hand). usage: make_paper_tables.py"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "paper/gen"; OUT.mkdir(parents=True, exist_ok=True)
LAB = {"sonnet55-medium": "Sonnet med.", "sonnet55-high": "Sonnet high", "opus55-medium": "Opus med.", "opus55-high": "Opus high"}
CON = {"C1": "vs.\\ random", "C2": "vs.\\ closed form", "C3": "S3 $-$ L2"}
def primary(md):
    sec = md.split("## Primary comparisons")[1].split("##")[0]; rows = []
    for l in sec.splitlines():
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) == 8 and c[0] in LAB: rows.append(c)
    return rows
def table(rows, cap, label):
    t = ["\\begin{table}[t]", "\\centering", f"\\caption{{{cap}}}", f"\\label{{{label}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llrcr@{}}", "\\toprule", "Editor & Contrast & Diff & 95\\% CI & Holm $p$ \\\\", "\\midrule"]
    for a, i, _c, n, d, ci, p, ph in rows:
        ci = ci.replace("[", "").replace("]", "").replace("+", "").replace("-", "$-$").replace(", ", ", ")
        t.append(f"{LAB[a]} & {CON[i]} & {d.replace('-', '$-$').replace('+', '')} & [{ci}] & {ph} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]; return "\n".join(t) + "\n"
for name, f, cap, lab in (("t0", "MAIN_RESULTS.md", "Tier-0, pre-registered comparisons (16 checkpoints; L2 = faulty or failing checkpoints, 10; S3 = healthy, 6). Differences in hidden success rate; Holm correction over the twelve tests.", "tab:t0"),
                          ("mw", "MAIN_RESULTS_MW.md", "Meta-World, pre-registered comparisons (17 checkpoints; 11 L2, 6 S3). Holm correction over the six tests.", "tab:mw")):
    (OUT / f"main_{name}.tex").write_text(table(primary((ROOT / "reports" / f).read_text()), cap, lab))
(OUT / "main_table.tex").write_text("% see main_t0.tex and main_mw.tex\n"); print("wrote", [p.name for p in OUT.glob("*.tex")])
