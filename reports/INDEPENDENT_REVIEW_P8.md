# Independent review P8 (2026-11-08): findings and responses

Reviewer: a read-only subagent (same model family, fresh context, no project imports, no network). Scope: recomputation from raw `run.json`, pre-registration timeline, statistics review of `scripts/analyze_main.py`, claim audit of the paper. This is a self-organised review, not an external one; an outside reader or a different model family has not yet reviewed the work.

**Verdict (reviewer)**: headline numbers reproduce from the raw files to within 0.001 (per-arm means, control contrasts, sigma_run, WRI and its interval, exact sign-flip p). Primary analysis was frozen 10 to 12 s before the first run (local git and file birth times). Issues are in inference and wording, not arithmetic.

| ID | Severity | Finding | Response |
|---|---|---|---|
| F-T1 | important | Paper said all analysis scripts were committed before the first run; WRI and sigma_run come from later scripts | Fixed in paper: only C1–C3 plus the secondary family were frozen; WRI, run-to-run spread, attempt rates labelled exploratory |
| F-S1 | important | CI stratified by task is conditional on the tasks and too narrow with 4/4/2 checkpoints per stratum | Unstratified intervals added (`reports/REVIEW_P8_ADDENDUM.md` §1, e.g. Sonnet-med vs closed form [−0.144, +0.449] vs [−0.092, +0.375]); paper says the table intervals are task-conditional. Frozen report unchanged. |
| F-S2 | important | No completeness guard; Holm family size not asserted | `scripts/check_completeness.py` added (counts, statuses, clusters, arms, family size); all six experiments pass |
| F-S3 | minor | Monte-Carlo sign-flip p for n=10 where exact enumeration is trivial | Exact enumeration in the addendum (0.0020 / 0.0010 for C1; values agree with the frozen report to Monte-Carlo error) |
| F-S4 | minor | Control pool mixes attempts for 25 checkpoints; descriptive table compares 10/6 LLM clusters with 41/14 control checkpoints | Noted; pairing is per checkpoint so contrasts are unaffected; the descriptive table is not used in the paper |
| F-S5 | minor | S3 minus L2 is mechanically negative (little headroom) | Paper states it is not a capability contrast |
| F-C1 | important | "Benefit comes from feedback and task description" unsupported; WRI interval ignored clustering; no equivalence margin; heterogeneous | Abstract, Results, Discussion, README rewritten: exploratory, clustered CI [−0.115, +0.115], large differences of both signs (6/16 within 0.05, 4/16 beyond 0.20), no feedback/task ablation was run |
| F-C2 | important | "Gains match a closed-form refit" overstated for n=10, MDE 0.175 | Now "not distinguishable (10 checkpoints, not evidence of equality)"; detectable difference about 0.18 stated |
| F-C3 | important | "differ by 0.08 to 0.13" misstates sigma_run; heavy tails; intervals overlap | Rewritten: sigma_run 0.129 (0.05–0.19) and 0.078 (0.02–0.12), RMS pair difference 0.18 and 0.11, without the two largest pairs 0.07 and 0.03 |
| F-C4 | important | "no real edit in 41%" conflates no valid patch with net-zero edit; abstention does not explain Opus | Both numbers given (Sonnet 7/17 no valid patch, 13/17 unchanged; Opus 0/17 and 7/17); explanation restricted to Sonnet |
| F-C5 | important | "beats random" is a weak bar | Stated as a low bar; null and gain search foregrounded; Opus vs gain search +0.02 on Meta-World |
| F-C6 | minor | "half of the checkpoints" imprecise | "10 of 17" |
| F-C7 | minor | 26% vs 64% not on the matched set; "more cautious" speculative | Matched set 0.33 vs 0.64; speculation removed |
| F-C8 | minor | Kendall tau and truncation not recomputed by the reviewer | Both computed by our scripts (tau 0.59 from `scripts/analyze_*`; 0 truncated rounds from run records); not independently verified |

**Residual items kept visible**: the unintended look at three gain values on 2026-11-07 (`runs/VIEW_LOG.tsv`); the C3 estimand correction (PREREG §10); the choice of arms in Phase 2 used Phase 1 results; timestamps come from local git with no external timestamping.
