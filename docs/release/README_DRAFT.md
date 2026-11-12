# WSArena

*Draft for the future public repository. Nothing here has been published; numbers are filled in from `reports/` after the main study.*

**Can a language model fix a neural network just by writing its weights?** No training, no gradients, no search, no tools. The model reads a small policy network as text, writes edits to the numbers, gets a short report on a few development episodes, and tries again. The edited network is then scored on hidden episodes the model never sees.

WSArena turns that question into a benchmark you can run and audit.

## What is in the box

- **A fixed protocol.** Four rounds, 16 development episodes per round, hidden in-distribution and out-of-distribution test episodes that are never fed back, no retries, the last round is scored.
- **Checkpoints with known ground truth.** Zero networks, networks with a planted fault (weak axis, bias shift, flipped unit), networks that fail on their own, and healthy networks that could still be better.
- **A ladder of controls.** Doing nothing, random edits, greedy edits, a one-knob gain search, a closed-form refit that sees expert labels, and an oracle that undoes the fault. A model has to beat the cheap controls before its edits count as skill.
- **Diagnostics that say what the model actually did.** Edit mechanism (scale, low-rank, local, rewrite), whether the model needed to see the weights at all (blind and shuffled prompts), whether the edit survives an equivalent re-arrangement of the weights, and what each run cost.
- **An official-benchmark tier.** Meta-World tasks with the same protocol, plus a compared-with-training track (behaviour cloning and evolution strategies) with a resource ledger.
- **Audit tooling.** Content-addressed checkpoints, bitwise regression checks, leak audit on every run, independent recomputation of scores, a frozen contract with a changelog.

## Quick start

```bash
git clone <repo> && cd wsarena
./scripts/setup_envs.sh          # builds the local virtualenvs
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_experiment.py configs/experiments/smoke-v0.yaml --attempt a1 --dry-run
```

Model calls go through the `claude` command line client on a subscription (no API key). Other providers plug in through the gateway contract in `docs/handoff/`.

## Headline results

Pre-registered study of Claude Sonnet 5.5 and Opus 5.5 (33 checkpoints, two task families; numbers from `reports/MAIN_RESULTS.md`, `MAIN_RESULTS_MW.md`, `A3_BLIND_CAP.md`; figures in `reports/figures/`).

- Every editor beats random edits (Holm p <= 0.024), a low bar; on Meta-World Opus only matches a one-parameter gain search.
- On the small control suite the editors are not distinguishable from a closed-form refit that sees expert labels (10 checkpoints; not evidence of equality); on Meta-World they fall clearly short of it (-0.22 to -0.35).
- Exploratory: hiding the weight values changes the gain by +0.006 on average (checkpoint-clustered 95% CI -0.115 to +0.115, 8 control-suite checkpoints), with large differences of both signs: we found no average benefit from seeing the values.
- The run-to-run standard deviation of an editor's gain on a fixed checkpoint is about 0.13 (control suite) and 0.08 (Meta-World), so rank models from checkpoint-level averages, not from single runs.
- Printing the weights of a width-256 policy takes about 246k tokens, which is why the benchmark has a separate probe track.

*(Scale-ladder and probe-track results will be added when those runs finish.)*

## Cite

*(BibTeX added at release)*

## License

*(chosen at release)*
