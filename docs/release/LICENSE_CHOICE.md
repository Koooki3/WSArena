# License choice (needs the owner's decision before release)
The code, the checkpoints and the data are different kinds of artefact. Suggested split, for the owner to confirm:
- **Code** (`wsarena/`, `scripts/`, `tests/`): MIT or Apache-2.0. Apache-2.0 adds an explicit patent grant; MIT is shorter and more common for research code.
- **Checkpoints and run records** (`artifacts/`, `runs/` summaries): CC BY 4.0 (attribution required).
- **Paper text and figures**: CC BY 4.0.
- **Third-party material**: nothing is vendored; the Meta-World, OCBench and Kinetix environments are installed from their own releases under their own licenses (see `configs/envs/*.freeze.txt`). The extracts under `survey/sources/` are quotations for review and must be re-checked for redistribution before release.
No license file is created until the owner chooses.
