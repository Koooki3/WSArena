#!/usr/bin/env bash
# Reproducible install of the official-benchmark environments, one venv each under venvs/ (gitignored).
# Pinned sources are recorded in configs/envs/*.freeze.txt after install. CPU variants first (this machine: RTX 4060
# 8 GB shared with the desktop, 11 GB RAM). Usage: scripts/setup_envs.sh metaworld|ocbench|kinetix
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; py=3.12
uv venv --python $py "venvs/$name" -q
P="venvs/$name/bin/python"
case "$name" in
  metaworld) uv pip install -q --python "$P" numpy scipy packaging "metaworld" gymnasium mujoco ;;
  ocbench)   uv pip install -q --python "$P" numpy scipy "ocbench" ;;                       # CPU variant (no mjwarp/jax)
  kinetix)   uv pip install -q --python "$P" numpy scipy "jax" "git+https://github.com/FLAIROx/Kinetix@80ee9c292837c825cafaf9889323fdcc212efd1b" ;;
  *) echo "unknown env $name"; exit 2 ;;
esac
# the harness itself runs inside each benchmark venv (the policy rollouts need the simulator)
uv pip install -q --python "$P" pydantic pyyaml pandas statsmodels pytest tabulate zstandard -e .
uv pip freeze --python "$P" > "configs/envs/$name.freeze.txt"
echo "installed $name: $(wc -l < configs/envs/$name.freeze.txt) packages"
