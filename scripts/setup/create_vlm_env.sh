#!/usr/bin/env bash
# Creates the pinned vision_llm_lora virtual env with uv and records every installed version in
# requirements_vlm_lock.txt. Run once on the login node (needs internet), from any directory:
#   bash scripts/setup/create_vlm_env.sh
# Needs uv on PATH (one-time, into ~/.local/bin: curl -LsSf https://astral.sh/uv/install.sh | sh).
# The env, uv's cache and its Python 3.10 build all live under /project: the personal home quota (100k files,
# 20 GB) is too small. uv hard-links packages from the cache into the env, which only works on the same filesystem.
# Activate with: source /project/nr_e2e_vlm/envs/navsim_vlm/bin/activate
# Afterwards, on a GPU node: python scripts/setup/check_vlm_env.py
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/project/nr_e2e_vlm/envs/navsim_vlm}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/project/nr_e2e_vlm/envs/uv_cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/project/nr_e2e_vlm/envs/uv_python}
export UV_LINK_MODE=hardlink
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

command -v uv > /dev/null || { echo "uv not found; install it first (see the header of this script)" >&2; exit 1; }

uv venv --python 3.10 "$ENV_PREFIX"
PYTHON="$ENV_PREFIX/bin/python"

uv pip install --python "$PYTHON" torch==2.8.0 torchvision==0.23.0
uv pip install --python "$PYTHON" -r "$REPO/requirements.txt" -c "$REPO/requirements_vlm.txt"
uv pip install --python "$PYTHON" flash-linear-attention -c "$REPO/requirements_vlm.txt"
uv pip install --python "$PYTHON" --no-deps -e "$REPO"
# No flash-attn / causal-conv1d: their prebuilt wheels need glibc 2.32, Komondor (RHEL 8) has 2.28. Attention uses
# PyTorch SDPA (agent YAML), the linear-attention layers use fla's Triton kernels, which compile on the machine.

uv pip check --python "$PYTHON" || echo "WARNING: uv pip check reported conflicts (see above)"
uv pip freeze --python "$PYTHON" --exclude-editable > "$REPO/requirements_vlm_lock.txt"
echo "Env ready: source $ENV_PREFIX/bin/activate   (versions in requirements_vlm_lock.txt)"
