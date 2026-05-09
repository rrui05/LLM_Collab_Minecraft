#!/usr/bin/env bash
set -euo pipefail

# Create a clean conda environment for reproducing HouseBuild on a Slurm cluster.
# Usage:
#   bash scripts/setup_superpod_env.sh
#
# Optional environment variables:
#   CONDA_ENV=llm-collab-mc
#   PYTHON_VERSION=3.11
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   MODULES="anaconda3 cuda/12.1"
#   CONDA_SETUP=/path/to/conda.sh

CONDA_ENV="${CONDA_ENV:-llm-collab-mc}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh || true
fi

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1 && [[ -n "${MODULES:-}" ]]; then
  for mod in ${MODULES}; do
    echo "module load ${mod}"
    module load "${mod}"
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "${CONDA_SETUP:-}" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${conda_sh}" && -f "${conda_sh}" ]]; then
      # shellcheck source=/dev/null
      source "${conda_sh}"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load your cluster's conda/anaconda module first, or run with MODULES='anaconda3' / CONDA_SETUP=/path/to/conda.sh." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
  echo "Conda env '${CONDA_ENV}' already exists. Reusing it."
else
  conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip
fi

conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install --upgrade pip
conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install torch --index-url "${TORCH_INDEX_URL}"
conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install comlrl==1.3.7 datasets wandb accelerate openai

conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import sys
import torch
import comlrl
import transformers

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"cuda:{idx}", torch.cuda.get_device_name(idx))
print("comlrl:", getattr(comlrl, "__version__", "no_version"))
print("transformers:", transformers.__version__)
PY
