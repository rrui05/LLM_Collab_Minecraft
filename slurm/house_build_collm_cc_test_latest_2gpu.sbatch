#!/usr/bin/env bash
#SBATCH --job-name=hb-cc-test-2gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_DIR}"

CONDA_ENV="${CONDA_ENV:-llm-collab-mc}"
MODULES="${MODULES:-Anaconda3}"
CONDA_SETUP="${CONDA_SETUP:-}"

MODEL_DIR="${MODEL_DIR:-latest}"
MODEL_KIND="${MODEL_KIND:-best}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${REPO_DIR}/runs/house_build_collm_cc}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/house_build/configs/house_build_maac_config.yaml}"
DATASET_JSON="${DATASET_JSON:-}"
SPLIT="${SPLIT:-}"
NUM_TURNS="${NUM_TURNS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.6}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-42}"
EXTERNAL_MODE="${EXTERNAL_MODE:-score_feedback}"
AGENT_DEVICES="${AGENT_DEVICES:-cuda:0,cuda:1}"
NO_SAMPLE="${NO_SAMPLE:-false}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_SUFFIX="${RUN_SUFFIX:-latest_best}"
RUN_NAME="${RUN_NAME:-house_build_collm_cc_test_${RUN_SUFFIX}_${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/runs/house_build_collm_cc_tests}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_MD="${RESULT_MD:-${RUN_DIR}/results/latest_model_test.md}"
mkdir -p "${LOG_DIR}" "${RUN_DIR}/results"

exec > >(tee -a "${LOG_DIR}/stdout.log") 2> >(tee -a "${LOG_DIR}/stderr.log" >&2)

echo "===== HouseBuild CoLLM-CC Latest Checkpoint Test ====="
date
echo "repo_dir=${REPO_DIR}"
echo "run_dir=${RUN_DIR}"
echo "model_dir=${MODEL_DIR}"
echo "model_kind=${MODEL_KIND}"
echo "train_run_root=${TRAIN_RUN_ROOT}"
echo "result_md=${RESULT_MD}"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "slurm_node_list=${SLURM_NODELIST:-none}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh || true
fi
if command -v module >/dev/null 2>&1; then
  if [[ -n "${MODULES}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      echo "conda already found; skipping MODULES='${MODULES}'"
    else
      for mod in ${MODULES}; do
        echo "module load ${mod}"
        module load "${mod}"
      done
    fi
  fi
  module list || true
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ -n "${CONDA_SETUP}" && -f "${CONDA_SETUP}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SETUP}"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "/opt/conda/etc/profile.d/conda.sh"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Set MODULES='Anaconda3' or CONDA_SETUP=/path/to/conda.sh when submitting." >&2
  exit 1
fi

CONDA_PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)

{
  echo "===== Git ====="
  git rev-parse HEAD || true
  git status --short || true
  git remote -v || true
  echo "===== System ====="
  hostname || true
  nvidia-smi || true
  "${CONDA_PYTHON[@]}" -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.device_count())' || true
} | tee "${LOG_DIR}/env.txt"

CMD=(
  "${CONDA_PYTHON[@]}"
  house_build/infer_latest_maac.py
  --config "${CONFIG_PATH}"
  --model-dir "${MODEL_DIR}"
  --run-root "${TRAIN_RUN_ROOT}"
  --model-kind "${MODEL_KIND}"
  --num-turns "${NUM_TURNS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --external-mode "${EXTERNAL_MODE}"
  --agent-devices "${AGENT_DEVICES}"
  --output-md "${RESULT_MD}"
)

if [[ -n "${DATASET_JSON}" ]]; then
  CMD+=(--dataset-json "${DATASET_JSON}")
fi
if [[ -n "${SPLIT}" ]]; then
  CMD+=(--split "${SPLIT}")
fi

case "${NO_SAMPLE}" in
  true|TRUE|1|yes|YES)
    CMD+=(--no-sample)
    ;;
esac

printf '%q ' "${CMD[@]}" > "${LOG_DIR}/command.txt"
printf '\n' >> "${LOG_DIR}/command.txt"

echo "===== Test Command ====="
cat "${LOG_DIR}/command.txt"
echo "===== Start Test ====="

"${CMD[@]}"

echo "===== Finished ====="
echo "result_md=${RESULT_MD}"
