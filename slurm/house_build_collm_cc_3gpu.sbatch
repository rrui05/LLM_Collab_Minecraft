#!/usr/bin/env bash
#SBATCH --job-name=hb-collm-cc-3gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --exclude=dgx-31,dgx-54

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_DIR}"

CONDA_ENV="${CONDA_ENV:-llm-collab-mc}"
MODULES="${MODULES:-Anaconda3}"
CONDA_SETUP="${CONDA_SETUP:-}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
CRITIC_MODEL="${CRITIC_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DTYPE="${DTYPE:-bf16}"

TRAIN_SPLIT="${TRAIN_SPLIT:-[:8]}"
EVAL_SPLIT="${EVAL_SPLIT:-[8:]}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-150}"
NUM_TURNS="${NUM_TURNS:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
AGENT_LEARNING_RATE="${AGENT_LEARNING_RATE:-5e-6}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-5e-6}"
VALUE_LOSS_COEF="${VALUE_LOSS_COEF:-0.6}"
ROLLOUT_BUFFER_SIZE="${ROLLOUT_BUFFER_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-true}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-true}"
DRY_RUN="${DRY_RUN:-false}"

WANDB_PROJECT="${WANDB_PROJECT:-house_build}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-collm-cc,maac,house_build,slurm,3gpu,h800,qwen3-4b}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_SUFFIX="${RUN_SUFFIX:-seed${SEED}}"
RUN_NAME="${RUN_NAME:-house_build_collm_cc_3gpu_${RUN_SUFFIX}_${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/runs/house_build_collm_cc}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-${REPO_DIR}/.cache}/llm_collab_minecraft}"
mkdir -p "${LOG_DIR}" "${RUN_DIR}/wandb" "${RUN_DIR}/output"

exec > >(tee -a "${LOG_DIR}/stdout.log") 2> >(tee -a "${LOG_DIR}/stderr.log" >&2)

echo "===== HouseBuild CoLLM-CC 3-GPU Slurm Run ====="
date
echo "repo_dir=${REPO_DIR}"
echo "run_dir=${RUN_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "slurm_node_list=${SLURM_NODELIST:-none}"
echo "sbatch_launch_args=${SBATCH_LAUNCH_ARGS:-}"
echo "dry_run=${DRY_RUN}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # Some Slurm jobs do not source the environment-modules init script by default.
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

export WANDB_MODE
export WANDB_PROJECT
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${RUN_DIR}/wandb/cache"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

echo "===== Git =====" | tee "${LOG_DIR}/env.txt"
{
  git rev-parse HEAD || true
  git status --short || true
  git remote -v || true
} | tee -a "${LOG_DIR}/env.txt"

echo "===== System =====" | tee -a "${LOG_DIR}/env.txt"
{
  hostname || true
  uname -a || true
  nvidia-smi || true
  conda info --envs || true
  "${CONDA_PYTHON[@]}" -c 'import sys; print(sys.executable)' || true
  "${CONDA_PYTHON[@]}" --version || true
  "${CONDA_PYTHON[@]}" - <<'PY'
import sys
import torch
import comlrl
import transformers
print("python_executable:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}:", torch.cuda.get_device_name(i))
print("comlrl:", getattr(comlrl, "__version__", "no_version"))
print("transformers:", transformers.__version__)
PY
} | tee -a "${LOG_DIR}/env.txt"

"${CONDA_PYTHON[@]}" -m pip freeze > "${LOG_DIR}/pip-freeze.txt"

if [[ "${RUN_PREFLIGHT}" =~ ^(true|TRUE|1|yes|YES)$ ]]; then
  echo "===== CoLLM-CC Preflight ====="
  "${CONDA_PYTHON[@]}" scripts/preflight_housebuild_collm_cc.py \
    --config house_build/configs/house_build_maac_config.yaml \
    --expect-cuda-devices 3
fi

(
  while true; do
    date
    nvidia-smi
    sleep "${NVIDIA_SMI_INTERVAL:-60}"
  done
) >> "${LOG_DIR}/nvidia-smi.log" 2>&1 &
NVIDIA_MONITOR_PID=$!
trap 'kill ${NVIDIA_MONITOR_PID} >/dev/null 2>&1 || true' EXIT

WANDB_ENTITY_OVERRIDE="wandb.entity=None"
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_OVERRIDE="wandb.entity='${WANDB_ENTITY}'"
fi

WANDB_TAG_OVERRIDE="wandb.tags=[]"
if [[ -n "${WANDB_TAGS}" ]]; then
  IFS=',' read -r -a WANDB_TAG_ARRAY <<< "${WANDB_TAGS}"
  WANDB_TAG_OVERRIDE="wandb.tags=["
  TAG_SEP=""
  for tag in "${WANDB_TAG_ARRAY[@]}"; do
    tag="${tag#"${tag%%[![:space:]]*}"}"
    tag="${tag%"${tag##*[![:space:]]}"}"
    tag="${tag//\'/}"
    if [[ -n "${tag}" ]]; then
      WANDB_TAG_OVERRIDE+="${TAG_SEP}'${tag}'"
      TAG_SEP=","
    fi
  done
  WANDB_TAG_OVERRIDE+="]"
fi

EVAL_SPLIT_OVERRIDE="dataset.eval_split='${EVAL_SPLIT}'"
case "${EVAL_SPLIT}" in
  None|none|null|NULL|"")
    EVAL_SPLIT_OVERRIDE="dataset.eval_split=None"
    ;;
esac

OVERRIDES=(
  "seed=${SEED}"
  "agent_model.name='${AGENT_MODEL}'"
  "agent_model.dtype='${DTYPE}'"
  "critic_model.name='${CRITIC_MODEL}'"
  "critic_model.dtype='${DTYPE}'"
  "dataset.train_split='${TRAIN_SPLIT}'"
  "${EVAL_SPLIT_OVERRIDE}"
  "output.base_dir='${RUN_DIR}/output'"
  "output.save_final_model=${SAVE_FINAL_MODEL}"
  "output.save_path='${RUN_DIR}/final_model'"
  "external.mode='score_feedback'"
  "external.original_prompt=true"
  "external.previous_response=true"
  "maac.parallel_training='mp'"
  "maac.agent_devices=['cuda:0','cuda:1']"
  "maac.critic_devices=['cuda:2']"
  "maac.num_agents=2"
  "maac.num_turns=${NUM_TURNS}"
  "maac.num_generations=${NUM_GENERATIONS}"
  "maac.num_train_epochs=${NUM_EPOCHS}"
  "maac.agent_learning_rate=${AGENT_LEARNING_RATE}"
  "maac.critic_learning_rate=${CRITIC_LEARNING_RATE}"
  "maac.value_loss_coef=${VALUE_LOSS_COEF}"
  "maac.max_new_tokens=${MAX_NEW_TOKENS}"
  "maac.rollout_buffer_size=${ROLLOUT_BUFFER_SIZE}"
  "maac.train_batch_size=${TRAIN_BATCH_SIZE}"
  "maac.eval_interval=${EVAL_INTERVAL}"
  "maac.eval_num_samples=${EVAL_NUM_SAMPLES}"
  "maac.eval_batch_size=${EVAL_BATCH_SIZE}"
  "wandb.enabled=true"
  "wandb.project='${WANDB_PROJECT}'"
  "${WANDB_ENTITY_OVERRIDE}"
  "wandb.run_name='${RUN_NAME}'"
  "wandb.dir='${RUN_DIR}/wandb'"
  "${WANDB_TAG_OVERRIDE}"
)

{
  printf '%s\n' "conda run --no-capture-output -n ${CONDA_ENV} python house_build/train/train_maac.py --config house_build/configs/house_build_maac_config.yaml --override \\"
  printf '  %q \\\n' "${OVERRIDES[@]}"
} > "${LOG_DIR}/command.txt"

printf '%s\n' "${OVERRIDES[@]}" > "${LOG_DIR}/overrides.txt"
printf 'sbatch %s %s\n' "${SBATCH_LAUNCH_ARGS:---export=ALL}" "${REPO_DIR}/slurm/house_build_collm_cc_3gpu.sbatch" > "${LOG_DIR}/slurm_submit_command.txt"

echo "===== Training Command ====="
cat "${LOG_DIR}/command.txt"
echo "===== Start Training ====="

case "${DRY_RUN}" in
  true|TRUE|1|yes|YES)
    echo "DRY_RUN=true; environment and command generation finished, skipping training."
    exit 0
    ;;
esac

"${CONDA_PYTHON[@]}" house_build/train/train_maac.py \
  --config house_build/configs/house_build_maac_config.yaml \
  --override "${OVERRIDES[@]}"

echo "===== Training Finished ====="
date
echo "run_dir=${RUN_DIR}"
