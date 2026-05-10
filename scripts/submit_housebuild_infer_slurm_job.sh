#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_housebuild_infer_slurm_job.sh [config.env] [KEY=VALUE ...]

Examples:
  bash scripts/submit_housebuild_infer_slurm_job.sh
  bash scripts/submit_housebuild_infer_slurm_job.sh MODEL_DIR=/project/gzmcagent/LLM_Collab_Minecraft/runs/house_build/xxx/final_model
  bash scripts/submit_housebuild_infer_slurm_job.sh DATASET_JSON=/project/gzmcagent/LLM_Collab_Minecraft/house_build/dataset/unseen_eval.json NUM_TURNS=4 RUN_SUFFIX=unseen

Common overrides:
  MODEL_DIR=latest
  DATASET_JSON=/path/to/unseen_eval.json
  SPLIT='[:]'
  NUM_TURNS=4
  NO_SAMPLE=true
  PRINT_PROMPTS=true
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "${1}" != *=* ]]; then
  if [[ ! -f "${1}" ]]; then
    echo "Config file not found: ${1}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${1}"
  shift
fi

for kv in "$@"; do
  if [[ "${kv}" != *=* ]]; then
    echo "Invalid override: ${kv}. Use KEY=VALUE." >&2
    exit 1
  fi
  export "${kv}"
done

export REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-hb-infer-2gpu}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-gzmcagent}"
export SLURM_PARTITION="${SLURM_PARTITION:-normal}"
export SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-2}"
export SLURM_NODES="${SLURM_NODES:-1}"
export SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
export SLURM_TIME="${SLURM_TIME:-02:00:00}"
export SLURM_OUTPUT="${SLURM_OUTPUT:-slurm-%x-%j.out}"
export SLURM_ERROR="${SLURM_ERROR:-slurm-%x-%j.err}"
export MODULES="${MODULES:-Anaconda3}"
export CONDA_ENV="${CONDA_ENV:-llm-collab-mc}"
export MODEL_DIR="${MODEL_DIR:-latest}"
export DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/house_build/dataset/unseen_eval.json}"

SBATCH_ARGS=(
  --job-name "${SLURM_JOB_NAME}"
  --account "${SLURM_ACCOUNT}"
  --partition "${SLURM_PARTITION}"
  --nodes "${SLURM_NODES}"
  --gpus-per-node "${SLURM_GPUS_PER_NODE}"
  --ntasks-per-node "${SLURM_NTASKS_PER_NODE}"
  --cpus-per-task "${SLURM_CPUS_PER_TASK}"
  --time "${SLURM_TIME}"
  --output "${SLURM_OUTPUT}"
  --error "${SLURM_ERROR}"
  --export=ALL
)

if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  SBATCH_ARGS+=(--exclude "${SLURM_EXCLUDE}")
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Slurm job: ${SLURM_JOB_NAME}"
echo "Slurm account: ${SLURM_ACCOUNT}"
echo "Slurm partition: ${SLURM_PARTITION}"
echo "Slurm gpus/node: ${SLURM_GPUS_PER_NODE}"
echo "Conda env: ${CONDA_ENV}"
echo "Modules: ${MODULES}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "DATASET_JSON: ${DATASET_JSON}"

printf 'sbatch'
printf ' %q' "${SBATCH_ARGS[@]}" "${PROJECT_ROOT}/slurm/house_build_infer_2gpu.sbatch"
printf '\n'

if [[ "${SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${PROJECT_ROOT}/slurm/house_build_infer_2gpu.sbatch"
