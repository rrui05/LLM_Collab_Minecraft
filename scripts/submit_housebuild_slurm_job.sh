#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_housebuild_slurm_job.sh [config.env] [KEY=VALUE ...]

Examples:
  bash scripts/submit_housebuild_slurm_job.sh DRY_RUN=true RUN_SUFFIX=dryrun
  bash scripts/submit_housebuild_slurm_job.sh configs/cluster/house_build_2gpu.env DRY_RUN=true RUN_SUFFIX=dryrun
  bash scripts/submit_housebuild_slurm_job.sh RUN_SUFFIX=smoke AGENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct TRAIN_SPLIT='[:1]' EVAL_SPLIT=None NUM_EPOCHS=1 NUM_TURNS=1 MAX_NEW_TOKENS=32 EVAL_INTERVAL=0 SAVE_FINAL_MODEL=false

This wrapper submits slurm/house_build_magrpo_2gpu.sbatch with explicit
sbatch flags. Some clusters require --account on the sbatch command line and
do not accept only #SBATCH --account inside the batch script.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "${1:-}" != *=* ]]; then
  CONFIG_PATH="$1"
  shift
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Launch config not found: $CONFIG_PATH" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_PATH"
  set +a
  export LAUNCH_CONFIG_PATH="$CONFIG_PATH"
fi

for override in "$@"; do
  if [[ "$override" != *=* ]]; then
    echo "Unsupported override '$override'. Use KEY=VALUE form." >&2
    usage >&2
    exit 1
  fi
  export "$override"
done

export REPO_DIR="${REPO_DIR:-$PROJECT_ROOT}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-hb-magrpo-2gpu}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-gzmcagent}"
export SLURM_PARTITION="${SLURM_PARTITION:-normal}"
export SLURM_NODES="${SLURM_NODES:-1}"
export SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-2}"
export SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
export SLURM_TIME="${SLURM_TIME:-30:00:00}"
export SLURM_OUTPUT="${SLURM_OUTPUT:-slurm-%x-%j.out}"
export SLURM_ERROR="${SLURM_ERROR:-slurm-%x-%j.err}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"

export CONDA_ENV="${CONDA_ENV:-llm-collab-mc}"
export MODULES="${MODULES:-Anaconda3}"

if [[ -z "$SLURM_ACCOUNT" ]]; then
  echo "SLURM_ACCOUNT is required." >&2
  exit 1
fi

SBATCH_ARGS=(
  --job-name "$SLURM_JOB_NAME"
  --account "$SLURM_ACCOUNT"
  --partition "$SLURM_PARTITION"
  --nodes "$SLURM_NODES"
  --gpus-per-node "$SLURM_GPUS_PER_NODE"
  --ntasks-per-node "$SLURM_NTASKS_PER_NODE"
  --cpus-per-task "$SLURM_CPUS_PER_TASK"
  --time "$SLURM_TIME"
  --output "$SLURM_OUTPUT"
  --error "$SLURM_ERROR"
  --export ALL
)

if [[ -n "$SLURM_EXCLUDE" ]]; then
  SBATCH_ARGS+=(--exclude "$SLURM_EXCLUDE")
fi

SBATCH_LAUNCH_ARGS=""
for ((index=0; index<${#SBATCH_ARGS[@]}; index++)); do
  value="${SBATCH_ARGS[$index]}"
  if [[ "$value" == "--export" && $((index + 1)) -lt ${#SBATCH_ARGS[@]} ]]; then
    next="${SBATCH_ARGS[$((index + 1))]}"
    SBATCH_LAUNCH_ARGS+="$value=$next "
    index=$((index + 1))
    continue
  fi
  SBATCH_LAUNCH_ARGS+="$value "
done
export SBATCH_LAUNCH_ARGS="${SBATCH_LAUNCH_ARGS% }"

cat <<EOF
Project root: $PROJECT_ROOT
Slurm job: $SLURM_JOB_NAME
Slurm account: $SLURM_ACCOUNT
Slurm partition: $SLURM_PARTITION
Slurm gpus/node: $SLURM_GPUS_PER_NODE
Slurm cpus/task: $SLURM_CPUS_PER_TASK
Slurm time: $SLURM_TIME
Conda env: $CONDA_ENV
Modules: $MODULES
DRY_RUN: ${DRY_RUN:-false}
EOF

echo "sbatch ${SBATCH_LAUNCH_ARGS} $PROJECT_ROOT/slurm/house_build_magrpo_2gpu.sbatch"

if [[ "${SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "$PROJECT_ROOT/slurm/house_build_magrpo_2gpu.sbatch"
