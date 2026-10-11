#!/usr/bin/env bash
set -euo pipefail
profile="$1"; shift
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash scripts/train_${profile}.sh [--dry-run]"
  exit 0
fi
if [[ -z "${SCAD_EVAL_SPLIT:-}" ]]; then
  [[ $# -eq 0 || ( $# -eq 1 && "$1" == --dry-run ) ]] || { echo 'Unknown argument.' >&2; exit 2; }
fi
source "$(dirname "${BASH_SOURCE[0]}")/_scad_env.sh"
scad_modality text
case "${profile}" in
  4b) student=Qwen3-4B; default_gpus=4; default_devices=0,1,2,3; default_util=0.62; initial_validation=True ;;
  1_7b) student=Qwen3-1.7B; default_gpus=2; default_devices=0,1; default_util=0.52; initial_validation=False ;;
  *) exit 2 ;;
esac
export MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/Qwen/${student}}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODELS_ROOT}/Qwen/Qwen3-4B-Instruct-2507}"
export N_GPUS="${N_GPUS:-${default_gpus}}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${default_devices}}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${default_util}}"
export SEED="${SEED:-42}" TOTAL_STEPS="${TOTAL_STEPS:-60}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/scad_text_${profile}_seed${SEED}}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export STRATIFIED_SOURCE_SAMPLING=True
export TRAIN_BATCH_SIZE=8 ROLLOUT_N=8
export VAL_BATCH_SIZE=130 VAL_FREQ=10 SAVE_FREQ=10
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${initial_validation}}"
export AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-32}"
export PROMPT_LENGTH=8192 RESPONSE_LENGTH=32768 BRANCH_LEN=16384
export OPEN_PAGE_TOKENS=2500 BRANCH_MIN_OBSERVATION_TOKENS=1024
export ADV_ESTIMATOR=scad
export SCAD_ENABLE=True
export SCAD_NODE_MERGE=True SCAD_NODE_SEMANTIC=True SCAD_TREE_CREDIT=True
export SCAD_SUBTASK_THRESHOLD=0.83 SCAD_ENTITY_THRESHOLD=0.90
export JUDGE_ENV_FILE="${SCAD_ENV_FILE:-${PROJECT_ROOT}/.env}"
export EXPERIMENT_NAME="scad_text_${profile}_seed${SEED}"
if [[ -n "${SCAD_EVAL_SPLIT:-}" ]]; then
  exec "${PYTHON_BIN}" -m scripts.evaluate --profile "${profile}" --split "${SCAD_EVAL_SPLIT}" "$@"
fi
if [[ "${1:-}" == --dry-run ]]; then
  export DRY_RUN=1
else
  shopt -s nullglob dotglob
  checkpoints=("${OUTPUT_DIR}/checkpoints/"*)
  if [[ "${RESUME_MODE}" == disable && ( -s "${OUTPUT_DIR}/metrics.jsonl" || ${#checkpoints[@]} -gt 0 || -f "${OUTPUT_DIR}/best_checkpoint.json" ) ]]; then
    echo 'Existing output directory; choose OUTPUT_DIR or set RESUME_MODE=auto.' >&2; exit 1
  fi
  "${PYTHON_BIN}" -m scripts.prepare_scad_data --package-root "${PACKAGE_ROOT}" --output-root "${RUNTIME_ROOT}" \
    --check-training --search-url "${LOCAL_SEARCH_URL}"
fi
exec bash scripts/_launch_text.sh
