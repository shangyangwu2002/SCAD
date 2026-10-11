#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/train_2b.sh [--dry-run] (Qwen3-VL-2B multimodal student)'
  exit 0
fi
if [[ -z "${SCAD_EVAL_SPLIT:-}" ]]; then
  [[ $# -eq 0 || ( $# -eq 1 && "$1" == --dry-run ) ]] || { echo 'Unknown argument.' >&2; exit 2; }
fi
source "$(dirname "${BASH_SOURCE[0]}")/_scad_env.sh"
scad_modality multimodal
export MODEL_PATH="${MODEL_PATH:-${MODELS_ROOT}/Qwen/Qwen3-VL-2B-Instruct}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODELS_ROOT}/Qwen/Qwen3-VL-4B-Instruct}"
export N_GPUS="${N_GPUS:-2}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/scad_mm_2b_seed${SEED}}"
export MM_SAMPLER=stratified MM_SAMPLER_PRIMARY_PER_BATCH=2 MM_SAMPLER_SECONDARY_PER_BATCH=3
export MM_NODE_MERGE_EMBED_BATCH=16 VAL_BEFORE_TRAIN=False
export VAL_BATCH_SIZE=85
if [[ -n "${SCAD_EVAL_SPLIT:-}" ]]; then
  exec "${PYTHON_BIN}" -m scripts.evaluate --profile 2b --split "${SCAD_EVAL_SPLIT}" "$@"
fi
if [[ "${1:-}" != --dry-run ]]; then
  "${PYTHON_BIN}" -m scripts.prepare_scad_data --package-root "${PACKAGE_ROOT}" --output-root "${RUNTIME_ROOT}" \
    --check-training --search-url "${LOCAL_SEARCH_URL}"
fi
exec "${PYTHON_BIN}" -m scripts.train_scad_multimodal "$@"
