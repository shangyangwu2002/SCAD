#!/usr/bin/env bash
# Shared, portable paths. Source this file from the public entry points.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ -f "${SCAD_ENV_FILE:-${PROJECT_ROOT}/.env}" ]]; then
  set -a
  source "${SCAD_ENV_FILE:-${PROJECT_ROOT}/.env}"
  set +a
fi
export SCAD_DATA_ROOT="${SCAD_DATA_ROOT:-${PROJECT_ROOT}/../SCAD_DATA}"
export SCAD_RUNTIME_ROOT="${SCAD_RUNTIME_ROOT:-${PROJECT_ROOT}/runtime}"
export MODELS_ROOT="${MODELS_ROOT:-${PROJECT_ROOT}/models}"
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-${MODELS_ROOT}/Qwen/Qwen3-Embedding-4B}"
export WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
export CPU_BUDGET="${CPU_BUDGET:-$("${PYTHON_BIN}" -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)')}"
export FAISS_THREADS="${FAISS_THREADS:-$(( CPU_BUDGET < 16 ? CPU_BUDGET : 16 ))}"

scad_modality() {
  case "$1" in
    text) SCAD_SOURCES=wiki18,browsecomp_plus,webwalker; DEFAULT_SEARCH_PORT=8000 ;;
    multimodal) SCAD_SOURCES=wiki18,browsecomp_plus,webwalker,mm_opd_v1; DEFAULT_SEARCH_PORT=8010 ;;
    *) echo 'Expected text or multimodal.' >&2; return 2 ;;
  esac
  export SCAD_MODALITY="$1"
  export PACKAGE_ROOT="${SCAD_DATA_ROOT}/${SCAD_MODALITY}"
  export RUNTIME_ROOT="${SCAD_RUNTIME_ROOT}/${SCAD_MODALITY}"
  # Derive the complete asset set together so legacy environment paths cannot
  # silently mix another data release with this release's readiness checks.
  export DATA_DIR="${RUNTIME_ROOT}/dataset"
  export PASSAGE_ROOT="${PACKAGE_ROOT}/corpus/passages"
  export EMBEDDING_ROOT="${RUNTIME_ROOT}/embeddings"
  export ANN_ROOT="${RUNTIME_ROOT}/indices"
  export SEARCH_PORT="${SEARCH_PORT:-${DEFAULT_SEARCH_PORT}}"
  export LOCAL_SEARCH_URL="${LOCAL_SEARCH_URL:-http://127.0.0.1:${SEARCH_PORT}}"
  export SCAD_SOURCES
  if [[ "$1" == multimodal ]]; then
    export MM_DATASET_ROOT="${PACKAGE_ROOT}/dataset"
    export MM_CORPUS_ROOT="${PACKAGE_ROOT}/corpus"
    export MM_IMAGE_SEARCH_CACHE="${MM_CORPUS_ROOT}/image_search_results.json"
    export MM_IMAGE_URL_MAP="${MM_IMAGE_URL_MAP:-${DATA_DIR}/image_url_map.json}"
  fi
}

scad_print_command() {
  printf '%q ' "$@"
  printf '\n'
}
