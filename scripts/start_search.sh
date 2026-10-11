#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/start_search.sh {text|multimodal} [--dry-run]'
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_scad_env.sh"
scad_modality "$1"
[[ $# -le 2 && ( $# -eq 1 || "$2" == --dry-run ) ]] || { echo 'Unknown argument.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${SEARCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export SCAD_ENCODER_MAX_BATCH_SIZE="${SCAD_ENCODER_MAX_BATCH_SIZE:-64}"
export SCAD_ENCODER_MAX_BATCH_TOKENS="${SCAD_ENCODER_MAX_BATCH_TOKENS:-131072}"
export SCAD_ENCODER_RELEASE_CACHE="${SCAD_ENCODER_RELEASE_CACHE:-1}"
module=envs.search_server
extra=()
if [[ "${SCAD_MODALITY}" == multimodal ]]; then
  module=envs.mm_search_server
  extra+=(--document-root "${MM_CORPUS_ROOT}/documents" --rerank-workers "${MM_RERANK_WORKERS:-4}" --flat-candidate-k 4096)
fi
command=("${PYTHON_BIN}" -m "${module}" --model "${EMBEDDING_MODEL_PATH}"
  --passage-root "${PASSAGE_ROOT}" --embedding-root "${EMBEDDING_ROOT}" --ann-root "${ANN_ROOT}"
  --backend ann_cpu --sources "${SCAD_SOURCES}" --devices 0 --encoder-device 0
  --ann-candidates "${ANN_CANDIDATES:-32768}" --ann-nprobe "${ANN_NPROBE:-512}"
  --faiss-threads "${FAISS_THREADS}" --host "${SEARCH_HOST:-127.0.0.1}" --port "${SEARCH_PORT}" "${extra[@]}")
if [[ "${2:-}" == --dry-run ]]; then
  scad_print_command "${command[@]}"
  exit 0
fi
"${PYTHON_BIN}" -m scripts.prepare_scad_data --package-root "${PACKAGE_ROOT}" --output-root "${RUNTIME_ROOT}" --check-retrieval \
  --embedding-root "${EMBEDDING_ROOT}" --ann-root "${ANN_ROOT}"
exec "${command[@]}"
