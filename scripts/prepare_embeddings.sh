#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/prepare_embeddings.sh {text|multimodal} [--dry-run]'
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_scad_env.sh"
scad_modality "$1"
[[ $# -le 2 && ( $# -eq 1 || "$2" == --dry-run ) ]] || { echo 'Unknown argument.' >&2; exit 2; }
common=(--passages-root "${PASSAGE_ROOT}" --output-root "${EMBEDDING_ROOT}"
  --model-path "${EMBEDDING_MODEL_PATH}" --corpus-manifest "${PACKAGE_ROOT}/corpus/EMBEDDING_SPEC.json"
  --sources "${SCAD_SOURCES}" --dimension 2560 --max-length 512
  --max-batch-size "${EMBED_BATCH_SIZE:-256}" --max-batch-tokens "${EMBED_BATCH_TOKENS:-32768}")
encode=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1
  --nproc_per_node="${EMBED_N_GPUS:-1}" --module scripts.embed_unified_corpus "${common[@]}")
finalize=("${PYTHON_BIN}" -m scripts.embed_unified_corpus "${common[@]}" --finalize-only)
index=("${PYTHON_BIN}" -m scripts.build_ann_index --embedding-root "${EMBEDDING_ROOT}"
  --passage-root "${PASSAGE_ROOT}" --output-root "${ANN_ROOT}" --sources "${SCAD_SOURCES}"
  --threads "${FAISS_THREADS}" --ann-dimension 512 --seed 42)
if [[ "${2:-}" == --dry-run ]]; then
  scad_print_command "${encode[@]}"
  scad_print_command "${finalize[@]}"
  scad_print_command "${index[@]}"
  exit 0
fi
"${PYTHON_BIN}" -m scripts.prepare_scad_data --package-root "${PACKAGE_ROOT}" --output-root "${RUNTIME_ROOT}" --check-space
"${encode[@]}"
"${finalize[@]}"
"${index[@]}"
