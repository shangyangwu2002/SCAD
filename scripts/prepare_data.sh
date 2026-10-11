#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/prepare_data.sh {text|multimodal} [--tables-only|--dry-run]'
  exit 0
fi
[[ $# -le 2 ]] || { echo 'Too many arguments.' >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/_scad_env.sh"
scad_modality "$1"
stage="${2:-all}"
case "${stage}" in all|--tables-only|--dry-run) ;; *) echo 'Unknown preparation stage.' >&2; exit 2 ;; esac
prepare=("${PYTHON_BIN}" -m scripts.prepare_scad_data --package-root "${PACKAGE_ROOT}" --output-root "${RUNTIME_ROOT}")
if [[ -n "${MM_PUBLIC_IMAGE_BASE_URL:-}" ]]; then
  prepare+=(--image-base-url "${MM_PUBLIC_IMAGE_BASE_URL}")
fi
if [[ "${stage}" == --dry-run ]]; then
  bash scripts/download_data.sh "${SCAD_MODALITY}" --dry-run
  scad_print_command "${prepare[@]}"
  exec bash scripts/prepare_embeddings.sh "${SCAD_MODALITY}" --dry-run
fi
bash scripts/download_data.sh "${SCAD_MODALITY}"
"${prepare[@]}"
if [[ "${stage}" != --tables-only ]]; then
  exec bash scripts/prepare_embeddings.sh "${SCAD_MODALITY}"
fi
