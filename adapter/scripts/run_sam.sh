#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_IMAGE="$(realpath -m "${1:-${REPO_ROOT}/runs/input.png}")"
OUTPUT_DIR="$(realpath -m "${OM_SAM_OUTPUT_DIR:-${REPO_ROOT}/runs/sam_outputs}")"
CKPT="$(realpath -m "${OM_SAM_CKPT:-${REPO_ROOT}/ckpts/sam_vit_h_4b8939.pth}")"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: SAM checkpoint not found at ${CKPT}" >&2
  echo "Download with:" >&2
  echo "  mkdir -p $(dirname "${CKPT}")" >&2
  echo "  wget -O ${CKPT} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" >&2
  exit 1
fi
if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "ERROR: input image not found at ${INPUT_IMAGE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export OM_SAM_CKPT="${CKPT}"
export OM_SAM_OUTPUT_DIR="${OUTPUT_DIR}"
export OM_SAM_INPUT="${INPUT_IMAGE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISPLAY="${DISPLAY:-:0}"

PYTHON_BIN="${OM_PYTHON:-/opt/conda/bin/python}"

cd "${REPO_ROOT}/ObjectMorpher"
exec "${PYTHON_BIN}" -m preprocess.sam_processor
