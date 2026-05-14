#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_CROP="$(realpath -m "${1:-${REPO_ROOT}/runs/sam_outputs/objects/input_object.png}")"
INPUT_PLY="$(realpath -m "${2:-${REPO_ROOT}/runs/coarse.ply}")"
OUT_DIR="$(realpath -m "${3:-${REPO_ROOT}/runs/tail_lift}")"

if [[ ! -f "${INPUT_CROP}" ]]; then
  echo "ERROR: SAM crop not found at ${INPUT_CROP}" >&2; exit 1
fi
if [[ ! -f "${INPUT_PLY}" ]]; then
  echo "ERROR: input ply not found at ${INPUT_PLY}" >&2; exit 1
fi
mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${OM_PYTHON:-/opt/conda/bin/python}"

cd "${REPO_ROOT}"
exec env PYTHONPATH="${REPO_ROOT}/adapter${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m om_sg_adapter.cli.run_tail_lift \
  --input-crop "${INPUT_CROP}" \
  --input-ply "${INPUT_PLY}" \
  --out-dir "${OUT_DIR}" \
  --repo-root "${REPO_ROOT}" \
  "${@:4}"
