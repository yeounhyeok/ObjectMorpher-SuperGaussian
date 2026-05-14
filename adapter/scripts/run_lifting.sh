#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_CROP="$(realpath -m "${1:-${REPO_ROOT}/runs/sam_outputs/objects/input_object.png}")"
OUTPUT_PLY="$(realpath -m "${2:-${REPO_ROOT}/runs/coarse.ply}")"
OUTPUT_GLB="${OUTPUT_GLB:+$(realpath -m "${OUTPUT_GLB}")}"

if [[ ! -f "${INPUT_CROP}" ]]; then
  echo "ERROR: input crop not found at ${INPUT_CROP}" >&2
  echo "Hint: run SAM first via adapter/scripts/run_sam.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PLY}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

PYTHON_BIN="${OM_PYTHON:-/opt/conda/bin/python}"

ARGS=(--input-crop "${INPUT_CROP}" --output-ply "${OUTPUT_PLY}" --repo-root "${REPO_ROOT}")
if [[ -n "${OUTPUT_GLB}" ]]; then
  mkdir -p "$(dirname "${OUTPUT_GLB}")"
  ARGS+=(--output-glb "${OUTPUT_GLB}")
fi

cd "${REPO_ROOT}"
exec env PYTHONPATH="${REPO_ROOT}/adapter${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m om_sg_adapter.cli.run_lifting "${ARGS[@]}"
