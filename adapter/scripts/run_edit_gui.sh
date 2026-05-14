#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GS_PATH="$(realpath -m "${1:-${REPO_ROOT}/runs/coarse.ply}")"

if [[ ! -f "${GS_PATH}" ]]; then
  echo "ERROR: --gs_path not found at ${GS_PATH}" >&2; exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISPLAY="${DISPLAY:-:0}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"

PYTHON_BIN="${OM_PYTHON:-/opt/conda/bin/python}"

cd "${REPO_ROOT}"
exec env PYTHONPATH="${REPO_ROOT}/adapter:${REPO_ROOT}/ObjectMorpher/editing${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m om_sg_adapter.cli.run_edit_gui \
    --gs_path "${GS_PATH}" \
    --fit-camera \
    "${@:2}"
