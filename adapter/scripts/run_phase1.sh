#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/adapter${PYTHONPATH:+:$PYTHONPATH}" python3 -m om_sg_adapter.cli.run_phase1 "$@"
