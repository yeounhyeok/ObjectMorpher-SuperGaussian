#!/usr/bin/env bash
set -euo pipefail

echo "Repo root: $(cd "$(dirname "$0")/../.." && pwd)"
echo "Python3: $(command -v python3 || true)"
python3 --version || true
