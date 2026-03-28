#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  echo "Launching interactive benchmark runner"
  echo "Use comma-separated selections or ranges like 1,3-5 when prompted."
  echo ""
  python3 "${SCRIPT_DIR}/perf_benchmark.py" interactive
else
  python3 "${SCRIPT_DIR}/perf_benchmark.py" "$@"
fi
