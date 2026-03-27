#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${SCRIPT_DIR}/download_hf_model.py" \
  --provider llamacpp \
  --model-id bartowski/Qwen_Qwen3.5-35B-A3B-GGUF \
  --output-dir "${REPO_ROOT}/models" \
  --file "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf" \
  "$@"
