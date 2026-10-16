#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${STARVLA_PYTHON:-python}"

"${PYTHON_BIN}" -c 'import torch, torchvision' >/dev/null 2>&1 || {
    echo "[ERROR] Install CUDA-compatible torch and torchvision before running this script." >&2
    exit 1
}

"${PYTHON_BIN}" -m pip install -r "${PACKAGE_ROOT}/requirements-policy-server.txt"
"${PYTHON_BIN}" -m pip install --no-deps -e "${PACKAGE_ROOT}/transformer"

echo "[OK] Python dependencies and the bundled custom Transformers are installed."
echo "[NEXT] Install a torch/CUDA-compatible flash-attn, then run:"
echo "       ${PYTHON_BIN} ${PACKAGE_ROOT}/verify_install.py"

