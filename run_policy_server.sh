#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_RUN_DIR="${PACKAGE_ROOT}/models/hippovla"
CKPT_PATH="${1:-${ROBOSYN_CHECKPOINT:-${DEFAULT_RUN_DIR}/final_model/pytorch_model.pt}}"
BASE_VLM_PATH="${2:-${STARVLA_BASE_VLM:-${PACKAGE_ROOT}/models/RynnBrain8B}}"
GPU_ID="${3:-${ROBOSYN_SERVER_GPU:-0}}"
PORT="${4:-${ROBOSYN_SERVER_PORT:-5711}}"
PYTHON_BIN="${STARVLA_PYTHON:-python}"
IDLE_TIMEOUT="${ROBOSYN_SERVER_IDLE_TIMEOUT:--1}"
CONFIG_PATH="${5:-${STARVLA_CONFIG_PATH:-}}"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [[ ! -f "${BASE_VLM_PATH}/config.json" ]]; then
    echo "[ERROR] RynnBrain8B config.json not found under: ${BASE_VLM_PATH}" >&2
    exit 1
fi

RUN_DIR="$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)"
for metadata_file in config.yaml dataset_statistics.json; do
    if [[ ! -f "${RUN_DIR}/${metadata_file}" ]]; then
        echo "[ERROR] Missing ${RUN_DIR}/${metadata_file}" >&2
        echo "        The checkpoint must be under <run_dir>/final_model/ or <run_dir>/checkpoints/." >&2
        exit 1
    fi
done

USE_BF16_ARGS=()
if [[ "${ROBOSYN_USE_BF16:-1}" != "0" ]]; then
    USE_BF16_ARGS+=(--use_bf16)
fi

CONFIG_ARGS=()
if [[ -n "${CONFIG_PATH}" ]]; then
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "[ERROR] Model config not found: ${CONFIG_PATH}" >&2
        exit 1
    fi
    CONFIG_ARGS+=(--config_path "${CONFIG_PATH}")
fi

export PYTHONPATH="${PACKAGE_ROOT}/transformer/src:${PACKAGE_ROOT}:${PYTHONPATH:-}"
export STARVLA_BASE_VLM="$(cd "${BASE_VLM_PATH}" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DEBUG=""
if [[ "${ROBOSYN_SERVER_DEBUG:-0}" == "1" ]]; then
    export DEBUG="1"
fi

"${PYTHON_BIN}" "${PACKAGE_ROOT}/deployment/model_server/verify_robosyn_runtime.py"

echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Checkpoint: ${CKPT_PATH}"
echo "[INFO] Base VLM: ${STARVLA_BASE_VLM}"
echo "[INFO] GPU: ${GPU_ID}"
echo "[INFO] Listen: 0.0.0.0:${PORT}"
echo "[INFO] Model config: ${CONFIG_PATH:-${RUN_DIR}/config.yaml}"

exec "${PYTHON_BIN}" "${PACKAGE_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${CKPT_PATH}" \
    "${CONFIG_ARGS[@]}" \
    --port "${PORT}" \
    --idle_timeout "${IDLE_TIMEOUT}" \
    "${USE_BF16_ARGS[@]}"
