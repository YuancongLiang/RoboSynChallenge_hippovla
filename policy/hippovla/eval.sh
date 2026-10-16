#!/usr/bin/env bash
# Run from the copied policy/hippovla directory in RoboSynChallenge.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
PYTHON_BIN="${ROBOSYN_PYTHON:-python}"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash eval.sh TASK CHECKPOINT [GPU=0] [EPISODES=50] [--KEY VALUE ...]" >&2
    exit 2
fi

TASK_NAME="$1"
CHECKPOINT="$2"
GPU_ID="${3:-0}"
EPISODES="${4:-50}"
if [[ $# -gt 4 ]]; then shift 4; else set --; fi
CHECKPOINT="$(realpath "${CHECKPOINT}")"
for required in "${CHECKPOINT}" "$(dirname "${CHECKPOINT}")/../config.yaml" "$(dirname "${CHECKPOINT}")/../dataset_statistics.json"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing checkpoint or matching metadata: ${required}" >&2
        exit 1
    fi
done

# The official runner selects the physical simulator GPU via --gpu_id.
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="${BENCHMARK_ROOT}${EMBODICHAIN_ROOT:+:${EMBODICHAIN_ROOT}:${EMBODICHAIN_ROOT}/embodichain_tasks}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BENCHMARK_ROOT}"
exec "${PYTHON_BIN}" -u scripts/eval_policy.py \
    --config "${POLICY_DIR}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" --setting random --model_name hippovla \
    --policy_ckpt_path "${CHECKPOINT}" --gpu_id "${GPU_ID}" \
    --max_episodes "${EPISODES}" \
    --host "${ROBOSYN_POLICY_HOST:-127.0.0.1}" \
    --port "${ROBOSYN_SERVER_PORT:-5711}" "$@"
