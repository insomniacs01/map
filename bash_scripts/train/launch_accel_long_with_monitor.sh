#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-${REPO_ROOT}/experiments/long_runs/$(date +%Y%m%d_%H%M%S)_accel_bg}"
ROOT="$(mkdir -p "${ROOT}" && cd "${ROOT}" && pwd)"

cd "${REPO_ROOT}"

nohup bash "${REPO_ROOT}/bash_scripts/train/accel_long_coop_det_runs.sh" "${ROOT}" > "${ROOT}/driver.log" 2>&1 &
TRAIN_PID=$!

nohup bash "${REPO_ROOT}/bash_scripts/monitor/monitor_long_run.sh" "${ROOT}" 120 > "${ROOT}/monitor.nohup" 2>&1 &
MONITOR_PID=$!

echo "RUN_ROOT=${ROOT}"
echo "TRAIN_PID=${TRAIN_PID}"
echo "MONITOR_PID=${MONITOR_PID}"
