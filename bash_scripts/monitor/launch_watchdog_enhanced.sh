#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QUEUE_FILE="${1:-${REPO_ROOT}/experiments/long_runs/auto_queue_pinhole_geom_det_e2e_v1.txt}"
LOG_ROOT="${2:-${REPO_ROOT}/experiments/long_runs/$(basename "${QUEUE_FILE}" .txt)_logs}"

# Auto-fix + infinite retries + no-advance on failure.
export STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
export ADVANCE_ON_FAIL="${ADVANCE_ON_FAIL:-0}"
export STOP_ON_GATE_FAIL="${STOP_ON_GATE_FAIL:-0}"
export GATE_AUTO_FIX="${GATE_AUTO_FIX:-1}"

# Auto-tune geometry gate thresholds.
export GATE_TUNE="${GATE_TUNE:-1}"
export GATE_TUNE_FAILS_BEFORE_RELAX="${GATE_TUNE_FAILS_BEFORE_RELAX:-2}"
export GATE_TUNE_RELAX_FACTOR="${GATE_TUNE_RELAX_FACTOR:-1.15}"
export GATE_TUNE_MAX_CAPS="${GATE_TUNE_MAX_CAPS:-depth_z_mae_m=3.0,depth_z_rmse_m=9.0,pose_trans_l2_m=1.0,pose_rot_deg=2.0,scale_err_mean=10.0}"

# Ensure retries never stop at a fixed count.
export MAX_RETRIES="${MAX_RETRIES:-0}"

mkdir -p "${LOG_ROOT}"

exec bash "${SCRIPT_DIR}/auto_queue_watchdog.sh" "${QUEUE_FILE}" "${LOG_ROOT}"
