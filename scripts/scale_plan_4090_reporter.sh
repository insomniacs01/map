#!/bin/bash
set -euo pipefail

if [ "${MA_ALLOW_LEGACY_SCALE_PLAN_4090:-0}" != "1" ]; then
  echo "[DEPRECATED] ${0##*/} is legacy and disabled by default (see scripts/run_scale_plan_4090.sh)." >&2
  echo "To run anyway (NOT recommended): export MA_ALLOW_LEGACY_SCALE_PLAN_4090=1" >&2
  exit 2
fi

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
REPORT_LOG="/tmp/scale_plan_4090_report.log"
STATUS_FILE="/tmp/scale_plan_4090.status"

while true; do
  ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  latest_log=$(ls -t "${REPO_ROOT}"/experiments/local_runs/*/pinhole_pose_depth_scale/train.log 2>/dev/null | head -n 1 || true)
  run_tag="unknown"
  if [ -n "${latest_log}" ]; then
    run_tag=$(echo "${latest_log}" | awk -F'/' '{print $(NF-2)}')
  fi

  last_test_line="NONE"
  if [ -n "${latest_log}" ] && [ -f "${latest_log}" ]; then
    test_lines=$(rg -n "Test Epoch: \\[.*\\]  \\[203/204\\]" "${latest_log}" 2>/dev/null || true)
    if [ -n "${test_lines}" ]; then
      last_test_line=$(echo "${test_lines}" | tail -n 1 | sed 's/^[0-9]*://')
    fi
    if [ -z "${last_test_line}" ]; then
      last_test_line="NONE"
    fi
  fi

  last_train_line="NONE"
  if [ -n "${latest_log}" ] && [ -f "${latest_log}" ]; then
    last_train_line=$(tail -n 1 "${latest_log}" 2>/dev/null | tr -d '\r')
  fi

  train_active="0"
  if pgrep -af "scripts/train.py" >/dev/null 2>&1; then
    train_active="1"
  fi

  gpu_line="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | sed 's/;$/ /')
  fi

  echo "ts=${ts} run=${run_tag} train_active=${train_active} gpu=${gpu_line} last_test=${last_test_line} last_train=${last_train_line}" | tee -a "${REPORT_LOG}" >> "${STATUS_FILE}"
  sleep 1800
done
