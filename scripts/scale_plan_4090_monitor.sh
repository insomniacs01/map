#!/bin/bash
set -euo pipefail

if [ "${MA_ALLOW_LEGACY_SCALE_PLAN_4090:-0}" != "1" ]; then
  echo "[DEPRECATED] ${0##*/} is legacy and disabled by default (see scripts/run_scale_plan_4090.sh)." >&2
  echo "To run anyway (NOT recommended): export MA_ALLOW_LEGACY_SCALE_PLAN_4090=1" >&2
  exit 2
fi

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
WATCHDOG="${REPO_ROOT}/scripts/scale_plan_4090_watchdog.sh"
STATUS_FILE="/tmp/scale_plan_4090.status"
MONITOR_LOG="/tmp/scale_plan_4090_monitor.log"

while true; do
  ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

  if ! pgrep -af "scale_plan_4090_watchdog.sh" >/dev/null 2>&1; then
    echo "MONITOR restart watchdog ${ts}" | tee -a "${STATUS_FILE}" >> "${MONITOR_LOG}"
    setsid -f bash "${WATCHDOG}"
  fi

  train_proc_count=$(pgrep -af "scripts/train.py" | wc -l | awk '{print $1}')
  loop_proc_count=$(pgrep -af "run_scale_plan_4090_loop.sh" | wc -l | awk '{print $1}')

  gpu_line="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | sed 's/;$/ /')
  fi

  latest_log=""
  latest_line=""
  age_sec="-1"
  if latest_log=$(ls -t "${REPO_ROOT}"/experiments/local_runs/*/pinhole_pose_depth_scale/train.log 2>/dev/null | head -n 1); then
    if [ -n "${latest_log}" ] && [ -f "${latest_log}" ]; then
      now_ts=$(date +%s)
      log_ts=$(stat -c '%Y' "${latest_log}" 2>/dev/null || echo 0)
      age_sec=$((now_ts - log_ts))
      latest_line=$(tail -n 1 "${latest_log}" 2>/dev/null | tr -d '\r')
    fi
  fi

  echo "ts=${ts} loop_proc=${loop_proc_count} train_proc=${train_proc_count} log_age_sec=${age_sec} log=${latest_log} gpu=${gpu_line} last_line=${latest_line}" >> "${MONITOR_LOG}"
  sleep 300
done
