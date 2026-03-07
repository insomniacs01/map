#!/bin/bash
set -euo pipefail

if [ "${MA_ALLOW_LEGACY_SCALE_PLAN_4090:-0}" != "1" ]; then
  echo "[DEPRECATED] ${0##*/} is legacy and disabled by default (see scripts/run_scale_plan_4090.sh)." >&2
  echo "To run anyway (NOT recommended): export MA_ALLOW_LEGACY_SCALE_PLAN_4090=1" >&2
  exit 2
fi

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
LOOP_SCRIPT="${REPO_ROOT}/scripts/run_scale_plan_4090_loop.sh"
STATUS_FILE="/tmp/scale_plan_4090.status"
OUT_FILE="/tmp/scale_plan_4090_loop.out"
if pgrep -af "scale_plan_4090_watchdog.sh" | awk '{print $1}' | grep -v "^$$\$" >/dev/null 2>&1; then
  ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  echo "WATCHDOG already running ${ts}" | tee -a "${STATUS_FILE}"
  exit 0
fi

while true; do
  if pgrep -af "run_scale_plan_4090_loop.sh" >/dev/null 2>&1; then
    sleep 60
    continue
  fi
  ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  echo "WATCHDOG restart loop ${ts}" | tee -a "${STATUS_FILE}"
  setsid -f bash "${LOOP_SCRIPT}" >> "${OUT_FILE}" 2>&1
  sleep 5
done
