#!/bin/bash
set -euo pipefail

if [ "${MA_ALLOW_LEGACY_SCALE_PLAN_4090:-0}" != "1" ]; then
  echo "[DEPRECATED] ${0##*/} is legacy and disabled by default (see scripts/run_scale_plan_4090.sh)." >&2
  echo "To run anyway (NOT recommended): export MA_ALLOW_LEGACY_SCALE_PLAN_4090=1" >&2
  exit 2
fi

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PLAN_SCRIPT="${REPO_ROOT}/scripts/run_scale_plan_4090.sh"
STATUS_FILE="/tmp/scale_plan_4090.status"

trap 'ts=$(date -u "+%Y-%m-%d %H:%M:%S UTC"); echo "LOOP_EXIT ${ts}" | tee -a "${STATUS_FILE}"' EXIT

while true; do
  if pgrep -af "scripts/train.py" >/dev/null 2>&1; then
    ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    echo "TRAINING_ACTIVE ${ts}; sleep 300s" | tee -a "${STATUS_FILE}"
    sleep 300
    continue
  fi
  ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  echo "=== loop start ${ts} ===" | tee -a "${STATUS_FILE}"
  set +e
  bash "${PLAN_SCRIPT}"
  rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    echo "PASS at ${ts}" | tee -a "${STATUS_FILE}"
    exit 0
  fi
  echo "FAIL rc=${rc} at ${ts}; sleep 300s" | tee -a "${STATUS_FILE}"
  sleep 300
done
