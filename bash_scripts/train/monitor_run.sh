#!/bin/bash

# Lightweight monitor for MapAnything training runs.
#
# Usage:
#   bash map-anything/bash_scripts/train/monitor_run.sh <run_dir> [interval_sec]
#   nohup bash map-anything/bash_scripts/train/monitor_run.sh <run_dir> 60 > <run_dir>/monitor.log 2>&1 &

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_dir> [interval_sec]" >&2
  exit 1
fi

RUN_DIR="$1"
INTERVAL="${2:-60}"

RUN_DIR="$(readlink -f "$RUN_DIR")"
LOG="${RUN_DIR}/train.log"
JSON_LOG="${RUN_DIR}/log.txt"
PID_FILE="${RUN_DIR}/nohup.pid"

if [ ! -d "$RUN_DIR" ]; then
  echo "Error: run_dir not found: ${RUN_DIR}" >&2
  exit 2
fi

if command -v rg >/dev/null 2>&1; then
  GREP_CMD=(rg -n)
else
  GREP_CMD=(grep -nE)
fi

echo "[monitor] run_dir=${RUN_DIR}"
echo "[monitor] interval_sec=${INTERVAL}"
echo "[monitor] log=${LOG}"
echo "[monitor] json_log=${JSON_LOG}"
echo

while true; do
  TS="$(date '+%F %T')"
  echo "=== ${TS} ==="

  if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${PID:-}" ]; then
      ps -o pid,ppid,%cpu,%mem,etime,cmd -p "$PID" || echo "[monitor] PID ${PID} not running"
    fi
  fi

  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true

  if [ -f "$LOG" ]; then
    echo "--- train (latest) ---"
    "${GREP_CMD[@]}" "Epoch:\\s*\\[" "$LOG" | tail -n 1 || true
    echo "--- eval (latest) ---"
    "${GREP_CMD[@]}" "Testing on|Test Epoch" "$LOG" | tail -n 5 || true
    echo "--- errors (latest) ---"
    "${GREP_CMD[@]}" "(ERROR|Traceback|CUDA|RuntimeError|invalid argument)" "$LOG" | tail -n 3 || true
  fi

  if [ -f "$JSON_LOG" ]; then
    echo "--- log.txt (latest) ---"
    tail -n 1 "$JSON_LOG" || true
  fi

  echo
  sleep "$INTERVAL"
done

