#!/bin/bash

set -euo pipefail

ROOT="${1:?Usage: $0 <run_root> [interval_seconds]}"
INTERVAL="${2:-120}"

ROOT="$(cd "${ROOT}" && pwd)"
LOG="${ROOT}/monitor.log"

while true; do
  {
    echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo "-- torchrun --"
    pgrep -af "torchrun" || true
    for f in "${ROOT}"/*/train.log; do
      if [[ -f "${f}" ]]; then
        echo "-- tail ${f} --"
        tail -n 3 "${f}" || true
      fi
    done
    echo ""
  } >> "${LOG}" 2>&1
  sleep "${INTERVAL}"
done
