#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <single_run_dir> [coop_timestamp] [poll_seconds]" >&2
  echo "Example:" >&2
  echo "  bash $0 experiments/vggt/training/opv2v_single_vggt_pose_metric_long_a800_b1acc4/20260114_161359_single 20260114_161359_coop 60" >&2
  exit 2
fi

single_run_dir="$1"
coop_stamp="${2:-$(date +%Y%m%d_%H%M%S)_coop}"
poll_s="${3:-60}"

if [[ "${single_run_dir}" != /* ]]; then
  single_run_dir="${REPO_ROOT}/${single_run_dir}"
fi

echo "[queue] Waiting for single run to finish:"
echo "        single_run_dir=${single_run_dir}"
echo "        coop_stamp=${coop_stamp}"
echo "        poll_s=${poll_s}"

train_log="${single_run_dir}/train.log"
pid_file="${single_run_dir}/nohup.pid"

while true; do
  if [ -f "${single_run_dir}/checkpoint-final.pth" ]; then
    echo "[queue] Found checkpoint-final.pth; proceeding to coop."
    break
  fi

  # If the single run died with a CUDA error, stop waiting.
  if [ -f "${train_log}" ] && rg -q "CUDA error" "${train_log}"; then
    if [ -f "${pid_file}" ]; then
      pid="$(cat "${pid_file}" || true)"
      if [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1; then
        : # still running; keep waiting
      else
        echo "[queue] Detected CUDA error in ${train_log} and process not running; aborting." >&2
        tail -n 60 "${train_log}" >&2 || true
        exit 1
      fi
    else
      echo "[queue] Detected CUDA error in ${train_log}; aborting." >&2
      tail -n 60 "${train_log}" >&2 || true
      exit 1
    fi
  fi

  sleep "${poll_s}"
done

echo "[queue] Launching coop finetune..."
bash scripts/run_opv2v_vggt_finetune_coop.sh "${coop_stamp}"

