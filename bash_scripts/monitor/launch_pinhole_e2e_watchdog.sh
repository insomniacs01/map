#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_pinhole_e2e_wd}"
NAS_RUN_ROOT="${NAS_RUN_ROOT:-${REPO_ROOT}/experiments/long_runs/${RUN_TAG}}"
LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/experiments/local_runs/${RUN_TAG}}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/pinhole_e2e_fullres"
NAS_RUN_DIR="${NAS_RUN_ROOT}/pinhole_e2e_fullres"

mkdir -p "${LOCAL_RUN_DIR}" "${NAS_RUN_DIR}"

LATEST_LINK="${REPO_ROOT}/experiments/long_runs/_latest_pinhole_e2e"
ln -sfn "${NAS_RUN_ROOT}" "${LATEST_LINK}"

if [ -z "${MLP_WORKER_GPU:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    MLP_WORKER_GPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    MLP_WORKER_GPU="1"
  fi
fi
export MLP_WORKER_GPU

GPU_MEM_MB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
fi

export CODEX_PROJECT_TAG="${CODEX_PROJECT_TAG:-map-anything}"
export CODEX_TASK_TAG="${CODEX_TASK_TAG:-pinhole_e2e_wd}"
export WATCH_LOG="${WATCH_LOG:-${LOCAL_RUN_DIR}/launch.log}"
export WATCH_IDLE_SEC="${WATCH_IDLE_SEC:-1800}"
export WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-60}"
export WATCH_IDLE_UTIL="${WATCH_IDLE_UTIL:-5}"
export FAIL_REGEX="${FAIL_REGEX:-Bad loss=nan|loss: nan|nan|Reached max_bad_loss_count}"

MAX_RETRIES="${MAX_RETRIES:-3}"
SLEEP_SEC="${SLEEP_SEC:-120}"

DEFAULT_DET="${REPO_ROOT}/experiments/long_runs/_latest_pinhole_det_head_only/pinhole_det_head_only/checkpoint-best.pth"
DEFAULT_GEOM="${REPO_ROOT}/experiments/long_runs/_latest_pinhole_pose_depth_scale_recover/pinhole_pose_depth_scale/checkpoint-best.pth"
if [ -z "${RESUME_CKPT:-}" ] && [ -f "${DEFAULT_DET}" ]; then
  RESUME_CKPT="${DEFAULT_DET}"
fi
if [ -z "${RESUME_CKPT:-}" ] && [ -f "${DEFAULT_GEOM}" ]; then
  RESUME_CKPT="${DEFAULT_GEOM}"
fi
RESUME_CKPT="${RESUME_CKPT:-${REPO_ROOT}/experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30/pinhole_pose_depth_scale/checkpoint-best.pth}"

NPROC="${NPROC:-${MLP_WORKER_GPU}}"
if [ -z "${MAX_IMGS_PER_GPU:-}" ]; then
  if [ -n "${GPU_MEM_MB}" ] && [ "${GPU_MEM_MB}" -ge 60000 ]; then
    MAX_IMGS_PER_GPU=24
  elif [ -n "${GPU_MEM_MB}" ] && [ "${GPU_MEM_MB}" -ge 40000 ]; then
    MAX_IMGS_PER_GPU=16
  else
    MAX_IMGS_PER_GPU=8
  fi
fi
if [ -z "${NUM_WORKERS:-}" ]; then
  cpu_cnt="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"
  per_gpu=$((cpu_cnt / (MLP_WORKER_GPU > 0 ? MLP_WORKER_GPU : 1)))
  if [ "${per_gpu}" -lt 4 ]; then per_gpu=4; fi
  if [ "${per_gpu}" -gt 16 ]; then per_gpu=16; fi
  NUM_WORKERS="${per_gpu}"
fi
EPOCHS="${EPOCHS:-20}"
SYNC_SECS="${SYNC_SECS:-300}"
PRINT_FREQ="${PRINT_FREQ:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-+dataset.coop_max_agent_distance=32}"

RUN_WITH_RETRY="${SCRIPT_DIR}/run_with_retry.sh"
ENTRY_SCRIPT="${REPO_ROOT}/bash_scripts/train/run_pinhole_e2e_local_sync.sh"

exec "${RUN_WITH_RETRY}" "${MAX_RETRIES}" "${SLEEP_SEC}" -- env \
  NAS_RUN_ROOT="${NAS_RUN_ROOT}" \
  LOCAL_ROOT="${LOCAL_ROOT}" \
  RESUME_CKPT="${RESUME_CKPT}" \
  NPROC="${NPROC}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU}" \
  EPOCHS="${EPOCHS}" \
  SYNC_SECS="${SYNC_SECS}" \
  PRINT_FREQ="${PRINT_FREQ}" \
  EXTRA_ARGS="${EXTRA_ARGS}" \
  "${ENTRY_SCRIPT}" "${NAS_RUN_ROOT}" "${RESUME_CKPT}" "${LOCAL_ROOT}"
