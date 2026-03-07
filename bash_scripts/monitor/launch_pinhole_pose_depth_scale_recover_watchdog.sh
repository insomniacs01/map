#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_pinhole_pose_depth_scale_recover_wd}"
NAS_RUN_ROOT="${NAS_RUN_ROOT:-${REPO_ROOT}/experiments/long_runs/${RUN_TAG}}"
LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/experiments/local_runs/${RUN_TAG}}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/pinhole_pose_depth_scale"
NAS_RUN_DIR="${NAS_RUN_ROOT}/pinhole_pose_depth_scale"

mkdir -p "${LOCAL_RUN_DIR}" "${NAS_RUN_DIR}"

LATEST_LINK="${REPO_ROOT}/experiments/long_runs/_latest_pinhole_pose_depth_scale_recover"
ln -sfn "${NAS_RUN_ROOT}" "${LATEST_LINK}"

if [ -z "${MLP_WORKER_GPU:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    MLP_WORKER_GPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    MLP_WORKER_GPU="1"
  fi
fi
export MLP_WORKER_GPU
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-29655}"

GPU_MEM_MB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
fi

export CODEX_PROJECT_TAG="${CODEX_PROJECT_TAG:-map-anything}"
export CODEX_TASK_TAG="${CODEX_TASK_TAG:-pinhole_pose_depth_scale_recover_wd}"
export WATCH_LOG="${WATCH_LOG:-${LOCAL_RUN_DIR}/train.log}"
export WATCH_IDLE_SEC="${WATCH_IDLE_SEC:-600}"
export WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-60}"
export WATCH_IDLE_UTIL="${WATCH_IDLE_UTIL:-5}"
export FAIL_REGEX="${FAIL_REGEX:-Reached max_bad_loss_count|RuntimeError|CUDA out of memory|CUBLAS|NCCL|Traceback}"

MAX_RETRIES="${MAX_RETRIES:-3}"
SLEEP_SEC="${SLEEP_SEC:-120}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO_ROOT}/experiments/long_runs/20260202_geom_stage2_pose_depth_scale_strong/pinhole_pose_depth_scale/checkpoint-best.pth}"
if [ -z "${MAX_IMGS_PER_GPU:-}" ]; then
  if [ -n "${GPU_MEM_MB}" ] && [ "${GPU_MEM_MB}" -ge 60000 ]; then
    MAX_IMGS_PER_GPU=32
  elif [ -n "${GPU_MEM_MB}" ] && [ "${GPU_MEM_MB}" -ge 40000 ]; then
    MAX_IMGS_PER_GPU=24
  else
    MAX_IMGS_PER_GPU=16
  fi
fi
if [ -z "${NUM_WORKERS:-}" ]; then
  cpu_cnt="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"
  per_gpu=$((cpu_cnt / (MLP_WORKER_GPU > 0 ? MLP_WORKER_GPU : 1)))
  if [ "${per_gpu}" -lt 4 ]; then per_gpu=4; fi
  if [ "${per_gpu}" -gt 8 ]; then per_gpu=8; fi
  NUM_WORKERS="${per_gpu}"
fi
EPOCHS="${EPOCHS:-6}"
SYNC_SECS="${SYNC_SECS:-120}"
LOSS_NAME="${LOSS_NAME:-opv2v_vggt_pose_scale_loss_recover}"
MLP_ENTRY_EXTRA_ARGS="${MLP_ENTRY_EXTRA_ARGS:-++dataset.coop_max_agent_distance=32 ++train_params.print_freq=10}"

RUN_WITH_RETRY="${SCRIPT_DIR}/run_with_retry.sh"
ENTRY_SCRIPT="${REPO_ROOT}/bash_scripts/train/mlp_entry_pinhole_pose_depth_scale_recover.sh"

exec "${RUN_WITH_RETRY}" "${MAX_RETRIES}" "${SLEEP_SEC}" -- env \
  NAS_RUN_ROOT="${NAS_RUN_ROOT}" \
  LOCAL_ROOT="${LOCAL_ROOT}" \
  PRETRAINED_CKPT="${PRETRAINED_CKPT}" \
  MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${EPOCHS}" \
  SYNC_SECS="${SYNC_SECS}" \
  LOSS_NAME="${LOSS_NAME}" \
  MLP_ENTRY_EXTRA_ARGS="${MLP_ENTRY_EXTRA_ARGS}" \
  "${ENTRY_SCRIPT}"
