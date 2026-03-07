#!/bin/bash

set -euo pipefail

LOG_ROOT="/J6P-perception/yijinxiong_workspace/log/mlp_entry"
if ! mkdir -p "${LOG_ROOT}" 2>/dev/null; then
  LOG_ROOT="/mlp-vepfs/log/mlp_entry"
  if ! mkdir -p "${LOG_ROOT}" 2>/dev/null; then
    LOG_ROOT="/tmp/mlp_entry"
    mkdir -p "${LOG_ROOT}"
  fi
fi
TS="$(date +%Y%m%d_%H%M%S)"
HOST_TAG="$(hostname 2>/dev/null || echo unknown)"
PID_TAG="$$"
LOG_FILE="${LOG_ROOT}/entry_${TS}_${HOST_TAG}_pid${PID_TAG}_rank${MLP_ROLE_INDEX:-NA}.log"

touch "${LOG_FILE}"
exec 3>&1 4>&2
exec >> "${LOG_FILE}" 2>&1
tail -F "${LOG_FILE}" >&3 &
TAIL_PID=$!

trap 'echo "[ERR] line=${LINENO} status=$?"' ERR
trap 'kill "${TAIL_PID}" 2>/dev/null || true; echo "[EXIT] status=$?"' EXIT

echo "LOG_FILE=${LOG_FILE}"
echo "=== entry start ==="
date
echo "pwd=$(pwd)"
hostname || true

echo "=== MLP env ==="
for v in MLP_WORKER_GPU MLP_WORKER_NUM MLP_ROLE_INDEX MLP_WORKER_0_HOST MLP_WORKER_0_PORT; do
  echo "${v}=${!v-}"
done

missing=0
for v in MLP_WORKER_GPU MLP_WORKER_NUM MLP_ROLE_INDEX MLP_WORKER_0_HOST MLP_WORKER_0_PORT; do
  if [ -z "${!v-}" ]; then
    echo "ERROR: missing ${v}"
    missing=1
  fi
done
if [ "${MLP_DEBUG_ONLY:-0}" = "1" ]; then
  echo "MLP_DEBUG_ONLY=1: skip strict MLP env checks"
  missing=0
fi
if [ "${missing}" -ne 0 ]; then
  echo "WARN: MLP env missing; falling back to local single-node defaults."
  export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
  export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
  export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
  export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
  # Pick a free local port for rendezvous to avoid collisions.
  if [ -z "${MLP_WORKER_0_PORT:-}" ]; then
    if command -v ss >/dev/null 2>&1; then
      for p in $(seq 29500 29650); do
        if ! ss -ltn | awk '{print $4}' | grep -q ":${p}$"; then
          MLP_WORKER_0_PORT="${p}"
          break
        fi
      done
    fi
    if [ -z "${MLP_WORKER_0_PORT:-}" ]; then
      # Fallback: random high port (reduce collision risk).
      MLP_WORKER_0_PORT=$((20000 + RANDOM % 20000))
    fi
  else
    if command -v ss >/dev/null 2>&1; then
      if ss -ltn | awk '{print $4}' | grep -q ":${MLP_WORKER_0_PORT}$"; then
        echo "WARN: port ${MLP_WORKER_0_PORT} already in use."
      fi
    fi
  fi
  export MLP_WORKER_0_PORT
  echo "FALLBACK: MLP_WORKER_GPU=${MLP_WORKER_GPU} MLP_WORKER_NUM=${MLP_WORKER_NUM} MLP_ROLE_INDEX=${MLP_ROLE_INDEX} MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST} MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT}"
fi

REPO_ROOT="${REPO_ROOT:-}"
for base in \
  /J6P-perception/yijinxiong_workspace \
  /mlp-vepfs/J6P-perception/yijinxiong_workspace \
  /workspace/J6P-perception/yijinxiong_workspace \
  /data/J6P-perception/yijinxiong_workspace \
  /J6P-perception \
  /mlp-vepfs \
  /workspace \
  /data; do
  if [ -d "${base}/vggt_series_4_coop/map-anything" ]; then
    REPO_ROOT="${base}/vggt_series_4_coop/map-anything"
    break
  fi
done

if [ -z "${REPO_ROOT}" ]; then
  echo "ERROR: repo not found under common mount points."
  exit 3
fi
echo "REPO_ROOT=${REPO_ROOT}"

PYTHON="${PYTHON:-/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="python"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="${TORCH_HOME:-/J6P-perception/yijinxiong_workspace/.cache/torch}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
mkdir -p "${TORCH_HOME}" || true

# Keep temporary artifacts on a short path by default; long paths can trigger
# "AF_UNIX path too long" inside Python multiprocessing resource sharer.
WORK_TMP_ROOT="${WORK_TMP_ROOT:-/tmp/ma_tmp}"
if [ "${#WORK_TMP_ROOT}" -gt 80 ]; then
  echo "WARN: WORK_TMP_ROOT path is too long (${WORK_TMP_ROOT}); fallback to /tmp/ma_tmp"
  WORK_TMP_ROOT="/tmp/ma_tmp"
fi
mkdir -p "${WORK_TMP_ROOT}" || true
export TMPDIR="${TMPDIR:-${WORK_TMP_ROOT}}"
export TORCHELASTIC_TMPDIR="${TORCHELASTIC_TMPDIR:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}" "${TORCHELASTIC_TMPDIR}" || true

export MAPANYTHING_CACHE_DIR="${MAPANYTHING_CACHE_DIR:-${REPO_ROOT}/cache/opv2v_scene_cache}"
mkdir -p "${MAPANYTHING_CACHE_DIR}" || true
ln -sfn "${MAPANYTHING_CACHE_DIR}" "${TMPDIR%/}/mapanything_cache" 2>/dev/null || true

NAS_RUN_ROOT="${NAS_RUN_ROOT:-${REPO_ROOT}/experiments/long_runs/$(date +%Y%m%d_%H%M%S)_pinhole_pose_depth_scale_recover}"
LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/experiments/local_runs/pinhole_pose_depth_scale_recover}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/pinhole_pose_depth_scale"
NAS_RUN_DIR="${NAS_RUN_ROOT}/pinhole_pose_depth_scale"
mkdir -p "${LOCAL_RUN_DIR}" "${NAS_RUN_DIR}"

# Pre-fetch DINOv2 weights to avoid long torch.hub stalls during startup.
if [ "${DINO_PREFETCH:-1}" = "1" ]; then
  DINO_PREFETCH_MODEL="${DINO_PREFETCH_MODEL:-dinov2_vitl14}"
  DINO_PREFETCH_TIMEOUT="${DINO_PREFETCH_TIMEOUT:-900}"
  DINO_PREFETCH_HEARTBEAT_SEC="${DINO_PREFETCH_HEARTBEAT_SEC:-60}"
  export DINO_PREFETCH_MODEL DINO_PREFETCH_TIMEOUT
  touch "${LOCAL_RUN_DIR}/launch.log"
  {
    start_ts="$(date +%s)"
    while true; do
      now_ts="$(date +%s)"
      if [ $((now_ts - start_ts)) -ge "${DINO_PREFETCH_TIMEOUT}" ]; then
        break
      fi
      touch "${LOCAL_RUN_DIR}/launch.log"
      sleep "${DINO_PREFETCH_HEARTBEAT_SEC}"
    done
  } &
  PREFETCH_TOUCH_PID=$!
  set +e
  "${PYTHON}" - <<'PY'
import os
import signal
import sys

timeout = int(os.environ.get("DINO_PREFETCH_TIMEOUT", "900"))
model_name = os.environ.get("DINO_PREFETCH_MODEL", "dinov2_vitl14")

def _timeout_handler(signum, frame):
    raise TimeoutError(f"DINOv2 prefetch timed out after {timeout}s")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(timeout)
try:
    import torch
    try:
        from mapanything.utils.hf_utils.hf_helpers import _patch_torch_hub_load_for_offline_dinov2
        _patch_torch_hub_load_for_offline_dinov2()
    except Exception:
        pass
    torch.hub.load("facebookresearch/dinov2", model_name)
    print(f"[prefetch] torch.hub loaded {model_name}")
except Exception as exc:
    print(f"[prefetch] torch.hub load failed: {exc}", file=sys.stderr)
    raise
finally:
    signal.alarm(0)
PY
  PREFETCH_STATUS=$?
  set -e
  kill "${PREFETCH_TOUCH_PID}" 2>/dev/null || true
  wait "${PREFETCH_TOUCH_PID}" 2>/dev/null || true
  if [ "${PREFETCH_STATUS}" -ne 0 ]; then
    exit "${PREFETCH_STATUS}"
  fi
fi

SYNC_SECS="${SYNC_SECS:-120}"
sync_loop() {
  while true; do
    "${PYTHON}" - <<'PY'
import os, shutil
local_dir = os.environ.get("LOCAL_RUN_DIR")
nas_dir = os.environ.get("NAS_RUN_DIR")
if not local_dir or not nas_dir:
    raise SystemExit("LOCAL_RUN_DIR/NAS_RUN_DIR not set")
def should_copy(src, dst):
    if not os.path.exists(dst):
        return True
    s = os.stat(src); d = os.stat(dst)
    return (s.st_mtime > d.st_mtime) or (s.st_size != d.st_size)
for root, dirs, files in os.walk(local_dir):
    rel = os.path.relpath(root, local_dir)
    dst_root = nas_dir if rel == "." else os.path.join(nas_dir, rel)
    os.makedirs(dst_root, exist_ok=True)
    for fname in files:
        if fname.startswith("events") or fname.endswith(".tmp"):
            continue
        src = os.path.join(root, fname)
        dst = os.path.join(dst_root, fname)
        if should_copy(src, dst):
            shutil.copy2(src, dst)
PY
    sleep "${SYNC_SECS}"
  done
}

export LOCAL_RUN_DIR NAS_RUN_DIR
sync_loop > "${LOCAL_RUN_DIR}/sync.log" 2>&1 &
SYNC_PID=$!
trap 'kill "${SYNC_PID}" 2>/dev/null || true' EXIT

MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU:-32}"
# 24GB cards OOM at 8 imgs with this config; cap to 6 unless user lowers it.
GPU_MEM_MB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  set +e
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print int($1)}' || true)"
  set -e
fi
if [ -n "${GPU_MEM_MB}" ] && [ "${GPU_MEM_MB}" -le 24576 ] && [ "${MAX_IMGS_PER_GPU}" -gt 6 ]; then
  echo "WARN: GPU mem ${GPU_MEM_MB} MiB detected; capping MAX_IMGS_PER_GPU to 6 to avoid OOM."
  MAX_IMGS_PER_GPU=6
fi
# Cap to avoid OOM in DINOv2-large with 8-view 448px inputs.
if [ "${MAX_IMGS_PER_GPU}" -gt 32 ]; then
  echo "WARN: MAX_IMGS_PER_GPU=${MAX_IMGS_PER_GPU} too high for 8-view DINOv2-large; capping to 32."
  MAX_IMGS_PER_GPU=32
fi
NUM_WORKERS="${NUM_WORKERS:-16}"
PRINT_FREQ="${PRINT_FREQ:-50}"
EPOCHS="${EPOCHS:-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-2e-6}"
RESUME="${RESUME:-true}"
MAX_BAD_LOSS_COUNT="${MAX_BAD_LOSS_COUNT:-500}"
BAD_LOSS_BACKOFF_FACTOR="${BAD_LOSS_BACKOFF_FACTOR:-0.5}"
BAD_LOSS_BACKOFF_MIN_LR="${BAD_LOSS_BACKOFF_MIN_LR:-1e-7}"
BAD_LOSS_BACKOFF_MAX="${BAD_LOSS_BACKOFF_MAX:-20}"
BAD_LOSS_DISABLE_AMP="${BAD_LOSS_DISABLE_AMP:-true}"
BAD_LOSS_RESET_OPTIM="${BAD_LOSS_RESET_OPTIM:-true}"
LOSS_NAME="${LOSS_NAME:-opv2v_vggt_pose_scale_loss_recover}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30_m32_cache/pinhole_pose_depth_scale/checkpoint-best.pth}"
INFO_SHARING_GC="${INFO_SHARING_GC:-true}"
PRED_HEAD_GC="${PRED_HEAD_GC:-true}"
ENCODER_GC="${ENCODER_GC:-true}"
FULL_EVAL_FREQ="${FULL_EVAL_FREQ:-5}"
FREEZE_VAL_SAMPLES="${FREEZE_VAL_SAMPLES:-true}"

EXTRA_ARGS=()
if [ -n "${MLP_ENTRY_EXTRA_ARGS:-}" ]; then
  read -r -a _extra <<< "${MLP_ENTRY_EXTRA_ARGS}"
  EXTRA_ARGS+=("${_extra[@]}")
fi

set -x
"${PYTHON}" -m torch.distributed.launch \
  --nproc_per_node "${MLP_WORKER_GPU}" \
  --master_addr "${MLP_WORKER_0_HOST}" \
  --node_rank "${MLP_ROLE_INDEX}" \
  --master_port "${MLP_WORKER_0_PORT}" \
  --nnodes "${MLP_WORKER_NUM}" \
  --use_env \
  scripts/train.py \
  machine=local_a800 \
  model=mapanything \
  dataset=opv2v_coop_ft_2a8v_full \
  loss="${LOSS_NAME}" \
  train_params=default \
  train_params.max_num_of_imgs_per_gpu="${MAX_IMGS_PER_GPU}" \
  train_params.accum_iter=1 \
  train_params.eval_freq=1 \
  train_params.full_eval_freq="${FULL_EVAL_FREQ}" \
  train_params.freeze_val_samples_across_all_epochs="${FREEZE_VAL_SAMPLES}" \
  train_params.epochs="${EPOCHS}" \
  train_params.print_freq="${PRINT_FREQ}" \
  train_params.save_freq=1 \
  train_params.warmup_epochs="${WARMUP_EPOCHS}" \
  train_params.lr="${LR}" \
  train_params.min_lr="${MIN_LR}" \
  train_params.max_bad_loss_count="${MAX_BAD_LOSS_COUNT}" \
  ++train_params.bad_loss_backoff=true \
  ++train_params.bad_loss_backoff_factor="${BAD_LOSS_BACKOFF_FACTOR}" \
  ++train_params.bad_loss_backoff_max="${BAD_LOSS_BACKOFF_MAX}" \
  ++train_params.bad_loss_backoff_min_lr="${BAD_LOSS_BACKOFF_MIN_LR}" \
  ++train_params.bad_loss_disable_amp="${BAD_LOSS_DISABLE_AMP}" \
  ++train_params.bad_loss_reset_optim="${BAD_LOSS_RESET_OPTIM}" \
  train_params.resume="${RESUME}" \
  model.encoder.gradient_checkpointing="${ENCODER_GC}" \
  model.info_sharing.module_args.gradient_checkpointing="${INFO_SHARING_GC}" \
  model.pred_head.gradient_checkpointing="${PRED_HEAD_GC}" \
  +model.model_config.pretrained_checkpoint_path="${PRETRAINED_CKPT}" \
  dataset.num_workers="${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}" \
  hydra.run.dir="${LOCAL_RUN_DIR}" \
  2>&1 | tee "${LOCAL_RUN_DIR}/launch.log"
