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
LOG_FILE="${LOG_ROOT}/entry_${TS}_${HOST_TAG}_rank${MLP_ROLE_INDEX:-NA}.log"

touch "${LOG_FILE}"
# Duplicate original stdout/stderr, then stream log back to console via tail.
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
id || true
hostname || true
ulimit -a || true

echo "=== env ==="
env | sort

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
  if [ -z "${MLP_WORKER_GPU:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      # Count visible devices using shell only (no python dependency).
      _cuda_count=0
      IFS=',' read -r -a _cuda_devs <<< "${CUDA_VISIBLE_DEVICES}"
      for _d in "${_cuda_devs[@]}"; do
        _d="${_d//[[:space:]]/}"
        if [ -n "${_d}" ]; then
          _cuda_count=$((_cuda_count + 1))
        fi
      done
      if [ "${_cuda_count}" -le 0 ]; then
        _cuda_count=1
      fi
      MLP_WORKER_GPU="${_cuda_count}"
    else
      MLP_WORKER_GPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
      if [ -z "${MLP_WORKER_GPU}" ] || [ "${MLP_WORKER_GPU}" = "0" ]; then
        MLP_WORKER_GPU=1
      fi
    fi
  export MLP_WORKER_GPU
  fi
  export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
  export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
  export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
  # Pick a free local port for rendezvous to avoid collisions in debug runs.
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

echo "=== disk ==="
df -h || true
ls -la / || true
ls -la /J6P-perception 2>/dev/null || true
ls -la /J6P-perception/yijinxiong_workspace 2>/dev/null || true
ls -la /mlp-vepfs 2>/dev/null || true
ls -la /workspace 2>/dev/null || true
ls -la /data 2>/dev/null || true

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
  echo "HINT: ensure NAS is mounted to /J6P-perception/yijinxiong_workspace (preferred)."
  echo "HINT: or update entry command to use the actual mount path."
  find / -maxdepth 4 -type d -name map-anything 2>/dev/null | head -n 10 || true
  exit 3
fi
echo "REPO_ROOT=${REPO_ROOT}"

echo "=== nvidia-smi ==="
nvidia-smi || true

PYTHON="${PYTHON:-/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  echo "WARN: python venv not found at ${PYTHON}; fallback to python"
  PYTHON="python"
fi

"${PYTHON}" - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print("device count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed", repr(e))
    raise
PY

cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/J6P-perception/yijinxiong_workspace/.cache/torch}"
mkdir -p "${TORCH_HOME}" || true

RUN_ROOT="${PWD}/experiments/long_runs/$(date +%Y%m%d_%H%M%S)_cyl_a800_8g"
mkdir -p "${RUN_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"

if [ "${MLP_DEBUG_ONLY:-0}" = "1" ]; then
  echo "MLP_DEBUG_ONLY=1: preflight checks complete, skip training launch."
  exit 0
fi

EXTRA_ARGS=()
if [ "${SMOKE_TEST:-0}" = "1" ]; then
  echo "SMOKE_TEST=1: applying short-run overrides"
  EXTRA_ARGS+=("train_params.epochs=1")
  EXTRA_ARGS+=("train_params.eval_freq=0")
  EXTRA_ARGS+=("train_params.print_freq=10")
  EXTRA_ARGS+=("train_params.resume=false")
  EXTRA_ARGS+=("train_params.max_num_of_imgs_per_gpu=2")
  EXTRA_ARGS+=("dataset.num_workers=2")
  EXTRA_ARGS+=("dataset.opv2v.train.max_scenes=20")
  EXTRA_ARGS+=("dataset.opv2v.val.max_scenes=20")
  EXTRA_ARGS+=("dataset.opv2v.test.max_scenes=20")
  EXTRA_ARGS+=("dataset.opv2v.train.max_agent_distance=null")
  EXTRA_ARGS+=("dataset.opv2v.val.max_agent_distance=null")
  EXTRA_ARGS+=("dataset.opv2v.test.max_agent_distance=null")
fi
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
  model=mapanything_det \
  dataset=opv2v_cyl_coop_det_ft_r60_nearest \
  loss=opv2v_det_e2e_vehicle_aux_v1 \
  train_params=opv2v_cyl_det_e2e_a800_2gpu \
  model/det_head=bev_centernet_wide_e2e_v1 \
  train_params.max_num_of_imgs_per_gpu=8 \
  train_params.accum_iter=1 \
  train_params.eval_freq=0 \
  train_params.epochs=20 \
  train_params.disable_cudnn_benchmark=false \
  train_params.disable_tensorboard=true \
  train_params.resume=true \
  +train_params.resume_ckpt=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/experiments/long_runs/20260126_002000_accel_split/cyl_e2e_fullres/checkpoint-last.pth \
  dataset.num_workers=4 \
  "${EXTRA_ARGS[@]}" \
  hydra.run.dir="${RUN_ROOT}"
