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
  export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-29631}"
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

NAS_RUN_ROOT="${NAS_RUN_ROOT:-${REPO_ROOT}/experiments/long_runs/$(date +%Y%m%d_%H%M%S)_pinhole_e2e_a800_8g}"
LOCAL_ROOT="${LOCAL_ROOT:-/tmp/mapanything_runs/pinhole_e2e_a800_8g}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/pinhole_e2e_fullres"
NAS_RUN_DIR="${NAS_RUN_ROOT}/pinhole_e2e_fullres"
mkdir -p "${LOCAL_RUN_DIR}" "${NAS_RUN_DIR}"

SYNC_SECS="${SYNC_SECS:-300}"
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

MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU:-48}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRINT_FREQ="${PRINT_FREQ:-200}"
EPOCHS="${EPOCHS:-20}"
RESUME_CKPT="${RESUME_CKPT:-/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/experiments/long_runs/20260125_235813_pinhole_split/pinhole_e2e_fullres/checkpoint-last.pth}"
if [ ! -f "${RESUME_CKPT}" ]; then
  # Auto-pick the newest pinhole checkpoint if the provided one is missing.
  _auto_ckpt="$(ls -t "${REPO_ROOT}"/experiments/long_runs/*pinhole*/pinhole_e2e_fullres/checkpoint-last.pth 2>/dev/null | head -n 1 || true)"
  if [ -n "${_auto_ckpt}" ]; then
    RESUME_CKPT="${_auto_ckpt}"
  fi
fi
echo "RESUME_CKPT=${RESUME_CKPT}"

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
  model=mapanything_det \
  dataset=opv2v_coop_det_ft_2a8v \
  loss=opv2v_det_e2e_vehicle_aux_v1 \
  train_params=opv2v_det_e2e_ft \
  model/det_head=bev_centernet_wide_e2e_v1 \
  train_params.max_num_of_imgs_per_gpu="${MAX_IMGS_PER_GPU}" \
  train_params.accum_iter=1 \
  train_params.eval_freq=0 \
  train_params.epochs="${EPOCHS}" \
  train_params.disable_cudnn_benchmark=false \
  train_params.disable_tensorboard=true \
  train_params.resume=true \
  train_params.print_freq="${PRINT_FREQ}" \
  +train_params.resume_ckpt="${RESUME_CKPT}" \
  dataset.num_workers="${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}" \
  hydra.run.dir="${LOCAL_RUN_DIR}" \
  2>&1 | tee "${LOCAL_RUN_DIR}/launch.log"
