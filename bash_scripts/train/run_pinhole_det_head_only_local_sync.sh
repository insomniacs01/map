#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NAS_ROOT="${1:?Usage: $0 <nas_run_root> <resume_ckpt> [local_root]}"
RESUME_CKPT="${2:?Usage: $0 <nas_run_root> <resume_ckpt> [local_root]}"
LOCAL_ROOT="${3:-${REPO_ROOT}/experiments/local_runs/$(date +%Y%m%d_%H%M%S)_pinhole_det_head_only}"

SYNC_SECS="${SYNC_SECS:-300}"
PRINT_FREQ="${PRINT_FREQ:-100}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29640}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU:-32}"
EPOCHS="${EPOCHS:-10}"
DATASET="${DATASET:-opv2v_coop_det_ft_2a8v}"
LOSS_CFG="${LOSS_CFG:-opv2v_det_only_wide_v4}"
DET_HEAD_CFG="${DET_HEAD_CFG:-bev_centernet_wide_v4}"
TRAIN_PARAMS="${TRAIN_PARAMS:-opv2v_det_head_only}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TORCHRUN=/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/torchrun

LOCAL_RUN_DIR="${LOCAL_ROOT}/pinhole_det_head_only"
NAS_RUN_DIR="${NAS_ROOT}/pinhole_det_head_only"

mkdir -p "${LOCAL_RUN_DIR}" "${NAS_RUN_DIR}"

TORCH_HOME_DEFAULT=/J6P-perception/yijinxiong_workspace/.cache/torch
export TORCH_HOME="${TORCH_HOME:-${TORCH_HOME_DEFAULT}}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"

sync_loop() {
  while true; do
    rsync -a --inplace --partial --update \
      --exclude 'events*' \
      --exclude '*.tmp' \
      "${LOCAL_RUN_DIR}/" "${NAS_RUN_DIR}/"
    sleep "${SYNC_SECS}"
  done
}

sync_loop > "${LOCAL_RUN_DIR}/sync.log" 2>&1 &
SYNC_PID=$!

cleanup() {
  kill "${SYNC_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep torch elastic/temp artifacts on workspace storage by default.
WORK_TMP_ROOT="${WORK_TMP_ROOT:-${REPO_ROOT}/experiments/local_tmp}"
mkdir -p "${WORK_TMP_ROOT}" || true
export TMPDIR="${TMPDIR:-${WORK_TMP_ROOT}}"
export TORCHELASTIC_TMPDIR="${TORCHELASTIC_TMPDIR:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}" "${TORCHELASTIC_TMPDIR}" || true

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" PYTHONPATH="${REPO_ROOT}" \
  "${TORCHRUN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  scripts/train.py \
    machine=local_a800 model=mapanything_det dataset="${DATASET}" \
    loss="${LOSS_CFG}" train_params="${TRAIN_PARAMS}" model/det_head="${DET_HEAD_CFG}" \
    train_params.max_num_of_imgs_per_gpu="${MAX_IMGS_PER_GPU}" train_params.accum_iter=1 train_params.eval_freq=1 train_params.epochs="${EPOCHS}" \
    train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
    train_params.print_freq="${PRINT_FREQ}" \
    +train_params.resume_ckpt="${RESUME_CKPT}" \
    dataset.num_workers="${NUM_WORKERS}" \
    hydra.run.dir="${LOCAL_RUN_DIR}" \
    ${EXTRA_ARGS} \
  2>&1 | tee "${LOCAL_RUN_DIR}/launch.log"

# final sync
rsync -a --inplace --partial --update \
  --exclude 'events*' \
  --exclude '*.tmp' \
  "${LOCAL_RUN_DIR}/" "${NAS_RUN_DIR}/"
