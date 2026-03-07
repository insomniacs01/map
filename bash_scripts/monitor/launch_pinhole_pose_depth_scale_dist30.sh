#!/bin/bash

set -euo pipefail

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
WORK_TMP_ROOT="${WORK_TMP_ROOT:-${REPO_ROOT}/experiments/local_tmp}"
mkdir -p "${WORK_TMP_ROOT}" || true

export TMPDIR="${TMPDIR:-${WORK_TMP_ROOT}}"
export TORCHELASTIC_TMPDIR="${TORCHELASTIC_TMPDIR:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}" "${TORCHELASTIC_TMPDIR}" || true

export TORCH_HOME=/J6P-perception/yijinxiong_workspace/.cache/torch
export TORCH_HUB_DIR=/J6P-perception/yijinxiong_workspace/.cache/torch/hub
export TORCH_HUB_DISABLE_DOWNLOAD=1

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600

export MAPANYTHING_CACHE_DIR="${MAPANYTHING_CACHE_DIR:-${REPO_ROOT}/cache/opv2v_scene_cache}"

export MLP_WORKER_GPU=8
export MLP_WORKER_NUM=1
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=127.0.0.1
export MLP_WORKER_0_PORT=29655

export NAS_RUN_ROOT="${NAS_RUN_ROOT:-${REPO_ROOT}/experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30_m32_cache}"
export LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/experiments/local_runs/pinhole_pose_depth_scale_superstrong_dist30_m32_cache}"
export MAX_IMGS_PER_GPU=32
export NUM_WORKERS=8
export EPOCHS=20
export PRINT_FREQ=50
export INFO_SHARING_GC=true
export PRED_HEAD_GC=true
export FULL_EVAL_FREQ=5
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${LOCAL_ROOT}/pinhole_pose_depth_scale/checkpoint-best.pth}"
export MLP_ENTRY_EXTRA_ARGS="dataset.coop_max_agent_distance=32 train_params.resume=false"

# Reset launch log mtime so run_with_retry hang detection doesn't kill immediately.
mkdir -p "${LOCAL_ROOT}/pinhole_pose_depth_scale"
: > "${LOCAL_ROOT}/pinhole_pose_depth_scale/launch.log"

/bin/bash -lc "${REPO_ROOT}/bash_scripts/train/mlp_entry_pinhole_pose_depth_scale_superstrong.sh"
