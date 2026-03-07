#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <ckpt_path> <det_head_cfg> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
DET_HEAD_CFG="$2"
OUT_ROOT="$3"
CUDA_DEV="${4:-0}"

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
CONTRACT_JSON="${CONTRACT_JSON:-${REPO_ROOT}/eval_runs/frames_test2170_nearest_seed42_full.json}"
KEEP_CAMERA_POSES="${KEEP_CAMERA_POSES:-0}"
KEEP_MAIN_AGENT_POSES="${KEEP_MAIN_AGENT_POSES:-1}"
MODEL_TASK="${MODEL_TASK:-calibrated_sfm}"

export CONTRACT_JSON
export KEEP_CAMERA_POSES
export KEEP_MAIN_AGENT_POSES
export MODEL_TASK

bash "${REPO_ROOT}/scripts/eval_det_ckpt_opv2v_tier.sh" "${CKPT_PATH}" "${DET_HEAD_CFG}" final "${OUT_ROOT}" "${CUDA_DEV}"
