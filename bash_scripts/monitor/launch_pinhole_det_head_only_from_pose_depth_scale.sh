#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NAS_RUN_ROOT="${NAS_RUN_ROOT:-${REPO_ROOT}/experiments/long_runs/20260201_pinhole_det_head_only_from_pose_depth_scale_dist30_m32}"
LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/experiments/local_runs/pinhole_det_head_only_dist30_m32}"
GEOM_CKPT="${GEOM_CKPT:-${REPO_ROOT}/experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30/pinhole_pose_depth_scale/checkpoint-best.pth}"
RESUME_CKPT="${RESUME_CKPT:-${GEOM_CKPT}}"

WORK_TMP_ROOT="${WORK_TMP_ROOT:-${REPO_ROOT}/experiments/local_tmp}"
mkdir -p "${WORK_TMP_ROOT}" || true
export TMPDIR="${TMPDIR:-${WORK_TMP_ROOT}}"
export TORCHELASTIC_TMPDIR="${TORCHELASTIC_TMPDIR:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}" "${TORCHELASTIC_TMPDIR}" || true

SYNC_SECS="${SYNC_SECS:-300}"
PRINT_FREQ="${PRINT_FREQ:-100}"
NPROC="${NPROC:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU:-32}"
EPOCHS="${EPOCHS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-+dataset.coop_max_agent_distance=32}"

mkdir -p "${LOCAL_ROOT}"

export NAS_RUN_ROOT LOCAL_ROOT RESUME_CKPT SYNC_SECS PRINT_FREQ NPROC NUM_WORKERS MAX_IMGS_PER_GPU EPOCHS EXTRA_ARGS

bash "${REPO_ROOT}/bash_scripts/train/run_pinhole_det_head_only_local_sync.sh" \
  "${NAS_RUN_ROOT}" "${RESUME_CKPT}" "${LOCAL_ROOT}"
