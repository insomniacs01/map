#!/bin/bash
set -euo pipefail

# Run the baseline-distance bucket suite on the mainline geometry checkpoints,
# then generate a consolidated comparison report.
#
# Usage:
#   bash map-anything/scripts/run_geom_bucket_suite_mainlines_20260301.sh [cuda_device=0]
#
# Overrides (env):
#   SUITE_ROOT: where to write outputs (default: map-anything/eval_runs/geom_bucket_suite_mainlines_20260301)
#   CONTRACTS: override contracts used by eval_geom_ckpt_bucket_suite.sh
#   EVAL_MODES / MODEL_TASK / BUCKET_EDGES: passed through to underlying scripts

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

CUDA_DEV="${1:-0}"
SUITE_ROOT="${SUITE_ROOT:-${REPO_ROOT}/eval_runs/geom_bucket_suite_mainlines_20260301}"

mkdir -p "${SUITE_ROOT}/logs"

NAMES=(
  sfix_direct
  nm4_base
  promoted_v2
  mid30_60_detach_pose12
  vggt_e30_best
)

CKPTS=(
  "${REPO_ROOT}/experiments/local_runs/a800_sfix_direct/pinhole_pose_depth_scale/checkpoint-best.pth"
  "${REPO_ROOT}/experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth"
  "${REPO_ROOT}/experiments/local_runs/20260226_srecover_near_from_midpose12_lr1e5/pinhole_pose_depth_scale/checkpoint-best.pth"
  "${REPO_ROOT}/experiments/local_runs/20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/pinhole_pose_depth_scale/checkpoint-best.pth"
  "${REPO_ROOT}/experiments/vggt/training/20260224_plan2_coop_metric_e30_full/checkpoint-best.pth"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  ckpt="${CKPTS[$i]}"
  out_root="${SUITE_ROOT}/${name}"
  log="${SUITE_ROOT}/logs/${name}.log"
  model_arch="mapanything"
  if [[ "${name}" == vggt* ]] || [[ "${ckpt}" == *"/experiments/vggt/"* ]]; then
    model_arch="vggt"
  fi

  echo "[RUN] name=${name}"
  echo "[RUN] ckpt=${ckpt}"
  echo "[RUN] out_root=${out_root}"
  echo "[RUN] cuda=${CUDA_DEV}"
  echo "[RUN] model_arch=${model_arch}"

  MODEL_ARCH="${model_arch}" \
  bash "${REPO_ROOT}/scripts/eval_geom_ckpt_bucket_suite.sh" \
    "${ckpt}" \
    "${out_root}" \
    "${CUDA_DEV}" \
    2>&1 | tee "${log}"
done

echo "[COMPARE] suite_root=${SUITE_ROOT}"
"${PYTHON}" "${REPO_ROOT}/scripts/compare_geom_bucket_suite.py" \
  --suite_root "${SUITE_ROOT}" \
  --out_md "${SUITE_ROOT}/compare.md"

echo "[DONE] ${SUITE_ROOT}"
