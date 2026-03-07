#!/bin/bash
set -euo pipefail

# Run the baseline-distance bucket suite on the mainline geometry checkpoints
# using larger, more stable OPV2V test contracts:
#   - Test2000 (representative + stress)
#   - full Test2170
#
# Usage:
#   bash map-anything/scripts/run_geom_bucket_suite_mainlines_large_20260301.sh [cuda_device=0]
#
# Resumable:
# - The underlying eval script defaults `SKIP_BATCH_EVAL_IF_EXISTS=1`, so reruns
#   will skip inference when `summary_test.json` + per-mode CSVs are present.
#
# Overrides (env):
#   SUITE_ROOT: where to write outputs (default: map-anything/eval_runs/geom_bucket_suite_mainlines_large_20260301)
#   CONTRACTS:  override contracts list (space-separated JSON paths)
#   EVAL_MODES / MODEL_TASK / BUCKET_EDGES: passed through to underlying scripts

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

CUDA_DEV="${1:-0}"
SUITE_ROOT="${SUITE_ROOT:-${REPO_ROOT}/eval_runs/geom_bucket_suite_mainlines_large_20260301}"

mkdir -p "${SUITE_ROOT}/logs"

# Default large-contract suite (Test2000 main + stress + full Test2170).
CONTRACTS_DEFAULT="${REPO_ROOT}/eval_runs/frames_test2000_nearest_seed42.json ${REPO_ROOT}/eval_runs/frames_test2000_nearest_stress_seed42.json ${REPO_ROOT}/eval_runs/frames_test2170_nearest_seed42_full.json"
CONTRACTS="${CONTRACTS:-${CONTRACTS_DEFAULT}}"

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
  echo "[RUN] contracts=${CONTRACTS}"

  MODEL_ARCH="${model_arch}" \
  CONTRACTS="${CONTRACTS}" \
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

