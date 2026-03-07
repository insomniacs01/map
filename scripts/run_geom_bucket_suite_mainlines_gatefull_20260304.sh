#!/bin/bash
set -euo pipefail

# Run the baseline-distance bucket suite on the mainline geometry checkpoints
# using the "contract ladder":
#   - quick: Test500 (main + stress)
#   - gate:  Test1000-stress
#   - final: full Test2170
#
# This script writes:
#   - compare.md (cross-model comparison)
#   - fairness_audit.md (strict fairness audit per contract)
#
# Usage:
#   bash map-anything/scripts/run_geom_bucket_suite_mainlines_gatefull_20260304.sh [cuda_device=0]
#
# Resumable:
# - The underlying eval script defaults `SKIP_BATCH_EVAL_IF_EXISTS=1`, so reruns
#   will skip inference when `summary_test.json` + per-mode CSVs are present.
#
# Force re-inference (to unify eval_script_md5 across runs):
#   SKIP_BATCH_EVAL_IF_EXISTS=0 bash map-anything/scripts/run_geom_bucket_suite_mainlines_gatefull_20260304.sh
#
# Overrides (env):
#   SUITE_ROOT: where to write outputs (default: map-anything/eval_runs/geom_bucket_suite_mainlines_gatefull_20260304)
#   CONTRACTS:  override contracts list (space-separated JSON paths)
#   EVAL_MODES / MODEL_TASK / BUCKET_EDGES / DATA_NORM_TYPE / VGGT_*: passed through

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

CUDA_DEV="${1:-0}"
SUITE_ROOT="${SUITE_ROOT:-${REPO_ROOT}/eval_runs/geom_bucket_suite_mainlines_gatefull_20260304}"

mkdir -p "${SUITE_ROOT}/logs"

# Protocol defaults (fairness-critical; may be overridden intentionally via env).
EVAL_MODES="${EVAL_MODES:-coop}"
MODEL_TASK="${MODEL_TASK:-calibrated_sfm}"
EVAL_KEEP_CAMERA_POSES="${EVAL_KEEP_CAMERA_POSES:-0}"
EVAL_KEEP_MAIN_AGENT_POSES="${EVAL_KEEP_MAIN_AGENT_POSES:-0}"

# Default ladder contracts (quick + gate + final).
CONTRACTS_DEFAULT="${REPO_ROOT}/eval_runs/frames_test500_nearest_seed42.json ${REPO_ROOT}/eval_runs/frames_test500_nearest_stress_seed42.json ${REPO_ROOT}/eval_runs/frames_test1000_nearest_stress_seed42.json ${REPO_ROOT}/eval_runs/frames_test2170_nearest_seed42_full.json"
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
  echo "[RUN] protocol: modes=${EVAL_MODES} task=${MODEL_TASK} keep_camera_poses=${EVAL_KEEP_CAMERA_POSES} keep_main_agent_poses=${EVAL_KEEP_MAIN_AGENT_POSES}"

  MODEL_ARCH="${model_arch}" \
  CONTRACTS="${CONTRACTS}" \
  EVAL_MODES="${EVAL_MODES}" \
  MODEL_TASK="${MODEL_TASK}" \
  EVAL_KEEP_CAMERA_POSES="${EVAL_KEEP_CAMERA_POSES}" \
  EVAL_KEEP_MAIN_AGENT_POSES="${EVAL_KEEP_MAIN_AGENT_POSES}" \
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

FAIRNESS_MD="${SUITE_ROOT}/fairness_audit.md"
echo "# Suite Fairness Audit" > "${FAIRNESS_MD}"
echo "" >> "${FAIRNESS_MD}"
echo "- suite_root: \`${SUITE_ROOT}\`" >> "${FAIRNESS_MD}"
echo "" >> "${FAIRNESS_MD}"

audit_rc=0
mapfile -t _contracts_found < <(find "${SUITE_ROOT}" -mindepth 3 -maxdepth 3 -name summary_test.json -print0 | xargs -0 -r -n1 dirname | xargs -r -n1 basename | sort -u)
for contract in "${_contracts_found[@]}"; do
  echo "## Contract: \`${contract}\`" >> "${FAIRNESS_MD}"
  echo "" >> "${FAIRNESS_MD}"

  mapfile -t _summaries < <(find "${SUITE_ROOT}" -mindepth 3 -maxdepth 3 -path "*/${contract}/summary_test.json" | sort)
  if [ "${#_summaries[@]}" -eq 0 ]; then
    echo "_No summaries found._" >> "${FAIRNESS_MD}"
    echo "" >> "${FAIRNESS_MD}"
    continue
  fi
  if [ "${#_summaries[@]}" -ne "${#NAMES[@]}" ]; then
    echo "_Incomplete: found ${#_summaries[@]} / ${#NAMES[@]} model summaries for this contract._" >> "${FAIRNESS_MD}"
    echo "" >> "${FAIRNESS_MD}"
    audit_rc=1
  fi

  set +e
  "${PYTHON}" "${REPO_ROOT}/scripts/audit_batch_eval_summaries.py" "${_summaries[@]}" \
    | sed -e 's/^### /##### /' -e 's/^## /#### /' -e 's/^# /### /' \
    >> "${FAIRNESS_MD}"
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    audit_rc=1
  fi
  echo "" >> "${FAIRNESS_MD}"
done

if [ "${audit_rc}" -ne 0 ]; then
  echo "[WARN] fairness audit reported FAIL (see ${FAIRNESS_MD})"
fi

echo "[DONE] ${SUITE_ROOT}"
exit "${audit_rc}"
