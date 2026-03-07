#!/bin/bash
set -euo pipefail

# Evaluate one det/e2e checkpoint on a fixed frames contract, then produce a
# baseline-distance bucket AP breakdown (no-inference re-scoring from cached preds).
#
# This is the det counterpart of `scripts/eval_geom_ckpt_contract_with_buckets.sh`.
#
# Usage:
#   bash map-anything/scripts/eval_det_ckpt_contract_with_buckets.sh \
#     <ckpt.pth> <det_head_cfg> <frames_json> <out_root> [cuda_device=0]
#
# Defaults (override via env):
# - Deployment-like protocol: KEEP_CAMERA_POSES=0, KEEP_MAIN_AGENT_POSES=1, MODEL_TASK=calibrated_sfm
# - Save det cache for offline bucket reports: --save_det_cache

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

if [ $# -lt 4 ]; then
  echo "Usage: $0 <ckpt_path> <det_head_cfg> <frames_json> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
DET_HEAD_CFG="$2"
FRAMES_JSON="$3"
OUT_ROOT="$4"
CUDA_DEV="${5:-0}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERR] missing checkpoint: ${CKPT_PATH}"
  exit 2
fi
if [ ! -f "${FRAMES_JSON}" ]; then
  echo "[ERR] missing frames_json: ${FRAMES_JSON}"
  exit 3
fi

mkdir -p "${OUT_ROOT}"

# Protocol knobs (override via env).
MODEL_ARCH="${MODEL_ARCH:-mapanything}"  # mapanything|vggt
EVAL_MODES="${EVAL_MODES:-single coop}"

KEEP_CAMERA_POSES="${KEEP_CAMERA_POSES:-0}"
KEEP_MAIN_AGENT_POSES="${KEEP_MAIN_AGENT_POSES:-1}"

# Default model task:
# - posed eval (KEEP_CAMERA_POSES=1): posed_sfm (upper bound / legacy)
# - unposed eval (KEEP_CAMERA_POSES=0): calibrated_sfm (deployment-like)
MODEL_TASK_DEFAULT="posed_sfm"
if [ "${KEEP_CAMERA_POSES}" = "0" ]; then
  MODEL_TASK_DEFAULT="calibrated_sfm"
fi
MODEL_TASK="${MODEL_TASK:-${MODEL_TASK_DEFAULT}}"

# Decode/eval knobs (override via env).
DET_SCORE_THRESH="${DET_SCORE_THRESH:-0.05}"
DET_MAX_DETS="${DET_MAX_DETS:-100}"
DET_NMS_IOU="${DET_NMS_IOU:-0.1}"
DET_IOU_THRESH="${DET_IOU_THRESH:-0.5}"

DET_X_MIN="${DET_X_MIN:--50}"
DET_X_MAX="${DET_X_MAX:-120}"
DET_Y_MIN="${DET_Y_MIN:--50}"
DET_Y_MAX="${DET_Y_MAX:-50}"
DET_VOXEL_SIZE="${DET_VOXEL_SIZE:-0.5}"

# Optional: load ONLY det_head weights from another checkpoint (composed eval).
DET_HEAD_CKPT="${DET_HEAD_CKPT:-""}"
det_head_ckpt_flag=()
if [ -n "${DET_HEAD_CKPT}" ]; then
  det_head_ckpt_flag+=(--det_head_ckpt "${DET_HEAD_CKPT}")
fi

# Bucketing knobs.
BUCKET_EDGES="${BUCKET_EDGES:-0,15,30,60,100,1000000000}"

split="$(
  "${PYTHON}" - "${FRAMES_JSON}" <<'PY'
import json, sys
p = sys.argv[1]
doc = json.loads(open(p, "r", encoding="utf-8").read())
print(str(doc.get("split") or "test"))
PY
)"

keep_flag=()
if [ "${KEEP_CAMERA_POSES}" = "1" ]; then
  keep_flag+=(--keep_camera_poses)
fi

main_pose_flag=()
if [ "${KEEP_MAIN_AGENT_POSES}" = "1" ]; then
  main_pose_flag+=(--keep_main_agent_poses)
fi

echo "[EVAL] ckpt=${CKPT_PATH}"
echo "[EVAL] det_head_cfg=${DET_HEAD_CFG}"
echo "[EVAL] out=${OUT_ROOT}"
echo "[EVAL] split=${split} modes=${EVAL_MODES} model_task=${MODEL_TASK} model_arch=${MODEL_ARCH}"
echo "[EVAL] frames_json=${FRAMES_JSON}"

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" \
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON}" -u "${REPO_ROOT}/scripts/batch_eval.py" \
  --split "${split}" \
  --frames_json "${FRAMES_JSON}" \
  --model_arch "${MODEL_ARCH}" \
  --modes ${EVAL_MODES} \
  --det_metrics \
  --save_det_cache \
  --det_score_thresh "${DET_SCORE_THRESH}" \
  --det_nms_iou "${DET_NMS_IOU}" \
  --det_max_dets "${DET_MAX_DETS}" \
  --det_iou_thresh "${DET_IOU_THRESH}" \
  --det_x_min "${DET_X_MIN}" \
  --det_x_max "${DET_X_MAX}" \
  --det_y_min "${DET_Y_MIN}" \
  --det_y_max "${DET_Y_MAX}" \
  --det_voxel_size "${DET_VOXEL_SIZE}" \
  --det_head_cfg "${DET_HEAD_CFG}" \
  "${det_head_ckpt_flag[@]}" \
  "${keep_flag[@]}" \
  "${main_pose_flag[@]}" \
  --model_task "${MODEL_TASK}" \
  --models det_model="${CKPT_PATH}" \
  --model_filter det_model \
  --output_root "${OUT_ROOT}"

# Bucket the contract (contract-only if baseline_distance_m exists; otherwise OPV2V YAML fallback).
BUCKET_DIR="${OUT_ROOT}/bucket_contracts"
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON}" "${REPO_ROOT}/scripts/bucket_frames_by_baseline_distance.py" \
  --frames_json "${FRAMES_JSON}" \
  --images_root "${REPO_ROOT}/data/opv2v_images" \
  --bucket_edges "${BUCKET_EDGES}" \
  --out_dir "${BUCKET_DIR}" \
  --tag "$(basename "${FRAMES_JSON}" .json)"

# No-inference re-score: det AP by bucket (requires det cache).
for mode in ${EVAL_MODES}; do
  cache_npz="${OUT_ROOT}/det_model/det_ap_cache_${mode}.npz"
  if [ ! -f "${cache_npz}" ]; then
    echo "[WARN] missing det cache for mode=${mode}: ${cache_npz} (skip bucket AP report)"
    continue
  fi
  out_md="${OUT_ROOT}/det_ap_by_baseline_buckets_${mode}.md"
  out_json="${OUT_ROOT}/det_ap_by_baseline_buckets_${mode}.json"
  PYTHONPATH="${REPO_ROOT}" \
  "${PYTHON}" "${REPO_ROOT}/scripts/report_det_ap_by_baseline_buckets.py" \
    --det_cache_npz "${cache_npz}" \
    --frames_json "${FRAMES_JSON}" \
    --bucket_report_json "${BUCKET_DIR}/bucket_report.json" \
    --out_md "${out_md}" \
    --out_json "${out_json}"
done

echo "[DONE] ${OUT_ROOT}"

