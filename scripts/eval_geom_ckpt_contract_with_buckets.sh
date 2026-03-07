#!/bin/bash
set -euo pipefail

# Evaluate one geometry checkpoint on a fixed frames contract, then produce a
# baseline-distance bucket breakdown (no-inference re-aggregation).
#
# This is the "strict & fair" entrypoint when you want:
# - fixed contract eval (reproducible)
# - overall metrics (summary_test.json)
# - per-bucket metrics (geom_by_baseline_buckets.md)
#
# It is contract-driven, so it can be reused for future datasets as long as the
# contract provides `baseline_distance_m` (recommended). If not, we fall back to
# OPV2V YAML to compute baseline distance for bucketing.
#
# Usage:
#   bash map-anything/scripts/eval_geom_ckpt_contract_with_buckets.sh \
#     <ckpt.pth> <frames_json> <out_root> [cuda_device=0]

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <ckpt_path> <frames_json> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
FRAMES_JSON="$2"
OUT_ROOT="$3"
CUDA_DEV="${4:-0}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERR] missing checkpoint: ${CKPT_PATH}"
  exit 2
fi
if [ ! -f "${FRAMES_JSON}" ]; then
  echo "[ERR] missing frames_json: ${FRAMES_JSON}"
  exit 3
fi

mkdir -p "${OUT_ROOT}"

# Eval protocol knobs (override via env as needed).
EVAL_MODES="${EVAL_MODES:-coop}"
MODEL_TASK="${MODEL_TASK:-calibrated_sfm}"
EVAL_KEEP_CAMERA_POSES="${EVAL_KEEP_CAMERA_POSES:-0}"
EVAL_KEEP_MAIN_AGENT_POSES="${EVAL_KEEP_MAIN_AGENT_POSES:-0}"
# Model family knobs.
MODEL_ARCH="${MODEL_ARCH:-mapanything}"  # mapanything | vggt
DATA_NORM_TYPE="${DATA_NORM_TYPE:-}"  # e.g. dinov2 | identity (usually leave empty)
VGGT_ENABLE_METRIC_SCALE_HEAD="${VGGT_ENABLE_METRIC_SCALE_HEAD:-0}"
VGGT_AUTOCAST_DTYPE="${VGGT_AUTOCAST_DTYPE:-}"  # auto|bf16|fp16|fp32
# When 1, reuse existing `${OUT_ROOT}/summary_test.json` + metrics CSVs (no re-inference).
SKIP_BATCH_EVAL_IF_EXISTS="${SKIP_BATCH_EVAL_IF_EXISTS:-1}"

# Bucketing knobs.
BUCKET_EDGES="${BUCKET_EDGES:-0,15,30,60,100,1000000000}"

# OPV2V paths (only used when we need to compute GT scale / distances from files).
IMAGES_ROOT="${IMAGES_ROOT:-${REPO_ROOT}/data/opv2v_images}"
DEPTH_ROOT="${DEPTH_ROOT:-${REPO_ROOT}/data/opv2v_depth}"

# Scale-to-GT needs a matching GT scale contract. For OPV2V we can auto-generate it.
AUTO_GT_SCALE_CONTRACT="${AUTO_GT_SCALE_CONTRACT:-1}"
GT_SCALE_CONTRACT_JSON="${GT_SCALE_CONTRACT_JSON:-}"

split="$(
  "${PYTHON}" - "${FRAMES_JSON}" <<'PY'
import json, sys
p = sys.argv[1]
doc = json.loads(open(p, "r", encoding="utf-8").read())
print(str(doc.get("split") or "test"))
PY
)"

if [ -z "${GT_SCALE_CONTRACT_JSON}" ]; then
  # Default: put it under eval_runs so multiple checkpoints evaluated on the same
  # contract can reuse the same GT-scale contract (depends only on frames_json).
  stem="$(basename "${FRAMES_JSON}" .json)"
  gt_stem="${stem}"
  if [[ "${gt_stem}" == frames_* ]]; then
    gt_stem="${gt_stem#frames_}"
  fi
  GT_SCALE_CONTRACT_JSON="${REPO_ROOT}/eval_runs/gt_scale_${gt_stem}_avg_dis.json"
fi

if [ "${AUTO_GT_SCALE_CONTRACT}" = "1" ] && [ ! -f "${GT_SCALE_CONTRACT_JSON}" ]; then
  echo "[INFO] computing gt_scale_contract_json=${GT_SCALE_CONTRACT_JSON}"
  PYTHONPATH="${REPO_ROOT}" \
  "${PYTHON}" "${REPO_ROOT}/scripts/compute_opv2v_gt_scale_contract.py" \
    --frames_json "${FRAMES_JSON}" \
    --images_root "${IMAGES_ROOT}" \
    --depth_root "${DEPTH_ROOT}" \
    --out_json "${GT_SCALE_CONTRACT_JSON}"
fi

gt_scale_flag=()
if [ -f "${GT_SCALE_CONTRACT_JSON}" ]; then
  gt_scale_flag+=(--gt_scale_contract_json "${GT_SCALE_CONTRACT_JSON}")
else
  echo "[WARN] gt_scale_contract_json missing: ${GT_SCALE_CONTRACT_JSON}"
  echo "[WARN] scale_to_gt_* metrics will be NA (not recommended for promotion/gating)."
fi

read -r -a _modes <<< "${EVAL_MODES}"
if [ "${#_modes[@]}" -eq 0 ]; then
  _modes=(coop)
fi

keep_flag=()
if [ "${EVAL_KEEP_CAMERA_POSES}" = "1" ]; then
  keep_flag+=(--keep_camera_poses)
fi

main_pose_flag=()
if [ "${EVAL_KEEP_MAIN_AGENT_POSES}" = "1" ]; then
  main_pose_flag+=(--keep_main_agent_poses)
fi

echo "[EVAL] ckpt=${CKPT_PATH}"
echo "[EVAL] out=${OUT_ROOT}"
echo "[EVAL] split=${split} modes=${EVAL_MODES} model_task=${MODEL_TASK} model_arch=${MODEL_ARCH}"
echo "[EVAL] frames_json=${FRAMES_JSON}"
echo "[EVAL] gt_scale_contract_json=${GT_SCALE_CONTRACT_JSON}"

skip_eval=0
if [ "${SKIP_BATCH_EVAL_IF_EXISTS}" = "1" ] && [ -f "${OUT_ROOT}/summary_test.json" ]; then
  skip_eval=1
  for m in "${_modes[@]}"; do
    if [ ! -f "${OUT_ROOT}/geom_model/${m}_metrics.csv" ]; then
      skip_eval=0
      break
    fi
  done
fi

if [ "${skip_eval}" = "1" ]; then
  echo "[SKIP] batch_eval (found existing summary + metrics csvs under ${OUT_ROOT}/geom_model)"
else
  arch_flag=(--model_arch "${MODEL_ARCH}")
  norm_flag=()
  if [ -n "${DATA_NORM_TYPE}" ]; then
    norm_flag+=(--data_norm_type "${DATA_NORM_TYPE}")
  fi
  vggt_flag=()
  if [ "${MODEL_ARCH}" = "vggt" ]; then
    if [ "${VGGT_ENABLE_METRIC_SCALE_HEAD}" = "1" ]; then
      vggt_flag+=(--vggt_enable_metric_scale_head)
    fi
    if [ -n "${VGGT_AUTOCAST_DTYPE}" ]; then
      vggt_flag+=(--vggt_autocast_dtype "${VGGT_AUTOCAST_DTYPE}")
    fi
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEV}" \
  PYTHONPATH="${REPO_ROOT}" \
  "${PYTHON}" -u "${REPO_ROOT}/scripts/batch_eval.py" \
    --split "${split}" \
    --frames_json "${FRAMES_JSON}" \
    --modes "${_modes[@]}" \
    "${keep_flag[@]}" \
    "${main_pose_flag[@]}" \
    --model_task "${MODEL_TASK}" \
    "${arch_flag[@]}" \
    "${norm_flag[@]}" \
    "${vggt_flag[@]}" \
    --models "geom_model=${CKPT_PATH}" \
    --model_filter geom_model \
    --output_root "${OUT_ROOT}" \
    "${gt_scale_flag[@]}"
fi

# Bucket the contract (contract-only if baseline_distance_m exists; otherwise OPV2V YAML fallback).
BUCKET_DIR="${OUT_ROOT}/bucket_contracts"
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON}" "${REPO_ROOT}/scripts/bucket_frames_by_baseline_distance.py" \
  --frames_json "${FRAMES_JSON}" \
  --images_root "${IMAGES_ROOT}" \
  --bucket_edges "${BUCKET_EDGES}" \
  --out_dir "${BUCKET_DIR}" \
  --tag "$(basename "${FRAMES_JSON}" .json)"

# No-inference re-aggregation over per-frame CSVs.
if [ -f "${GT_SCALE_CONTRACT_JSON}" ]; then
  PYTHONPATH="${REPO_ROOT}" \
  "${PYTHON}" "${REPO_ROOT}/scripts/report_geom_by_baseline_buckets.py" \
    --bucket_report_json "${BUCKET_DIR}/bucket_report.json" \
    --gt_scale_contract_json "${GT_SCALE_CONTRACT_JSON}" \
    --metrics_roots "geom_model=${OUT_ROOT}/geom_model" \
    --out_md "${OUT_ROOT}/geom_by_baseline_buckets.md" \
    --out_json "${OUT_ROOT}/geom_by_baseline_buckets.json"
fi

echo "[DONE] ${OUT_ROOT}"
