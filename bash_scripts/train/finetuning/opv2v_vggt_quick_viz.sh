#!/bin/bash

# Quick visualization sanity-check for OPV2V VGGT finetune checkpoints.
#
# It generates:
# - 2D camera box projections (GT=green, pred-pose=magenta)
# - (coop only) an interactive HTML comparing GT LiDAR + GT boxes vs predicted reconstruction
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_vggt_quick_viz.sh \
#     --mode single \
#     --checkpoint experiments/vggt/training/<run>/<ts>/checkpoint-last.pth \
#     --index 0 \
#     --out_dir eval_runs/opv2v_vggt_quick_viz/single_idx0
#
# For scale-head checkpoints, add: --scalehead

set -euo pipefail

MODE=""
CKPT=""
INDEX="0"
OUT_DIR=""
ENABLE_SCALE_HEAD="0"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --checkpoint)
      CKPT="$2"
      shift 2
      ;;
    --index)
      INDEX="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --scalehead)
      ENABLE_SCALE_HEAD="1"
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${MODE}" ] || [ -z "${CKPT}" ]; then
  echo "Usage: $0 --mode <single|coop> --checkpoint <path> [--index N] [--out_dir DIR] [--scalehead]" >&2
  exit 1
fi

if [ -z "${OUT_DIR}" ]; then
  OUT_DIR="eval_runs/opv2v_vggt_quick_viz/${MODE}_idx${INDEX}_$(date +%Y%m%d_%H%M%S)"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "$MODE" = "single" ]; then
  NUM_VIEWS="4"
elif [ "$MODE" = "coop" ]; then
  NUM_VIEWS="8"
else
  echo "Error: --mode must be 'single' or 'coop', got: ${MODE}" >&2
  exit 3
fi

SCALE_FLAG=()
if [ "${ENABLE_SCALE_HEAD}" = "1" ]; then
  SCALE_FLAG=(--enable_metric_scale_head)
fi

mkdir -p "${OUT_DIR}"

echo "[viz] box projections -> ${OUT_DIR}/boxproj"
PYTHONPATH="$(pwd)" python scripts/viz_opv2v_boxes_on_images_pred_pose.py \
  --checkpoint "${CKPT}" \
  "${SCALE_FLAG[@]}" \
  --mode "${MODE}" \
  --split validate \
  --index "${INDEX}" \
  --num_views "${NUM_VIEWS}" \
  --out_dir "${OUT_DIR}/boxproj"

if [ "$MODE" = "coop" ]; then
  echo "[viz] html -> ${OUT_DIR}/pred_vs_gt.html"
  PYTHONPATH="$(pwd)" python scripts/viz_opv2v_vggt_pred_vs_gt_boxes_html.py \
    --checkpoint "${CKPT}" \
    "${SCALE_FLAG[@]}" \
    --split validate \
    --index "${INDEX}" \
    --num_views "${NUM_VIEWS}" \
    --out_html "${OUT_DIR}/pred_vs_gt.html"
fi

echo "[done] ${OUT_DIR}"

