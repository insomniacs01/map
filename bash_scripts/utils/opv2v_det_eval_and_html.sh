#!/bin/bash

# Run OPV2V BEV detection eval + generate offline HTML (pointcloud + GT/pred 3D boxes).
#
# Usage:
#   cd map-anything
#   bash bash_scripts/utils/opv2v_det_eval_and_html.sh \
#     experiments/mapanything/training/<run>/checkpoint-last.pth \
#     bev_centernet_wide_v4 \
#     eval_runs/my_eval \
#     50 42
#
# Args:
#   1) CKPT_PATH       Path to checkpoint (.pth)
#   2) DET_HEAD_CFG    Hydra det head config name under `configs/model/det_head` (e.g., bev_centernet_wide_v4)
#   3) OUT_ROOT        Output directory under `eval_runs/`
#   4) SAMPLE_SIZE     Number of frames to sample (default: 50)
#   5) SEED            Sampling seed (default: 42)
#
# Env:
#   EVAL_MODES         Space-separated list of modes (e.g., "single", "coop" or "single coop")

set -euo pipefail

CKPT_PATH=${1:?checkpoint path required}
DET_HEAD_CFG=${2:-bev_centernet_wide_v4}
OUT_ROOT=${3:-eval_runs/opv2v_det_eval}
SAMPLE_SIZE=${4:-50}
SEED=${5:-42}

DET_SCORE_THRESH=${DET_SCORE_THRESH:-0.05}
DET_MAX_DETS=${DET_MAX_DETS:-100}
DET_NMS_IOU=${DET_NMS_IOU:-0.1}
DET_IOU_THRESH=${DET_IOU_THRESH:-0.5}

PRED_MIN_SCORE=${PRED_MIN_SCORE:-0.3}
PRED_MAX_BOXES=${PRED_MAX_BOXES:-50}
EVAL_MODES=${EVAL_MODES:-single}

export HYDRA_FULL_ERROR=1

python scripts/batch_eval.py \
  --split test \
  --sample_size "${SAMPLE_SIZE}" \
  --seed "${SEED}" \
  --modes ${EVAL_MODES} \
  --save_representative \
  --det_metrics \
  --det_score_thresh "${DET_SCORE_THRESH}" \
  --det_nms_iou "${DET_NMS_IOU}" \
  --det_max_dets "${DET_MAX_DETS}" \
  --det_iou_thresh "${DET_IOU_THRESH}" \
  --det_x_min -50 \
  --det_x_max 120 \
  --det_y_min -50 \
  --det_y_max 50 \
  --det_voxel_size 0.5 \
  --det_head_cfg "${DET_HEAD_CFG}" \
  --keep_camera_poses \
  --model_task posed_sfm \
  --models det_model="${CKPT_PATH}" \
  --model_filter det_model \
  --output_root "${OUT_ROOT}"

# Attach explicit run metadata for traceability (checkpoint + eval knobs).
export OUT_ROOT CKPT_PATH DET_HEAD_CFG SAMPLE_SIZE SEED EVAL_MODES
python - <<PY
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
summary_path = out_root / "summary_test.json"
if not summary_path.is_file():
    raise SystemExit(f"summary not found: {summary_path}")

data = json.loads(summary_path.read_text(encoding="utf-8"))
run_info = data.get("run_info") or {}
run_info.update(
    {
        "checkpoint_path": str(Path(os.environ["CKPT_PATH"]).resolve()),
        "det_head_cfg": os.environ["DET_HEAD_CFG"],
        "sample_size": int(os.environ["SAMPLE_SIZE"]),
        "seed": int(os.environ["SEED"]),
        "eval_modes": os.environ["EVAL_MODES"].split(),
    }
)
data["run_info"] = run_info

summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"[INFO] patched summary metadata -> {summary_path}")
PY

python scripts/make_batch_eval_pcd_html.py \
  --eval_root "${OUT_ROOT}" \
  --draw_gt_boxes \
  --draw_pred_boxes \
  --pred_min_score "${PRED_MIN_SCORE}" \
  --pred_max_boxes "${PRED_MAX_BOXES}"

echo "[OK] Open: ${OUT_ROOT}/html/index.html"
