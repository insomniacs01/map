#!/bin/bash

# Deterministic det/e2e eval on the canonical pinhole Test50 contract + offline viz.
#
# This script is the det counterpart of `scripts/eval_geom_ckpt_test500.sh`.
#
# Usage:
#   cd map-anything
#   bash scripts/eval_det_ckpt_test50.sh <ckpt.pth> <det_head_cfg> <out_root>
#
# Notes:
# - By default we DO feed GT extrinsics (`KEEP_CAMERA_POSES=1`) to match the
#   historical locked pinhole det/e2e baselines (`det_e2e_v5_*`).
# - Set `KEEP_CAMERA_POSES=0` to test the deployment-like regime (pose must be predicted);
#   note that many legacy det heads were configured with `point_pose_source=gt` and may
#   collapse under unposed inference unless retrained / reconfigured.
# - The evaluated frames are fixed by CONTRACT_JSON (default: frames_test50_det_e2e_v5.json).

set -euo pipefail

CKPT_PATH=${1:?checkpoint path required}
DET_HEAD_CFG=${2:?det_head_cfg required (configs/model/det_head/<name>.yaml)}
OUT_ROOT=${3:?output dir required (e.g., eval_runs/det_fairlock_test50_<tag>)}
CUDA_DEV=${4:-0}

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

MODEL_ARCH=${MODEL_ARCH:-"mapanything"}  # mapanything|vggt

CONTRACT_JSON=${CONTRACT_JSON:-"${REPO_ROOT}/eval_runs/frames_test50_det_e2e_v5.json"}
EVAL_MODES=${EVAL_MODES:-"single coop"}

# Decode/eval knobs (locked for fair comparisons).
DET_SCORE_THRESH=${DET_SCORE_THRESH:-0.05}
DET_MAX_DETS=${DET_MAX_DETS:-100}
DET_NMS_IOU=${DET_NMS_IOU:-0.1}
DET_IOU_THRESH=${DET_IOU_THRESH:-0.5}

# Visualization knobs (do not affect AP; only controls what boxes are drawn).
PRED_MIN_SCORE=${PRED_MIN_SCORE:-0.3}
PRED_MAX_BOXES=${PRED_MAX_BOXES:-50}

KEEP_CAMERA_POSES=${KEEP_CAMERA_POSES:-1}
KEEP_MAIN_AGENT_POSES=${KEEP_MAIN_AGENT_POSES:-0}

# Default model task:
# - posed eval (KEEP_CAMERA_POSES=1): posed_sfm (legacy locked baselines)
# - unposed eval (KEEP_CAMERA_POSES=0): calibrated_sfm (deployment-like; pose must be predicted)
MODEL_TASK_DEFAULT="posed_sfm"
if [ "${KEEP_CAMERA_POSES}" = "0" ]; then
  MODEL_TASK_DEFAULT="calibrated_sfm"
fi
MODEL_TASK=${MODEL_TASK:-${MODEL_TASK_DEFAULT}}

# Optional: load ONLY det_head weights from another checkpoint (composed eval).
# This is useful when evaluating a geometry-only checkpoint with a fixed det head.
DET_HEAD_CKPT=${DET_HEAD_CKPT:-""}
det_head_ckpt_flag=()
if [ -n "${DET_HEAD_CKPT}" ]; then
  det_head_ckpt_flag+=(--det_head_ckpt "${DET_HEAD_CKPT}")
fi

export HYDRA_FULL_ERROR=1

keep_flag=()
if [ "${KEEP_CAMERA_POSES}" = "1" ]; then
  keep_flag+=(--keep_camera_poses)
fi

main_pose_flag=()
if [ "${KEEP_MAIN_AGENT_POSES}" = "1" ]; then
  main_pose_flag+=(--keep_main_agent_poses)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" \
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON}" "${REPO_ROOT}/scripts/batch_eval.py" \
  --split test \
  --frames_json "${CONTRACT_JSON}" \
  --model_arch "${MODEL_ARCH}" \
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
  "${det_head_ckpt_flag[@]}" \
  "${keep_flag[@]}" \
  "${main_pose_flag[@]}" \
  --model_task "${MODEL_TASK}" \
  --models det_model="${CKPT_PATH}" \
  --model_filter det_model \
  --output_root "${OUT_ROOT}"

PYTHONPATH="${REPO_ROOT}" \
"${PYTHON}" "${REPO_ROOT}/scripts/make_batch_eval_pcd_html.py" \
  --eval_root "${OUT_ROOT}" \
  --draw_gt_boxes \
  --draw_pred_boxes \
  --pred_min_score "${PRED_MIN_SCORE}" \
  --pred_max_boxes "${PRED_MAX_BOXES}"

# Generate simple 2D BEV PNG overlays for representatives (more readable than 3D HTML).
export OUT_ROOT CONTRACT_JSON REPO_ROOT
PYTHONPATH="${REPO_ROOT}" "${PYTHON}" - <<'PY'
import os
import re
import json
import sys
from pathlib import Path
import subprocess

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
out_root = Path(os.environ["OUT_ROOT"]).resolve()
contract_json = Path(os.environ["CONTRACT_JSON"]).resolve()

doc = json.loads(contract_json.read_text(encoding="utf-8"))
split = str(doc.get("split") or "test")
frames = doc.get("frames") or []
lookup = {}
for fr in frames:
    if not isinstance(fr, dict):
        continue
    seq = fr.get("sequence")
    frame = fr.get("frame")
    main = fr.get("main_agent")
    if isinstance(seq, str) and isinstance(frame, str) and main is not None:
        lookup[(seq, frame)] = str(main)

rep_dirs = []
for p in out_root.rglob("*_representatives"):
    if p.is_dir():
        rep_dirs.append(p)

png_dir = out_root / "bev_png"
png_dir.mkdir(parents=True, exist_ok=True)

# NOTE: This is a Python regex (raw string). Do not double-escape backslashes here.
pat = re.compile(r"^(?P<seq>.+)_(?P<frame>\d{6})_(?P<tag>best|median|worst)\.pcd$")

items = []
for rep_dir in sorted(rep_dirs):
    # rep_dir like: <out>/<model>/<mode>_representatives
    mode = rep_dir.name.replace("_representatives", "")
    model = rep_dir.parent.name
    for pcd in sorted(rep_dir.glob("*.pcd")):
        m = pat.match(pcd.name)
        if not m:
            continue
        seq = m.group("seq")
        frame = m.group("frame")
        main = lookup.get((seq, frame))
        if main is None:
            # Fallback: main_agent is typically the smallest id under the sequence dir.
            seq_dir = repo_root / "data" / "opv2v" / split / seq
            agents = sorted([d.name for d in seq_dir.iterdir() if d.is_dir() and d.name.isdigit()]) if seq_dir.is_dir() else []
            main = agents[0] if agents else None
        if main is None:
            continue
        yaml_path = repo_root / "data" / "opv2v" / split / seq / main / f"{frame}.yaml"
        if not yaml_path.is_file():
            continue
        out_png = png_dir / model / mode / (pcd.stem + "_bev.png")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "viz_batch_eval_bev_png.py"),
            "--pred_pcd", str(pcd),
            "--yaml", str(yaml_path),
            "--out_png", str(out_png),
            "--pred_min_score", "0.05",
            "--det_x_min", "-50",
            "--det_x_max", "120",
            "--det_y_min", "-50",
            "--det_y_max", "50",
            "--color_pred_by_match",
        ]
        subprocess.run(cmd, check=True)
        items.append((model, mode, out_png.relative_to(png_dir)))

# Simple index.html (embed images; easier to eyeball than raw links).
summary_path = out_root / "summary_test.json"
meta_html = ""
metrics_html = ""
if summary_path.is_file():
    try:
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = summ.get("meta") or {}
        m = summ.get("metrics") or {}
        # Only one model key is expected for this script, but keep it generic.
        metric_lines = []
        for model_key, per_mode in m.items():
            if not isinstance(per_mode, dict):
                continue
            for mode_key, md in per_mode.items():
                if not isinstance(md, dict):
                    continue
                ap = md.get("det_ap_iou")
                prec = md.get("det_precision_iou")
                rec = md.get("det_recall_iou")
                metric_lines.append(f"{model_key}/{mode_key}: ap={ap} prec={prec} rec={rec}")
        metrics_html = "<pre>" + "\n".join(metric_lines) + "</pre>" if metric_lines else ""
        meta_html = "<pre>" + json.dumps(
            {
                "frames_hash_md5": meta.get("frames_hash_md5"),
                "frames_json": meta.get("frames_json"),
                "det_head_cfg": meta.get("det_head_cfg"),
                "det_decode_cfg": meta.get("det_decode_cfg"),
                "model_task": meta.get("model_task"),
                "keep_camera_poses": meta.get("keep_camera_poses"),
                "eval_script_md5": meta.get("eval_script_md5"),
            },
            indent=2,
        ) + "</pre>"
    except Exception:
        pass

by_model_mode = {}
for model, mode, rel in items:
    by_model_mode.setdefault((model, mode), []).append(rel.as_posix())
for k in by_model_mode:
    by_model_mode[k].sort()

index = png_dir / "index.html"
rows = []
rows.append("<html><head><meta charset=\"utf-8\"/>")
rows.append("<style>body{font-family:Arial,Helvetica,sans-serif} img{max-width:520px;height:auto;border:1px solid #ddd} .grid{display:flex;flex-wrap:wrap;gap:12px} .tile{width:540px}</style>")
rows.append("</head><body>")
rows.append("<h1>Det BEV PNG Gallery</h1>")
rows.append(f"<p>eval_root: <code>{out_root}</code></p>")
if metrics_html:
    rows.append("<h2>Det Summary</h2>")
    rows.append(metrics_html)
if meta_html:
    rows.append("<h2>Protocol Meta</h2>")
    rows.append(meta_html)

for (model, mode), rels in sorted(by_model_mode.items()):
    rows.append(f"<h2>{model} / {mode}</h2>")
    rows.append("<div class=\"grid\">")
    for rel in rels:
        rows.append("<div class=\"tile\">")
        rows.append(f"<div><code>{rel}</code></div>")
        rows.append(f"<a href=\"{rel}\"><img src=\"{rel}\"/></a>")
        rows.append("</div>")
    rows.append("</div>")

rows.append("</body></html>")
index.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"[OK] BEV PNG index -> {index}")
PY

echo "[OK] HTML: ${OUT_ROOT}/html/index.html"
echo "[OK] BEV PNG: ${OUT_ROOT}/bev_png/index.html"
