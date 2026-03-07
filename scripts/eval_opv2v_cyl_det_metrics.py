#!/usr/bin/env python3
"""
Evaluate BEV detection metrics for OPV2V cylindrical cooperative scenes.

This script mirrors the detection metrics in scripts/batch_eval.py, but uses
OPV2VCoopCylindricalDataset for cylindrical panoramas and cooperative views.

It reports:
  - det_ap_iou / det_precision_iou / det_recall_iou (single IoU threshold)
  - det_mean_iou (mean of per-frame mean IoU)
  - aggregate det_num_gt/det_num_pred/det_tp_iou/det_fp_iou/det_fn_iou
  - per-frame TP/FP/FN and mean IoU (in `per_frame`)

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/eval_opv2v_cyl_det_metrics.py \
    --checkpoint experiments/opv2v_cyl_det_e2e/checkpoint-last.pth \
    --split validate --sequence 2021_09_11_00_33_16 --main_agent 1016 \
    --frames 000269,000271,000273,000275,000277 --num_views 4 \
    --max_agent_distance 60 --agent_selection_policy nearest \
    --det_head_cfg bev_centernet_wide_e2e_v1 --score_thresh 0.05 \
    --out_json eval_runs/cyl_det_eval/metrics_posegeom_det_r60.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse detection decoding + metrics from batch_eval for consistency.
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from batch_eval import (  # type: ignore
    DetMetricConfig,
    _match_boxes_greedy,
    compute_ap_bev,
    decode_bev_centernet,
)


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _scene_indices_for_frames(
    dataset: OPV2VCoopCylindricalDataset,
    *,
    sequence: str | None,
    frames: List[str],
    main_agent: str | None,
    frame_specs: List[Tuple[str, str]] | None = None,
) -> List[int]:
    wanted = []
    allowed = None
    if frame_specs:
        allowed = {(str(seq), str(frame)) for seq, frame in frame_specs}
    for idx, scene in enumerate(dataset.scenes):
        scene_seq = str(scene.get("sequence"))
        scene_frame = str(scene.get("frame"))
        if allowed is not None:
            if (scene_seq, scene_frame) not in allowed:
                continue
        elif sequence is not None and scene_seq != str(sequence):
            continue
        if frames and scene_frame not in frames:
            continue
        if main_agent is not None:
            agents = [str(a) for a in (scene.get("agents") or [])]
            if str(main_agent) not in agents:
                continue
        wanted.append(int(idx))
    return wanted


def _load_frames_from_summary(path: Path) -> Tuple[str | None, List[str], List[str], List[Tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames") or []
    sequences = []
    main_agents = []
    frame_ids = []
    frame_specs: List[Tuple[str, str]] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        seq = str(item.get("sequence")) if item.get("sequence") is not None else None
        frame = str(item.get("frame")) if item.get("frame") is not None else None
        agent = str(item.get("main_agent")) if item.get("main_agent") is not None else None
        if seq and frame:
            sequences.append(seq)
            frame_ids.append(frame)
            frame_specs.append((seq, frame))
        if agent:
            main_agents.append(agent)
    # If the summary mixes sequences, let caller pass --sequence explicitly.
    sequence = sequences[0] if sequences and len(set(sequences)) == 1 else None
    main_agent = main_agents[0] if main_agents and len(set(main_agents)) == 1 else None
    return sequence, frame_ids, main_agents, frame_specs


def _load_det_cfg(det_head_cfg: str | None) -> Dict[str, Any]:
    if det_head_cfg is None:
        return {}
    cfg_path = REPO_ROOT / "configs" / "model" / "det_head" / f"{det_head_cfg}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frames", type=str, default=None)
    parser.add_argument("--frames_json", type=Path, default=None)
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="posed_sfm")
    parser.add_argument(
        "--max_agent_distance",
        type=float,
        default=None,
        help="Only sample cooperative agents whose GT ego distance to main is <= this XY radius (meters).",
    )
    parser.add_argument(
        "--agent_selection_policy",
        type=str,
        default="nearest",
        choices=("random", "nearest"),
    )
    parser.add_argument("--det_head_cfg", type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument("--nms_iou", type=float, default=0.1)
    parser.add_argument("--max_dets", type=int, default=100)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--x_range", type=float, nargs=2, default=None)
    parser.add_argument("--y_range", type=float, nargs=2, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--min_density", type=float, default=None)
    parser.add_argument("--min_high_ratio", type=float, default=None)
    parser.add_argument("--min_var_z", type=float, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    ckpt_path = (REPO_ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    local_cfg = {
        "path": str((REPO_ROOT / "configs" / "train.yaml").resolve()),
        "checkpoint_path": str(ckpt_path),
        "config_overrides": [
            "machine=local_a800",
            "model=mapanything_det",
            "model.encoder.uses_torch_hub=false",
            f"model/det_head={args.det_head_cfg}" if args.det_head_cfg else None,
            f"model/task={args.task}",
        ],
        "strict": False,
    }
    local_cfg["config_overrides"] = [v for v in local_cfg["config_overrides"] if v]
    model = initialize_mapanything_local(local_cfg, device)
    model.eval()

    dataset = OPV2VCoopCylindricalDataset(
        split=args.split,
        ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
        depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
        camera_ids=(0, 1, 2, 3),
        num_views=int(args.num_views),
        variable_num_views=False,
        resolution=(1008, 252),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=777,
        include_vehicle_boxes=True,
        max_num_boxes=128,
        bbox_range=120.0,
        min_agents=2,
        min_num_views=int(args.num_views),
        max_num_views=int(args.num_views),
        panorama_resolution=(1008, 252),
        panorama_vertical_fov_deg=90.0,
        panorama_elevation_center_deg=0.0,
        view_selection_margin=0.05,
        max_num_retries=0,
        main_agent=args.main_agent,
        ensure_main_agent_first=True,
        max_agent_distance=float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        agent_selection_policy=str(args.agent_selection_policy),
    )

    frames = _parse_csv_list(args.frames)
    sequence = args.sequence
    main_agent = args.main_agent
    frame_specs = None
    if args.frames_json is not None:
        summary_path = (REPO_ROOT / args.frames_json).resolve() if not args.frames_json.is_absolute() else args.frames_json
        seq_from_json, frames_from_json, main_agents, frame_specs = _load_frames_from_summary(summary_path)
        if sequence is None:
            sequence = seq_from_json
        if not frames:
            frames = frames_from_json
        if main_agent is None and main_agents:
            # If multiple agents exist, leave main_agent None and filter per-frame by dataset scenes.
            if len(set(main_agents)) == 1:
                main_agent = main_agents[0]

    wanted = _scene_indices_for_frames(
        dataset,
        sequence=str(sequence) if sequence is not None else None,
        frames=frames,
        main_agent=str(main_agent) if main_agent is not None else None,
        frame_specs=frame_specs,
    )
    if not wanted:
        raise RuntimeError("No scenes found for the requested filters.")
    wanted = sorted(wanted, key=lambda i: str(dataset.scenes[i].get("frame", "")))

    det_cfg = DetMetricConfig(
        enabled=True,
        score_thresh=float(args.score_thresh),
        nms_iou=float(args.nms_iou),
        max_dets=int(args.max_dets),
        iou_thresh=float(args.iou_thresh),
        min_density=float(args.min_density) if args.min_density is not None else None,
        min_high_ratio=float(args.min_high_ratio) if args.min_high_ratio is not None else None,
        min_var_z=float(args.min_var_z) if args.min_var_z is not None else None,
    )
    det_head_cfg = _load_det_cfg(args.det_head_cfg)
    if args.x_range is not None:
        det_cfg.x_range = (float(args.x_range[0]), float(args.x_range[1]))
    elif "x_range" in det_head_cfg:
        det_cfg.x_range = tuple(float(v) for v in det_head_cfg["x_range"])
    if args.y_range is not None:
        det_cfg.y_range = (float(args.y_range[0]), float(args.y_range[1]))
    elif "y_range" in det_head_cfg:
        det_cfg.y_range = tuple(float(v) for v in det_head_cfg["y_range"])
    if args.voxel_size is not None:
        det_cfg.voxel_size = float(args.voxel_size)
    elif "voxel_size" in det_head_cfg:
        det_cfg.voxel_size = float(det_head_cfg["voxel_size"])

    det_preds: List[Tuple[float, Tuple[str, str], np.ndarray]] = []
    det_gts: Dict[Tuple[str, str], np.ndarray] = {}
    per_frame: List[Dict[str, Any]] = []

    for scene_idx in wanted:
        subset = torch.utils.data.Subset(dataset, [int(scene_idx)])
        loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
        views = next(iter(loader))
        for view in views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device)

        scene = dataset.scenes[int(scene_idx)]
        frame_id = str(scene.get("frame"))
        sequence_id = str(scene.get("sequence"))
        frame_key = (sequence_id, frame_id)

        with torch.no_grad():
            preds = model(views)

        det_out = preds[0].get("bev_det") if preds else None
        pred_boxes = decode_bev_centernet(det_out, det_cfg) if isinstance(det_out, dict) else []

        gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
        gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
        gt_boxes = gt_boxes[gt_mask]

        det_tp, det_fp, det_fn, det_mean_iou = _match_boxes_greedy(
            pred_boxes, gt_boxes, iou_thresh=float(det_cfg.iou_thresh)
        )
        det_gts[frame_key] = gt_boxes
        for item in pred_boxes:
            score = item.get("score")
            box = item.get("box")
            if isinstance(score, (float, int)) and isinstance(box, np.ndarray):
                det_preds.append((float(score), frame_key, box))

        per_frame.append(
            {
                "sequence": sequence_id,
                "frame": frame_id,
                "num_gt": int(gt_boxes.shape[0]),
                "num_pred": int(len(pred_boxes)),
                "det_tp": int(det_tp),
                "det_fp": int(det_fp),
                "det_fn": int(det_fn),
                "det_mean_iou": float(det_mean_iou) if det_mean_iou == det_mean_iou else float("nan"),
            }
        )

    ap, precision, recall = compute_ap_bev(
        det_preds, det_gts, iou_thresh=float(det_cfg.iou_thresh)
    )

    det_mean_iou = float(
        np.nanmean([float(x.get("det_mean_iou", float("nan"))) for x in per_frame])
        if per_frame
        else float("nan")
    )
    det_num_gt = int(sum(int(x.get("num_gt", 0)) for x in per_frame))
    det_num_pred = int(sum(int(x.get("num_pred", 0)) for x in per_frame))
    det_tp_iou = int(sum(int(x.get("det_tp", 0)) for x in per_frame))
    det_fp_iou = int(sum(int(x.get("det_fp", 0)) for x in per_frame))
    det_fn_iou = int(sum(int(x.get("det_fn", 0)) for x in per_frame))

    summary = {
        "checkpoint": str(ckpt_path),
        "split": str(args.split),
        "sequence": str(sequence) if sequence is not None else None,
        "frames": frames,
        "main_agent": str(main_agent) if main_agent is not None else None,
        "num_views": int(args.num_views),
        "max_agent_distance": float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        "agent_selection_policy": str(args.agent_selection_policy),
        "det_cfg": asdict(det_cfg),
        "det_ap_iou": float(ap),
        "det_precision_iou": float(precision),
        "det_recall_iou": float(recall),
        "det_mean_iou": det_mean_iou,
        "det_num_gt": det_num_gt,
        "det_num_pred": det_num_pred,
        "det_tp_iou": det_tp_iou,
        "det_fp_iou": det_fp_iou,
        "det_fn_iou": det_fn_iou,
        "per_frame": per_frame,
    }

    print(
        f"[SUMMARY] det_ap_iou={summary['det_ap_iou']:.4f} "
        f"precision={summary['det_precision_iou']:.4f} "
        f"recall={summary['det_recall_iou']:.4f}"
    )

    if args.out_json is not None:
        out_path = args.out_json
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
