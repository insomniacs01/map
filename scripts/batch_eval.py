#!/usr/bin/env python3
"""
Batch evaluator for OPV2V models on sampled frames.

It supports both single-agent (solo) and cooperative (multi-agent) evaluations
using three checkpoints:
    1. Pretrained (no finetune)
    2. Stage1 finetune (single agent)
    3. Stage2 finetune (multi agent)

The script randomly samples frames from the specified split (default: test),
runs inference on a chosen GPU, logs pose/depth/scale metrics, and stores the
results plus representative point clouds/screenshots under a workspace path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.spatial import cKDTree

# Allow imports from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

# Reduce noisy INFO logs from intrinsics utilities during large-contract evals.
logging.getLogger("color_compare").setLevel(logging.ERROR)

from color_compare import (  # type: ignore  # noqa: E402
    IMAGE_EXTENSIONS,
    preprocess_inputs,
    rescale_intrinsics_to_image,
    strip_external_calibration_inputs,
)
from mapanything.datasets.opv2v import _convert_pose_to_opencv  # type: ignore  # noqa: E402
from mapanything.utils.geometry import depthmap_to_world_frame, quaternion_to_rotation_matrix  # noqa: E402
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local  # noqa: E402
from mapanything.utils.opv2v_pointclouds import predictions_to_pointcloud, save_point_cloud  # noqa: E402

from data_processing.opv2v_pose_utils import (  # type: ignore  # noqa: E402
    CARLA_TO_CAMERA_CV,
    cords_to_pose,
    get_camera_poses_in_ego,
    load_frame_metadata,
    load_ascii_pcd_xyz,
)


@dataclass
class FrameInfo:
    sequence: str
    frame: str
    main_agent: str = "641"
    coop_agents: Tuple[str, ...] = ("641", "650")
    # Optional metadata carried from a frames contract (useful for bucketed analysis).
    baseline_distance_m: float | None = None
    baseline_bucket: str | None = None


@dataclass
class EvalResult:
    frame: FrameInfo
    mode: str  # "single" or "coop"
    pose_abs: float
    pose_rot: float
    depth_rmse: float
    depth_mae: float
    depth_rel: float
    # Cross-agent relative pose (only meaningful for coop). These measure the error of the
    # relative transform between the main agent camera0 and the paired agent camera0.
    cross_agent_pose_trans: float | None = None
    cross_agent_pose_rot: float | None = None
    # Intra-agent rig consistency (single/coop). Average error of relative transforms
    # within each agent (camera0 -> camera1/2/3). Useful to track whether we are
    # wasting capacity on "known rig" DOFs.
    intra_agent_rig_pose_trans: float | None = None
    intra_agent_rig_pose_rot: float | None = None
    pose_ate: float | None = None
    pose_ate_rot: float | None = None
    scale_err: float | None = None
    scale_log_err: float | None = None
    scale_eq_rel_err: float | None = None
    scale_ratio_mean: float | None = None
    scale_ratio_median: float | None = None
    scale_ratio_p90: float | None = None
    scale_gt_factor: float | None = None
    scale_to_gt_err: float | None = None
    scale_to_gt_log_err: float | None = None
    scale_to_gt_eq_rel_err: float | None = None
    scale_to_gt_ratio_mean: float | None = None
    scale_to_gt_ratio_median: float | None = None
    scale_to_gt_ratio_p90: float | None = None
    chamfer_pred_to_gt: float | None = None
    chamfer_gt_to_pred: float | None = None
    chamfer_filtered_pred_to_gt: float | None = None
    chamfer_filtered_gt_to_pred: float | None = None
    bev_iou_raw: float | None = None
    bev_iou_filtered: float | None = None
    det_num_gt: int | None = None
    det_num_pred: int | None = None
    det_tp_iou: int | None = None
    det_fp_iou: int | None = None
    det_fn_iou: int | None = None
    det_mean_iou: float | None = None
    det_ap_iou: float | None = None
    det_precision_iou: float | None = None
    det_recall_iou: float | None = None


@dataclass
class PCMetricConfig:
    enabled: bool = False
    z_min: float | None = None
    z_max: float | None = None
    radius_max: float | None = None
    bev_range: float = 120.0
    bev_resolution: float = 0.5
    save_dir: Path | None = None


@dataclass
class DetMetricConfig:
    enabled: bool = False
    score_thresh: float = 0.3
    nms_iou: float = 0.1
    max_dets: int = 100
    iou_thresh: float = 0.5
    min_density: float | None = None
    min_high_ratio: float | None = None
    min_var_z: float | None = None
    bbox_range: float | None = 120.0
    max_num_boxes: int = 128
    x_range: Tuple[float, float] = (0.0, 120.0)
    y_range: Tuple[float, float] = (-50.0, 50.0)
    voxel_size: float = 0.5


DEFAULT_MODELS: Dict[str, str] = {
    "pretrain": str(REPO_ROOT / "checkpoints" / "facebook_map-anything-v1.pth"),
    "stage1": str(REPO_ROOT / "experiments" / "opv2v_ft_stage1" / "checkpoint-best.pth"),
    "stage2": str(REPO_ROOT / "experiments" / "opv2v_coop_stage2" / "checkpoint-best.pth"),
}


def resolve_data_norm_type(model_arch: str, data_norm_type: str | None) -> str:
    """Resolve the actual normalization type used by preprocess_inputs.

    We record this value in eval summaries for strict fairness checks.
    """
    norm_type = data_norm_type
    if not norm_type:
        norm_type = "identity" if model_arch == "vggt" else "dinov2"
    return norm_type


def predictions_to_ego_pointcloud(
    predictions: Sequence[dict],
    gt_ref_pose_c2e_cv: np.ndarray,
    *,
    colorize: bool = False,
    gt_pose_c2e_cv_list: Sequence[np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Convert model predictions to a point cloud in the ego (LiDAR) frame.

    MapAnything predicts per-view camera poses with an arbitrary gauge. For OPV2V we want
    point clouds aligned to the ego LiDAR coordinate frame stored in GT `.pcd` files.

    Strategy:
      1) Gauge-fix predicted poses by converting them to transforms into the reference
         view-0 camera frame: T(C_ref <- C_i) = inv(T(W <- C_ref)) @ T(W <- C_i).
      2) Convert each view's depth map into the reference camera frame using the gauge-fixed pose.
      3) Apply GT extrinsics for the reference camera (T(E <- C_ref)) to bring points into ego.
      4) Stack all views.
    """

    if not predictions:
        raise ValueError("Empty predictions sequence")
    gt_ref_pose_c2e_cv = np.asarray(gt_ref_pose_c2e_cv, dtype=np.float32)
    if gt_ref_pose_c2e_cv.shape != (4, 4):
        raise ValueError(f"gt_ref_pose_c2e_cv must be 4x4; got {gt_ref_pose_c2e_cv.shape}")
    use_gt_poses = gt_pose_c2e_cv_list is not None
    if use_gt_poses:
        if len(gt_pose_c2e_cv_list) < len(predictions):
            raise ValueError(
                f"gt_pose_c2e_cv_list too short: {len(gt_pose_c2e_cv_list)} < {len(predictions)}"
            )
        gt_ref_pose = torch.tensor(gt_ref_pose_c2e_cv, dtype=torch.float32)
        gt_ref_inv = torch.linalg.inv(gt_ref_pose)
    else:
        pred_ref_pose = predictions[0].get("camera_poses")
        if pred_ref_pose is None:
            raise ValueError("Predictions missing 'camera_poses' for reference view")
        pred_ref_pose = pred_ref_pose[0]
        pred_ref_inv = torch.linalg.inv(pred_ref_pose)

    point_list: List[np.ndarray] = []
    color_list: List[np.ndarray] = []
    use_colors = colorize

    for view_idx, pred in enumerate(predictions):
        depth = pred["depth_z"][0].squeeze(-1)
        intrinsics = pred["intrinsics"][0]
        if use_gt_poses:
            gt_pose = np.asarray(gt_pose_c2e_cv_list[view_idx], dtype=np.float32)
            if gt_pose.shape != (4, 4):
                raise ValueError(f"gt_pose_c2e_cv_list[{view_idx}] must be 4x4; got {gt_pose.shape}")
            gt_pose_t = torch.tensor(gt_pose, dtype=torch.float32, device=depth.device)
            rel_pose = gt_ref_inv.to(depth.device) @ gt_pose_t
        else:
            pose = pred.get("camera_poses")
            if pose is None:
                continue
            pose = pose[0]
            rel_pose = pred_ref_inv @ pose
        pts_ref, valid_mask = depthmap_to_world_frame(depth, intrinsics, rel_pose)

        valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        mask_np = pred["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
        final_mask = mask_np & valid_mask_np

        pts_np = pts_ref.detach().cpu().numpy()
        pts_np = pts_np[final_mask]

        pts_h = np.concatenate(
            [pts_np.astype(np.float32, copy=False), np.ones((pts_np.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        pts_ego = (gt_ref_pose_c2e_cv @ pts_h.T).T[:, :3]
        point_list.append(pts_ego)

        if use_colors:
            colors = pred.get("img_no_norm")
            if colors is None:
                use_colors = False
            else:
                colors_np = colors[0].detach().cpu().numpy()
                color_list.append(colors_np[final_mask])

    if not point_list:
        raise ValueError("No valid prediction entries for point cloud conversion")

    points = np.concatenate(point_list, axis=0)
    if use_colors and color_list:
        colors = np.concatenate(color_list, axis=0)
        colors = np.clip(colors, 0.0, 1.0)
        return points, colors
    return points, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="Dataset split (train/validate/test)")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of frames to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--frames_json", type=Path, help="Optional JSON containing explicit frames (uses 'frames' key)")
    parser.add_argument(
        "--gt_scale_contract_json",
        type=Path,
        default=None,
        help="Optional JSON produced by `scripts/compute_opv2v_gt_scale_contract.py` that stores per-frame "
        "GT scale factors (`gt_scale_single/coop`). When provided, batch_eval will validate the contract hash "
        "matches the evaluated frames and use the cached GT scale instead of recomputing it from depth maps "
        "(significant speedup for large contracts).",
    )
    parser.add_argument("--images_root", type=Path, default=REPO_ROOT / "data" / "opv2v")
    parser.add_argument("--depth_root", type=Path, default=REPO_ROOT / "data" / "opv2v_depth")
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "eval_runs" / "opv2v_batch_eval")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model_task", default="images_only", help="Hydra override for model/task (e.g., images_only, posed_sfm)")
    parser.add_argument(
        "--model_arch",
        default="mapanything",
        choices=("mapanything", "vggt"),
        help="Which model family to initialize. 'mapanything' uses model.infer(); 'vggt' uses forward().",
    )
    parser.add_argument(
        "--data_norm_type",
        default=None,
        help="Override image normalization type passed to preprocess_inputs (e.g., dinov2, identity). "
        "If omitted, defaults to dinov2 for mapanything and identity for vggt.",
    )
    parser.add_argument(
        "--vggt_enable_metric_scale_head",
        action="store_true",
        help="Enable VGGT metric scale head at eval time (only if the checkpoint includes it).",
    )
    parser.add_argument(
        "--vggt_autocast_dtype",
        default=None,
        help="Optional override for VGGT autocast_dtype (auto|bf16|fp16|fp32).",
    )
    parser.add_argument("--models", nargs="*", help="Optional custom model mappings name=checkpoint")
    parser.add_argument("--model_filter", nargs="*", help="Only evaluate the specified model keys (e.g., stage2)")
    parser.add_argument("--modes", nargs="*", choices=("single", "coop"), default=("single", "coop"), help="Which modes to evaluate")
    parser.add_argument("--save_representative", action="store_true", help="Save PCD + stats for best/worst frames per model")
    parser.add_argument("--pc_metrics", action="store_true", help="Compute Chamfer / BEV detection-side metrics")
    parser.add_argument("--pc_filter_z_min", type=float, default=None, help="Z-min threshold for filtered Chamfer/IoU")
    parser.add_argument("--pc_filter_z_max", type=float, default=None, help="Z-max threshold for filtered Chamfer/IoU")
    parser.add_argument("--pc_filter_radius", type=float, default=None, help="Radius threshold for filtered Chamfer/IoU")
    parser.add_argument("--pc_bev_range", type=float, default=120.0, help="BEV grid range for detection metrics")
    parser.add_argument("--pc_bev_resolution", type=float, default=0.5, help="BEV grid resolution for detection metrics")
    parser.add_argument("--pc_save_dir", type=Path, help="Optional directory to store per-frame predicted point clouds (.npy)")
    parser.add_argument("--det_metrics", action="store_true", help="Compute BEV 3D box detection metrics (AP/precision/recall)")
    parser.add_argument(
        "--save_det_cache",
        action="store_true",
        help=(
            "When --det_metrics is enabled, save a compact det cache (.npz) with per-pred TP flags "
            "so you can re-score AP on arbitrary subsets (sample-size sensitivity, baseline buckets) "
            "without re-running inference."
        ),
    )
    parser.add_argument("--det_score_thresh", type=float, default=0.3, help="Score threshold for decoded boxes")
    parser.add_argument("--det_nms_iou", type=float, default=0.1, help="NMS IoU threshold (BEV oriented)")
    parser.add_argument("--det_max_dets", type=int, default=100, help="Max decoded boxes per frame after NMS")
    parser.add_argument("--det_iou_thresh", type=float, default=0.5, help="IoU threshold for TP/FP matching")
    parser.add_argument("--det_min_density", type=float, default=None, help="Min BEV density at decoded peak (optional)")
    parser.add_argument("--det_min_high_ratio", type=float, default=None, help="Min BEV high-z ratio at decoded peak (optional)")
    parser.add_argument("--det_min_var_z", type=float, default=None, help="Min BEV height variance at decoded peak (optional)")
    parser.add_argument("--det_bbox_range", type=float, default=120.0, help="Max XY range (meters) for GT boxes")
    parser.add_argument("--det_max_num_boxes", type=int, default=128, help="Max number of GT boxes per frame")
    parser.add_argument("--det_x_min", type=float, default=0.0, help="Detection grid x_min (meters)")
    parser.add_argument("--det_x_max", type=float, default=120.0, help="Detection grid x_max (meters)")
    parser.add_argument("--det_y_min", type=float, default=-50.0, help="Detection grid y_min (meters)")
    parser.add_argument("--det_y_max", type=float, default=50.0, help="Detection grid y_max (meters)")
    parser.add_argument("--det_voxel_size", type=float, default=0.5, help="Detection grid voxel size (meters)")
    parser.add_argument(
        "--det_head_cfg",
        default="bev_centernet",
        help="Hydra config name under `model/det_head` (e.g., bev_centernet_wide).",
    )
    parser.add_argument(
        "--det_head_ckpt",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint path used to load ONLY det_head weights (composed eval). "
            "Useful when you want to sweep geometry checkpoints while keeping a fixed det head."
        ),
    )
    parser.add_argument("--keep_camera_poses", action="store_true", help="Do not strip GT camera poses from model inputs")
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Print a progress line every N (frame,mode) eval steps (0 disables).",
    )
    parser.add_argument(
        "--keep_main_agent_poses",
        action="store_true",
        help=(
            "Deployment-like coop option: when NOT using --keep_camera_poses, keep ONLY main-agent "
            "camera poses (rig extrinsics) as inputs, while stripping cross-agent poses. "
            "This anchors the world/ego frame for downstream det AP without leaking cross-agent GT."
        ),
    )
    return parser.parse_args()


def _move_views_to_device(views: Sequence[dict], device: torch.device) -> None:
    """Move per-view tensors onto `device` in-place (VGGT forward requires device tensors)."""
    for view in views:
        for k, v in list(view.items()):
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device, non_blocking=True)
            elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(t, torch.Tensor) for t in v):
                view[k] = (v[0].to(device, non_blocking=True), v[1].to(device, non_blocking=True))


def _vggt_preds_to_batch_eval_format(
    preds: Sequence[dict],
    processed_views: Sequence[dict],
) -> List[dict]:
    """Convert VGGTWrapper outputs to the subset of keys expected by batch_eval metrics."""
    out: List[dict] = []
    for pred, view in zip(preds, processed_views):
        pts3d_cam = pred.get("pts3d_cam")
        cam_quats = pred.get("cam_quats")
        cam_trans = pred.get("cam_trans")
        if pts3d_cam is None or cam_quats is None or cam_trans is None:
            out.append({})
            continue

        # Depth z in camera frame (B,H,W,1).
        depth_z = pts3d_cam[..., 2:3]

        # Camera pose (cam2world) as a 4x4 matrix (B,4,4).
        rot = quaternion_to_rotation_matrix(cam_quats)  # (B,3,3), quat order XYZW
        batch = int(rot.shape[0])
        pose = torch.eye(4, dtype=rot.dtype, device=rot.device).unsqueeze(0).repeat(batch, 1, 1)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = cam_trans.to(dtype=rot.dtype)

        mask = torch.isfinite(depth_z) & (depth_z > 0)

        rec: dict = {
            "depth_z": depth_z,
            "camera_poses": pose,
            "mask": mask,
        }
        # Optional: when using a VGGT wrapper that also runs a det head (vggt_det),
        # propagate the det output so batch_eval can decode AP/PR from the same pipeline.
        if "bev_det" in pred:
            rec["bev_det"] = pred.get("bev_det")
        intr = view.get("intrinsics")
        if isinstance(intr, torch.Tensor):
            rec["intrinsics"] = intr
        scale = pred.get("metric_scaling_factor")
        if scale is not None:
            rec["metric_scaling_factor"] = scale
        out.append(rec)
    return out


def _load_gt_scale_contract(
    contract_json: Path,
) -> tuple[
    str | None,
    str | None,
    dict[tuple[str, str], tuple[float | None, float | None]],
]:
    """Load a GT-scale contract into a (sequence, frame) -> (single, coop) lookup.

    The contract JSON is expected to be produced by `compute_opv2v_gt_scale_contract.py` and contain:
      - frames_hash_md5: legacy md5 over newline-joined, sorted unique `sequence/frame` items
      - frames_hash_md5_v2 (optional): fair md5 that also includes `main_agent` and `coop_agents`
      - frames: list of per-frame dicts with `sequence`, `frame`, `gt_scale_single`, `gt_scale_coop`
    """
    doc = json.loads(contract_json.read_text())
    contract_hash_v1 = doc.get("frames_hash_md5")
    contract_hash_v2 = doc.get("frames_hash_md5_v2")
    lookup: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    frames = doc.get("frames") or []
    if not isinstance(frames, list):
        return (
            str(contract_hash_v1) if contract_hash_v1 is not None else None,
            str(contract_hash_v2) if contract_hash_v2 is not None else None,
            lookup,
        )

    for fr in frames:
        if not isinstance(fr, dict):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        if not isinstance(seq, str) or not isinstance(frame, str):
            continue
        s = fr.get("gt_scale_single")
        c = fr.get("gt_scale_coop")
        s_val = float(s) if isinstance(s, (int, float)) and math.isfinite(float(s)) else None
        c_val = float(c) if isinstance(c, (int, float)) and math.isfinite(float(c)) else None
        key = (seq, frame)
        prev = lookup.get(key)
        if prev is not None and prev != (s_val, c_val):
            raise ValueError(
                f"Duplicate gt-scale entries with different values for {seq}/{frame}: {prev} vs {(s_val, c_val)}"
            )
        lookup[key] = (s_val, c_val)

    return (
        str(contract_hash_v1) if contract_hash_v1 is not None else None,
        str(contract_hash_v2) if contract_hash_v2 is not None else None,
        lookup,
    )


def discover_frames(images_root: Path, split: str) -> List[FrameInfo]:
    """Discover candidate frames for sampling.

    Prefer the OPV2V parquet index when available so pairing matches the default
    dataset behavior (`pair_agent_policy='nearest'`) instead of the legacy "first
    two agent IDs" heuristic. Fall back to directory scan when the index is
    missing or pandas is unavailable.
    """

    split_root = images_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory missing: {split_root}")

    index_file = images_root / "opv2v_index" / f"index_{split}.parquet"
    if index_file.is_file():
        try:
            import pandas as pd  # type: ignore

            want_cols = ["sequence", "frame", "main_agent", "pair_agent", "agent_has_assets"]
            try:
                df = pd.read_parquet(index_file, columns=want_cols)
            except Exception:
                df = pd.read_parquet(index_file)

            frames: List[FrameInfo] = []
            for row in df.itertuples(index=False):
                seq = getattr(row, "sequence", None)
                fr = getattr(row, "frame", None)
                main = getattr(row, "main_agent", None)
                pair = getattr(row, "pair_agent", None)
                if seq is None or fr is None or main is None or pair is None:
                    continue
                main_s = str(main)
                pair_s = str(pair)
                if main_s == pair_s:
                    continue

                has_assets = getattr(row, "agent_has_assets", None)
                if isinstance(has_assets, dict):
                    if not (has_assets.get(main_s, True) and has_assets.get(pair_s, True)):
                        continue

                frames.append(
                    FrameInfo(
                        sequence=str(seq),
                        frame=str(fr),
                        main_agent=main_s,
                        coop_agents=(main_s, pair_s),
                    )
                )

            if frames:
                return frames
        except Exception:
            # Fall back to YAML scan below.
            pass

    # Legacy fallback: directory scan + first two agent IDs.
    frames2: List[FrameInfo] = []
    for seq_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        agent_dirs = sorted([d.name for d in seq_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if len(agent_dirs) < 2:
            continue
        main_agent = agent_dirs[0]
        coop_agents = tuple(agent_dirs[:2])
        yaml_files = sorted((seq_dir / main_agent).glob("*.yaml"))
        for yaml_path in yaml_files:
            frame = yaml_path.stem
            if not frame.isdigit():
                continue
            if not all((seq_dir / agent / f"{frame}.yaml").is_file() for agent in coop_agents):
                continue
            frames2.append(
                FrameInfo(sequence=seq_dir.name, frame=frame, main_agent=main_agent, coop_agents=coop_agents)
            )
    return frames2


def sample_frames(all_frames: List[FrameInfo], sample_size: int, seed: int) -> List[FrameInfo]:
    if len(all_frames) <= sample_size:
        return all_frames
    random.Random(seed).shuffle(all_frames)
    return sorted(all_frames[:sample_size], key=lambda x: (x.sequence, x.frame))


def load_frames_from_json(json_path: Path) -> List[FrameInfo]:
    with json_path.expanduser().open() as fh:
        data = json.load(fh)
    frame_entries = data.get("frames", data)
    if not isinstance(frame_entries, list):
        raise ValueError(f"{json_path} must contain a list under 'frames'; got {type(frame_entries)}")
    frames: List[FrameInfo] = []
    for item in frame_entries:
        try:
            dist_raw = item.get("baseline_distance_m")
            dist = float(dist_raw) if isinstance(dist_raw, (int, float)) and math.isfinite(float(dist_raw)) else None
            bucket_raw = item.get("baseline_bucket")
            bucket = str(bucket_raw) if isinstance(bucket_raw, str) and bucket_raw else None
            frames.append(
                FrameInfo(
                    sequence=item["sequence"],
                    frame=item["frame"],
                    main_agent=item.get("main_agent", "641"),
                    coop_agents=tuple(item.get("coop_agents", (item.get("main_agent", "641"),))),
                    baseline_distance_m=dist,
                    baseline_bucket=bucket,
                )
            )
        except KeyError as exc:
            raise ValueError(f"Invalid frame entry in {json_path}: missing {exc}") from exc
    return frames


def _load_depth(depth_root: Path, split: str, sequence: str, agent: str, frame: str, cam_key: str) -> np.ndarray:
    depth_path = depth_root / split / sequence / agent / f"{frame}_{cam_key}_depth.npy"
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing depth map: {depth_path}")
    depth = np.load(depth_path).astype(np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def build_single_raw(images_root: Path, depth_root: Path, split: str, info: FrameInfo) -> Tuple[List[Dict], List[Dict]]:
    agent = info.main_agent
    yaml_path = images_root / split / info.sequence / agent / f"{info.frame}.yaml"
    meta = load_frame_metadata(yaml_path)
    cam_poses_ego = get_camera_poses_in_ego(meta)
    raw_views: List[Dict] = []
    camera_infos: List[Dict] = []
    for cam_key in sorted(cam_poses_ego.keys()):
        img_path = yaml_path.parent / f"{info.frame}_{cam_key}.png"
        if not img_path.is_file():
            continue
        img = Image.open(img_path).convert("RGB")
        depth = _load_depth(depth_root, split, info.sequence, agent, info.frame, cam_key)
        intr = torch.tensor(meta[cam_key]["intrinsic"], dtype=torch.float32)
        intr = rescale_intrinsics_to_image(intr, *img.size)
        pose_ego = cam_poses_ego[cam_key].astype(np.float32)
        pose_ego_cv = _convert_pose_to_opencv(pose_ego)
        pose_world = cords_to_pose(meta[cam_key]["cords"])
        pose_world_cv = _convert_pose_to_opencv(pose_world)
        raw_views.append(
            {
                "img": img,
                "depth_z": depth,
                "intrinsics": intr.numpy(),
                "camera_poses": pose_ego_cv.astype(np.float32),
            }
        )
        camera_infos.append(
            {
                "name": cam_key,
                "pose_C2W_cv": torch.tensor(pose_world_cv, dtype=torch.float32),
                "pose_C2E_cv": torch.tensor(pose_ego_cv, dtype=torch.float32),
            }
        )
    if not raw_views:
        raise RuntimeError(f"No valid camera views for {info.sequence}/{agent}/{info.frame}")
    return raw_views, camera_infos


def build_coop_raw(images_root: Path, depth_root: Path, split: str, info: FrameInfo) -> Tuple[List[Dict], List[Dict]]:
    sequence_dir = images_root / split / info.sequence
    depth_split = depth_root / split / info.sequence
    metadata = {}
    for agent in info.coop_agents:
        yaml_path = sequence_dir / agent / f"{info.frame}.yaml"
        metadata[agent] = load_frame_metadata(yaml_path)
    main_meta = metadata[info.main_agent]
    T_world_main = cords_to_pose(main_meta["lidar_pose"])
    T_main_world = np.linalg.inv(T_world_main)
    raw_views: List[Dict] = []
    camera_infos: List[Dict] = []
    for agent in info.coop_agents:
        meta = metadata[agent]
        for cam_key in sorted(k for k in meta.keys() if k.startswith("camera")):
            img_path = sequence_dir / agent / f"{info.frame}_{cam_key}.png"
            if not img_path.is_file():
                continue
            img = Image.open(img_path).convert("RGB")
            depth = _load_depth(depth_root, split, info.sequence, agent, info.frame, cam_key)
            intr = torch.tensor(meta[cam_key]["intrinsic"], dtype=torch.float32)
            intr = rescale_intrinsics_to_image(intr, *img.size)
            cam_pose_world = cords_to_pose(meta[cam_key]["cords"])
            cam_pose_main = T_main_world @ cam_pose_world
            pose_world_cv = _convert_pose_to_opencv(cam_pose_world)
            pose_main_cv = _convert_pose_to_opencv(cam_pose_main)
            raw_views.append(
                {
                    "img": img,
                    "depth_z": depth,
                    "intrinsics": intr.numpy(),
                    "camera_poses": pose_main_cv.astype(np.float32),
                }
            )
            camera_infos.append(
                {
                    "name": f"{cam_key}_{agent}",
                    "pose_C2W_cv": torch.tensor(pose_world_cv, dtype=torch.float32),
                    "pose_C2E_cv": torch.tensor(pose_main_cv, dtype=torch.float32),
                }
            )
    if not raw_views:
        raise RuntimeError(f"No cooperative views for {info.sequence}/{info.frame}")
    return raw_views, camera_infos


def _compute_pose_metrics(predictions, camera_info_list) -> Tuple[float, float, float, float]:
    if not camera_info_list or not predictions:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Cooperative inputs carry both poses; prefer ego-frame pose for consistent metric evaluation.
    pose_key = "pose_C2E_cv" if "pose_C2E_cv" in camera_info_list[0] else "pose_C2W_cv"
    gt_ref = camera_info_list[0][pose_key].cpu().numpy()
    gt_ref_inv = np.linalg.inv(gt_ref)
    pred_ref = None
    if predictions:
        ref = predictions[0].get("camera_poses")
        if ref is not None:
            pred_ref = ref[0].detach().cpu().numpy()
            pred_ref = np.linalg.inv(pred_ref)
    abs_list = []
    rot_list = []
    gt_poses = []
    pred_poses = []
    for idx, cam in enumerate(camera_info_list):
        if idx >= len(predictions):
            break
        pred_pose_tensor = predictions[idx].get("camera_poses")
        if pred_pose_tensor is None:
            continue
        pred_pose = pred_pose_tensor[0].detach().cpu().numpy()
        gt_pose = cam[pose_key].cpu().numpy()
        gt_rel = gt_ref_inv @ gt_pose
        pred_rel = pred_pose if pred_ref is None else pred_ref @ pred_pose
        metric = _pose_error(pred_rel, gt_rel)
        abs_list.append(metric["abs_trans"])
        rot_list.append(metric["abs_rot_deg"])
        gt_poses.append(gt_pose)
        pred_poses.append(pred_pose)
    if not abs_list:
        return float("nan"), float("nan"), float("nan"), float("nan")

    pose_ate = float("nan")
    pose_ate_rot = float("nan")
    if len(gt_poses) >= 2:
        src = np.stack([p[:3, 3] for p in pred_poses], axis=0)
        dst = np.stack([g[:3, 3] for g in gt_poses], axis=0)
        R_align, t_align = _rigid_align_umeyama(src, dst)
        ate_trans = []
        ate_rot = []
        for pred_pose, gt_pose in zip(pred_poses, gt_poses):
            pred_aligned = pred_pose.copy()
            pred_aligned[:3, :3] = R_align @ pred_pose[:3, :3]
            pred_aligned[:3, 3] = R_align @ pred_pose[:3, 3] + t_align
            metric = _pose_error(pred_aligned, gt_pose)
            ate_trans.append(metric["abs_trans"])
            ate_rot.append(metric["abs_rot_deg"])
        if ate_trans:
            pose_ate = float(np.mean(ate_trans))
            pose_ate_rot = float(np.mean(ate_rot))

    return float(np.mean(abs_list)), float(np.mean(rot_list)), pose_ate, pose_ate_rot


def _parse_agent_from_view_name(name: str | None) -> str | None:
    if not name:
        return None
    parts = str(name).split("_")
    if len(parts) < 2:
        # Single-agent naming uses plain "camera0..3" without an agent suffix.
        return "__single__"
    return parts[-1]


def _compute_cross_agent_pose_metrics(predictions, camera_info_list) -> tuple[float, float]:
    """Cross-agent relative pose error (main camera0 <-> each coop agent's camera0).

    For 2-agent OPV2V contracts this reduces to the original definition.
    For N-agent cooperative samples, we report the *worst-case* (max) error across
    all non-main agents to better match `baseline_distance_m = max_i d_i`.
    """
    if not camera_info_list or not predictions:
        return float("nan"), float("nan")

    main_agent = _parse_agent_from_view_name(camera_info_list[0].get("name"))
    idx_main: int | None = None
    idx_pairs: list[int] = []
    for idx, cam in enumerate(camera_info_list):
        name = cam.get("name")
        if not isinstance(name, str) or not name.startswith("camera0_"):
            continue
        agent = _parse_agent_from_view_name(name)
        if idx_main is None and (main_agent is None or agent == main_agent):
            idx_main = idx
            continue
        if idx_main is None and main_agent is None:
            # Fallback: first camera0 becomes the main reference when agent ids are unavailable.
            idx_main = idx
            continue
        if idx_main is not None and (main_agent is None or agent != main_agent):
            idx_pairs.append(idx)

    if idx_main is None or not idx_pairs:
        return float("nan"), float("nan")
    if idx_main >= len(predictions) or any(i >= len(predictions) for i in idx_pairs):
        return float("nan"), float("nan")

    pose_key = (
        "pose_C2E_cv" if "pose_C2E_cv" in camera_info_list[0] else "pose_C2W_cv"
    )
    try:
        gt_main = camera_info_list[idx_main][pose_key].cpu().numpy()
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")

    pred_main = predictions[idx_main].get("camera_poses")
    if pred_main is None:
        return float("nan"), float("nan")
    pred_main = pred_main[0].detach().cpu().numpy()

    trans_errs: list[float] = []
    rot_errs: list[float] = []
    for idx_pair in idx_pairs:
        try:
            gt_pair = camera_info_list[idx_pair][pose_key].cpu().numpy()
            gt_rel = np.linalg.inv(gt_main) @ gt_pair
        except Exception:  # noqa: BLE001
            continue

        pred_pair = predictions[idx_pair].get("camera_poses")
        if pred_pair is None:
            continue
        pred_pair = pred_pair[0].detach().cpu().numpy()
        pred_rel = np.linalg.inv(pred_main) @ pred_pair
        metric = _pose_error(pred_rel, gt_rel)
        trans_errs.append(float(metric["abs_trans"]))
        rot_errs.append(float(metric["abs_rot_deg"]))

    if not trans_errs:
        return float("nan"), float("nan")
    return float(max(trans_errs)), float(max(rot_errs))


def _compute_intra_agent_rig_metrics(predictions, camera_info_list) -> tuple[float, float]:
    """Mean relative-pose error inside each agent (camera0 -> camera1/2/3)."""
    if not camera_info_list or not predictions:
        return float("nan"), float("nan")

    pose_key = (
        "pose_C2E_cv" if "pose_C2E_cv" in camera_info_list[0] else "pose_C2W_cv"
    )

    # Group view indices by agent id inferred from the view name.
    agent_to_indices: dict[str, list[int]] = {}
    for idx, cam in enumerate(camera_info_list):
        name = cam.get("name")
        if not isinstance(name, str):
            continue
        agent = _parse_agent_from_view_name(name)
        if agent is None:
            continue
        agent_to_indices.setdefault(agent, []).append(idx)

    trans_errs: list[float] = []
    rot_errs: list[float] = []
    for agent, indices in agent_to_indices.items():
        # Find the agent's camera0 as the local reference.
        idx0 = None
        for idx in indices:
            name = camera_info_list[idx].get("name")
            if isinstance(name, str) and (name == "camera0" or name.startswith("camera0_")):
                idx0 = idx
                break
        if idx0 is None or idx0 >= len(predictions):
            continue
        pred0 = predictions[idx0].get("camera_poses")
        if pred0 is None:
            continue
        pred0 = pred0[0].detach().cpu().numpy()
        gt0 = camera_info_list[idx0].get(pose_key)
        if gt0 is None:
            continue
        gt0 = gt0.cpu().numpy()

        for idx in indices:
            if idx == idx0 or idx >= len(predictions):
                continue
            name = camera_info_list[idx].get("name")
            if not isinstance(name, str):
                continue
            if not (
                name in ("camera1", "camera2", "camera3")
                or name.startswith(("camera1_", "camera2_", "camera3_"))
            ):
                continue
            pred = predictions[idx].get("camera_poses")
            if pred is None:
                continue
            pred = pred[0].detach().cpu().numpy()
            gt = camera_info_list[idx].get(pose_key)
            if gt is None:
                continue
            gt = gt.cpu().numpy()

            pred_rel = np.linalg.inv(pred0) @ pred
            gt_rel = np.linalg.inv(gt0) @ gt
            metric = _pose_error(pred_rel, gt_rel)
            trans_errs.append(float(metric["abs_trans"]))
            rot_errs.append(float(metric["abs_rot_deg"]))

    if not trans_errs:
        return float("nan"), float("nan")
    return float(np.mean(trans_errs)), float(np.mean(rot_errs))


def _rigid_align_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (no scale) that maps src -> dst."""
    if src.shape[0] < 2:
        raise ValueError("Need at least 2 points for rigid alignment.")
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (src_c.T @ dst_c) / float(src.shape[0])
    U, _, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def _pose_error(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, float]:
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]
    abs_trans = float(np.linalg.norm(pred_t - gt_t))
    trace_val = np.clip((np.trace(gt_R.T @ pred_R) - 1.0) / 2.0, -1.0, 1.0)
    abs_rot = math.degrees(math.acos(trace_val))
    return {"abs_trans": abs_trans, "abs_rot_deg": abs_rot}


def depth_metrics(predictions, gt_depths: List[torch.Tensor]) -> Tuple[float, float, float]:
    rmse_vals = []
    mae_vals = []
    rel_vals = []
    for pred, gt in zip(predictions, gt_depths):
        pred_depth = pred["depth_z"][0].squeeze(-1).detach().cpu()
        gt_depth = gt.squeeze(0).detach().cpu()
        mask = gt_depth > 0
        if mask.sum() == 0:
            continue
        diff = pred_depth[mask] - gt_depth[mask]
        rmse_vals.append(torch.sqrt((diff**2).mean()).item())
        mae_vals.append(torch.mean(diff.abs()).item())
        rel_vals.append(torch.mean(diff.abs() / torch.clamp(gt_depth[mask], min=1e-3)).item())
    if not rmse_vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(rmse_vals)), float(np.mean(mae_vals)), float(np.mean(rel_vals))


def _gt_scale_factor_from_raw_views(raw_views: Sequence[Dict[str, Any]]) -> float | None:
    """Compute GT normalization factor used by `norm_mode=avg_dis`.

    We aggregate all valid depth pixels from all views after transforming points into
    the reference view0 camera frame, then compute mean Euclidean distance to origin.

    This is the ground-truth metric scale target implied by
    `normalize_multiple_pointclouds(..., norm_mode=avg_dis)` in training losses.
    """

    if not raw_views:
        return None
    if "camera_poses" not in raw_views[0]:
        return None

    pose_ref = np.asarray(raw_views[0]["camera_poses"], dtype=np.float64)
    if pose_ref.shape != (4, 4):
        return None
    pose_ref_inv = np.linalg.inv(pose_ref)

    total_sum = 0.0
    total_cnt = 0
    for rv in raw_views:
        depth = rv.get("depth_z")
        intr = rv.get("intrinsics")
        pose = rv.get("camera_poses")
        if depth is None or intr is None or pose is None:
            continue

        depth = np.asarray(depth, dtype=np.float64)
        intr = np.asarray(intr, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or intr.shape != (3, 3):
            continue

        T_ref_cur = pose_ref_inv @ pose
        R = T_ref_cur[:3, :3]
        t = T_ref_cur[:3, 3]

        valid = depth > 0.0
        v, u = np.nonzero(valid)
        if v.size == 0:
            continue

        z = depth[v, u]
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts = np.stack([x, y, z], axis=0)
        pts_ref = (R @ pts) + t.reshape(3, 1)
        dis = np.linalg.norm(pts_ref, axis=0)
        total_sum += float(dis.sum())
        total_cnt += int(dis.size)

    if total_cnt <= 0:
        return None
    return total_sum / float(total_cnt)


def _pred_scale_factor_from_predictions(
    predictions: Sequence[dict],
    gt_depths: Sequence[torch.Tensor],
) -> float | None:
    """Compute the same `avg_dis` normalization factor from *predicted* geometry.

    This is used as a fallback for models that do not output `metric_scaling_factor`
    (e.g. VGGT when `enable_metric_scale_head=false`).

    For efficiency and comparability, we compute the statistic only on pixels that
    are valid in the OPV2V GT sparse depth maps (`gt_depth > 0`).
    """
    if not predictions or not gt_depths:
        return None
    if len(gt_depths) < len(predictions):
        return None

    ref = predictions[0].get("camera_poses")
    if ref is None:
        return None
    try:
        pose_ref = np.asarray(ref[0].detach().cpu().numpy(), dtype=np.float64)
        pose_ref_inv = np.linalg.inv(pose_ref)
    except Exception:  # noqa: BLE001
        return None

    total_sum = 0.0
    total_cnt = 0
    for pred, gt in zip(predictions, gt_depths):
        depth_t = pred.get("depth_z")
        intr_t = pred.get("intrinsics")
        pose_t = pred.get("camera_poses")
        if depth_t is None or intr_t is None or pose_t is None:
            continue

        try:
            depth = np.asarray(depth_t[0].squeeze(-1).detach().cpu().numpy(), dtype=np.float64)
            intr = np.asarray(intr_t[0].detach().cpu().numpy(), dtype=np.float64)
            pose = np.asarray(pose_t[0].detach().cpu().numpy(), dtype=np.float64)
            gt_depth = np.asarray(gt.squeeze(0).detach().cpu().numpy(), dtype=np.float64)
        except Exception:  # noqa: BLE001
            continue

        if intr.shape != (3, 3) or pose.shape != (4, 4):
            continue

        try:
            T_ref_cur = pose_ref_inv @ pose
        except Exception:  # noqa: BLE001
            continue
        R = T_ref_cur[:3, :3]
        t = T_ref_cur[:3, 3]

        valid = (gt_depth > 0.0) & np.isfinite(depth) & (depth > 0.0)
        v, u = np.nonzero(valid)
        if v.size == 0:
            continue

        z = depth[v, u]
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        if fx == 0.0 or fy == 0.0:
            continue

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts = np.stack([x, y, z], axis=0)  # (3, N)
        pts_ref = (R @ pts) + t.reshape(3, 1)
        dis = np.linalg.norm(pts_ref, axis=0)
        total_sum += float(dis.sum())
        total_cnt += int(dis.size)

    if total_cnt <= 0:
        return None
    return total_sum / float(total_cnt)


def scale_metric_details(
    predictions,
    *,
    gt_scale_factor: float | None = None,
    fallback_scale_factor: float | None = None,
) -> Dict[str, float] | None:
    """Return scale diagnostics from `metric_scaling_factor`.

    Legacy fields (kept for backward compatibility) treat raw `s` as ratio-to-1:
      - `scale_err`, `scale_log_err`, `scale_eq_rel_err`, `scale_ratio_*`

    Preferred fields compare prediction against GT normalization factor `g`:
      - `scale_gt_factor`
      - `scale_to_gt_err`: mean(|s/g - 1|)
      - `scale_to_gt_log_err`: mean(|log(s/g)|)
      - `scale_to_gt_eq_rel_err`: exp(mean(|log(s/g)|)) - 1
      - `scale_to_gt_ratio_*`: stats on s/g
    """

    vals: List[float] = []
    for pred in predictions:
        scale = pred.get("metric_scaling_factor")
        if scale is None:
            continue
        if isinstance(scale, torch.Tensor):
            vals.append(scale.mean().item())
    used_fallback = False
    if not vals:
        if (
            fallback_scale_factor is not None
            and isinstance(fallback_scale_factor, (int, float))
            and math.isfinite(float(fallback_scale_factor))
            and float(fallback_scale_factor) > 1e-8
        ):
            # No explicit scale head; use the implied normalization factor from predicted geometry.
            # Repeat so the downstream "stats across views" logic still works.
            used_fallback = True
            vals = [float(fallback_scale_factor)] * max(1, len(predictions))
        else:
            return None

    ratios = np.asarray(vals, dtype=np.float64)
    ratios = np.clip(ratios, 1e-8, None)
    out: Dict[str, float] = {}
    if not used_fallback:
        # Legacy ratio-to-1 diagnostics: only meaningful when the model explicitly predicts `s`.
        abs_rel = np.abs(ratios - 1.0)
        abs_log = np.abs(np.log(ratios))
        mean_abs_log = float(np.mean(abs_log))
        out.update(
            {
                "scale_err": float(np.mean(abs_rel)),
                "scale_log_err": mean_abs_log,
                "scale_eq_rel_err": float(np.expm1(mean_abs_log)),
                "scale_ratio_mean": float(np.mean(ratios)),
                "scale_ratio_median": float(np.median(ratios)),
                "scale_ratio_p90": float(np.quantile(ratios, 0.9)),
            }
        )

    if (
        gt_scale_factor is not None
        and isinstance(gt_scale_factor, (int, float))
        and math.isfinite(float(gt_scale_factor))
        and float(gt_scale_factor) > 1e-8
    ):
        g = float(gt_scale_factor)
        ratio_to_gt = ratios / g
        ratio_to_gt = np.clip(ratio_to_gt, 1e-8, None)
        abs_rel_gt = np.abs(ratio_to_gt - 1.0)
        abs_log_gt = np.abs(np.log(ratio_to_gt))
        mean_abs_log_gt = float(np.mean(abs_log_gt))
        out.update(
            {
                "scale_gt_factor": g,
                "scale_to_gt_err": float(np.mean(abs_rel_gt)),
                "scale_to_gt_log_err": mean_abs_log_gt,
                "scale_to_gt_eq_rel_err": float(np.expm1(mean_abs_log_gt)),
                "scale_to_gt_ratio_mean": float(np.mean(ratio_to_gt)),
                "scale_to_gt_ratio_median": float(np.median(ratio_to_gt)),
                "scale_to_gt_ratio_p90": float(np.quantile(ratio_to_gt, 0.9)),
            }
        )

    return out


def scale_metric(predictions) -> float | None:
    details = scale_metric_details(predictions)
    if details is None:
        return None
    return details["scale_err"]


def _points_opencv_to_carla(points_cv: np.ndarray) -> np.ndarray:
    """Convert Nx3 points from OpenCV convention to CARLA/UE convention.

    OPV2V GT LiDAR `.pcd` files are in CARLA/UE coordinates (X forward, Y right, Z up).
    MapAnything point clouds are produced in OpenCV convention (X right, Y down, Z forward).
    """

    if points_cv.size == 0:
        return points_cv
    basis = CARLA_TO_CAMERA_CV[:3, :3].astype(points_cv.dtype, copy=False)
    # Column-vector form: p_cv = basis * p_carla  =>  p_carla = basis^T * p_cv
    return (basis.T @ points_cv.T).T


def _filter_points(
    points: np.ndarray,
    z_min: float | None,
    z_max: float | None,
    radius_max: float | None,
) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    if radius_max is not None:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= radius_max
    return points[mask]


def _chamfer_metrics(pred_points: np.ndarray, gt_points: np.ndarray) -> Tuple[float, float]:
    if pred_points.size == 0 or gt_points.size == 0:
        return float("nan"), float("nan")
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    dist_pred_gt, _ = gt_tree.query(pred_points, k=1)
    dist_gt_pred, _ = pred_tree.query(gt_points, k=1)
    return float(np.mean(dist_pred_gt)), float(np.mean(dist_gt_pred))


def _bev_occupancy(points: np.ndarray, bev_range: float, bev_resolution: float) -> np.ndarray:
    grid_size = int(np.ceil((2 * bev_range) / bev_resolution))
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    if points.size == 0:
        return grid
    coords = points[:, :2]
    mask = np.all(np.abs(coords) <= bev_range, axis=1)
    coords = coords[mask]
    if coords.size == 0:
        return grid
    idx = ((coords + bev_range) / bev_resolution).astype(int)
    idx = np.clip(idx, 0, grid_size - 1)
    grid[idx[:, 0], idx[:, 1]] = True
    return grid


def _bev_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def compute_detection_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    config: PCMetricConfig,
) -> Dict[str, float]:
    def _sample(points: np.ndarray, max_points: int = 200_000) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        return points[idx]

    pred_points_sample = _sample(pred_points)
    gt_points_sample = _sample(gt_points)

    chamfer_pred_gt, chamfer_gt_pred = _chamfer_metrics(pred_points_sample, gt_points_sample)
    bev_pred = _bev_occupancy(pred_points, config.bev_range, config.bev_resolution)
    bev_gt = _bev_occupancy(gt_points, config.bev_range, config.bev_resolution)
    metrics = {
        "chamfer_pred_to_gt": chamfer_pred_gt,
        "chamfer_gt_to_pred": chamfer_gt_pred,
        "bev_iou_raw": _bev_iou(bev_pred, bev_gt),
    }
    if any(v is not None for v in (config.z_min, config.z_max, config.radius_max)):
        filtered = _filter_points(pred_points, config.z_min, config.z_max, config.radius_max)
        filtered_sample = _sample(filtered)
        chamfer_f_pred_gt, chamfer_f_gt_pred = _chamfer_metrics(filtered_sample, gt_points_sample)
        bev_filtered = _bev_occupancy(filtered, config.bev_range, config.bev_resolution)
        metrics.update(
            {
                "chamfer_filtered_pred_to_gt": chamfer_f_pred_gt,
                "chamfer_filtered_gt_to_pred": chamfer_f_gt_pred,
                "bev_iou_filtered": _bev_iou(bev_filtered, bev_gt),
            }
        )
    else:
        metrics.update(
            {
                "chamfer_filtered_pred_to_gt": float("nan"),
                "chamfer_filtered_gt_to_pred": float("nan"),
                "bev_iou_filtered": float("nan"),
            }
        )
    return metrics


def _extract_vehicle_boxes_in_ego(
    frame_meta: Dict, *, max_range: float | None, max_num_boxes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract padded vehicle boxes in ego CARLA/UE convention."""
    vehicles = frame_meta.get("vehicles") or {}
    if not isinstance(vehicles, dict) or len(vehicles) == 0:
        return (
            np.zeros((max_num_boxes, 7), dtype=np.float32),
            np.zeros((max_num_boxes,), dtype=bool),
        )

    T_world_ego = cords_to_pose(frame_meta["lidar_pose"])
    T_ego_world = np.linalg.inv(T_world_ego)

    boxes: list[list[float]] = []
    for veh_data in vehicles.values():
        if not isinstance(veh_data, dict):
            continue
        if "location" not in veh_data or "angle" not in veh_data:
            continue
        if "extent" not in veh_data or "center" not in veh_data:
            continue

        T_world_vehicle = cords_to_pose((*veh_data["location"], *veh_data["angle"]))
        center_local = np.asarray(veh_data["center"], dtype=np.float64)
        extent = np.asarray(veh_data["extent"], dtype=np.float64)

        pose_center = T_world_vehicle.copy()
        pose_center[:3, 3] = (
            T_world_vehicle[:3, 3] + T_world_vehicle[:3, :3] @ center_local
        )
        T_ego_bbox = T_ego_world @ pose_center
        bbox_center = T_ego_bbox[:3, 3].astype(np.float32)

        if max_range is not None:
            if float(np.linalg.norm(bbox_center[:2])) > float(max_range):
                continue

        dims = (2.0 * extent).astype(np.float32)
        rot = T_ego_bbox[:3, :3]
        yaw = float(math.atan2(rot[1, 0], rot[0, 0]))

        boxes.append(
            [
                float(bbox_center[0]),
                float(bbox_center[1]),
                float(bbox_center[2]),
                float(dims[0]),
                float(dims[1]),
                float(dims[2]),
                yaw,
            ]
        )

    num = min(len(boxes), max_num_boxes)
    padded = np.zeros((max_num_boxes, 7), dtype=np.float32)
    mask = np.zeros((max_num_boxes,), dtype=bool)
    if num > 0:
        padded[:num] = np.asarray(boxes[:num], dtype=np.float32)
        mask[:num] = True
    return padded, mask


def _box_corners_bev_xy(box: np.ndarray) -> np.ndarray:
    """Return 4x2 BEV corners (CCW) for box [x,y,z,l,w,h,yaw]."""
    x, y, _, l, w, _, yaw = [float(v) for v in box]
    dx = l * 0.5
    dy = w * 0.5
    corners = np.array(
        [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy],
        ],
        dtype=np.float32,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def _polygon_area(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    x = np.asarray([p[0] for p in poly], dtype=np.float64)
    y = np.asarray([p[1] for p in poly], dtype=np.float64)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(p: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> bool:
    # left-of test for CCW clip polygon
    return float((edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0])) >= 0.0


def _segment_intersection(
    s: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Intersect segment s->e with infinite line a->b."""
    se = e - s
    ab = b - a
    denom = float(se[0] * ab[1] - se[1] * ab[0])
    if abs(denom) < 1e-9:
        return e.copy()
    t = float(((a[0] - s[0]) * ab[1] - (a[1] - s[1]) * ab[0]) / denom)
    return s + t * se


def _convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clipping for convex polygons (Nx2)."""
    output = subject
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        if output.size == 0:
            break
        input_list = output
        output_list: list[np.ndarray] = []
        s = input_list[-1]
        for e in input_list:
            e_in = _inside(e, a, b)
            s_in = _inside(s, a, b)
            if e_in:
                if not s_in:
                    output_list.append(_segment_intersection(s, e, a, b))
                output_list.append(e)
            elif s_in:
                output_list.append(_segment_intersection(s, e, a, b))
            s = e
        output = np.stack(output_list, axis=0) if output_list else np.zeros((0, 2), dtype=np.float32)
    return output


def _oriented_bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    poly_a = _box_corners_bev_xy(box_a)
    poly_b = _box_corners_bev_xy(box_b)
    inter_poly = _convex_clip(poly_a, poly_b)
    inter = _polygon_area(inter_poly.tolist())
    area_a = _polygon_area(poly_a.tolist())
    area_b = _polygon_area(poly_b.tolist())
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def _nms_bev_oriented(
    boxes: List[Dict[str, object]],
    *,
    iou_thresh: float,
    max_dets: int,
) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for item in boxes:
        if len(kept) >= max_dets:
            break
        box = item.get("box")
        if not isinstance(box, np.ndarray) or box.shape != (7,):
            continue
        should_keep = True
        for other in kept:
            other_box = other.get("box")
            if not isinstance(other_box, np.ndarray):
                continue
            if _oriented_bev_iou(box, other_box) >= iou_thresh:
                should_keep = False
                break
        if should_keep:
            kept.append(item)
    return kept


def decode_bev_centernet(
    det_out: Dict[str, torch.Tensor],
    cfg: DetMetricConfig,
) -> List[Dict[str, object]]:
    heat = det_out.get("heatmap")
    reg = det_out.get("reg")
    bev_feat = det_out.get("bev_feat")
    if heat is None or reg is None:
        return []
    if not isinstance(heat, torch.Tensor) or not isinstance(reg, torch.Tensor):
        return []
    if heat.ndim != 4 or reg.ndim != 4:
        return []
    if heat.shape[0] != 1:
        heat = heat[:1]
    if reg.shape[0] != 1:
        reg = reg[:1]

    heat_t = heat[0, 0].detach().float()  # (H, W)
    reg_t = reg[0].detach().float()  # (8, H, W)

    # Prefer local-maximum peak selection (CenterNet-style) to avoid decoding thousands of flat scores.
    pooled = F.max_pool2d(heat_t[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    keep = (heat_t >= float(cfg.score_thresh)) & (heat_t >= pooled)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    if ys.numel() == 0:
        return []

    scores = heat_t[ys, xs]
    pre_nms_topk = int(max(4 * int(cfg.max_dets), 1000))
    if scores.numel() > pre_nms_topk:
        scores, order = torch.topk(scores, k=pre_nms_topk, largest=True, sorted=True)
        ys = ys[order]
        xs = xs[order]
    else:
        scores, order = torch.sort(scores, descending=True)
        ys = ys[order]
        xs = xs[order]

    # Optional point-support filtering based on BEV rasterizer features.
    # This can suppress dense false positives in empty regions, especially early in finetuning.
    if isinstance(bev_feat, torch.Tensor) and bev_feat.ndim == 4:
        if bev_feat.shape[0] != 1:
            bev_feat = bev_feat[:1]
        bev_feat_t = bev_feat.detach().float()  # (1, C, H, W)
        # Use local-max support within a small neighborhood; vehicle centers may be empty
        # even when there are strong surface points nearby.
        support_kernel = 5
        support_pad = support_kernel // 2
        density_map = bev_feat_t[0, 0]  # (H, W)
        density_pool = F.max_pool2d(
            density_map[None, None],
            kernel_size=support_kernel,
            stride=1,
            padding=support_pad,
        )[0, 0]
        density = density_pool[ys, xs]
        support_mask = torch.ones_like(scores, dtype=torch.bool)
        if cfg.min_density is not None:
            support_mask &= density >= float(cfg.min_density)
        # When both extra channels are enabled, rasterizer output order is:
        # [density, high_ratio, mean_z, var_z]
        if cfg.min_high_ratio is not None and bev_feat_t.shape[1] >= 4:
            high_map = bev_feat_t[0, 1]
            high_pool = F.max_pool2d(
                high_map[None, None],
                kernel_size=support_kernel,
                stride=1,
                padding=support_pad,
            )[0, 0]
            high_ratio = high_pool[ys, xs]
            support_mask &= high_ratio >= float(cfg.min_high_ratio)
        if cfg.min_var_z is not None and bev_feat_t.shape[1] >= 4:
            var_map = bev_feat_t[0, 3]
            var_pool = F.max_pool2d(
                var_map[None, None],
                kernel_size=support_kernel,
                stride=1,
                padding=support_pad,
            )[0, 0]
            var_z = var_pool[ys, xs]
            support_mask &= var_z >= float(cfg.min_var_z)
        if not torch.any(support_mask):
            return []
        ys = ys[support_mask]
        xs = xs[support_mask]
        scores = scores[support_mask]

    reg_sel = reg_t[:, ys, xs].transpose(0, 1)  # (N, 8)
    offset_x = reg_sel[:, 0]
    offset_y = reg_sel[:, 1]
    log_l = reg_sel[:, 2]
    log_w = reg_sel[:, 3]
    z = reg_sel[:, 4]
    log_h = reg_sel[:, 5]
    sin_yaw = reg_sel[:, 6]
    cos_yaw = reg_sel[:, 7]

    cx = (xs.float() + offset_x) * float(cfg.voxel_size) + float(cfg.x_range[0])
    cy = (ys.float() + offset_y) * float(cfg.voxel_size) + float(cfg.y_range[0])
    l = torch.exp(log_l)
    w = torch.exp(log_w)
    h = torch.exp(log_h)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    boxes = (
        torch.stack([cx, cy, z, l, w, h, yaw], dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)

    decoded: List[Dict[str, object]] = []
    for i in range(int(scores_np.shape[0])):
        decoded.append({"score": float(scores_np[i]), "box": boxes[i]})

    decoded = _nms_bev_oriented(decoded, iou_thresh=float(cfg.nms_iou), max_dets=int(cfg.max_dets))
    return decoded


def _match_boxes_greedy(
    pred_boxes: List[Dict[str, object]],
    gt_boxes: np.ndarray,
    *,
    iou_thresh: float,
) -> tuple[int, int, int, float]:
    if gt_boxes.size == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return tp, fp, fn, float("nan")
    matched = np.zeros((gt_boxes.shape[0],), dtype=bool)
    # preds already sorted by score
    tp = 0
    fp = 0
    ious = []
    for item in pred_boxes:
        box = item.get("box")
        if not isinstance(box, np.ndarray) or box.shape != (7,):
            continue
        best_iou = 0.0
        best_j = -1
        for j in range(gt_boxes.shape[0]):
            if matched[j]:
                continue
            iou = _oriented_bev_iou(box, gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            matched[best_j] = True
            tp += 1
            ious.append(best_iou)
        else:
            fp += 1
    fn = int((~matched).sum())
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    return tp, fp, fn, mean_iou


def compute_ap_bev(
    preds: List[Tuple[float, Tuple[str, str], np.ndarray]],
    gt_by_frame: Dict[Tuple[str, str], np.ndarray],
    *,
    iou_thresh: float,
) -> Tuple[float, float, float]:
    """Compute AP, precision, recall for a single IoU threshold."""
    total_gt = int(sum(v.shape[0] for v in gt_by_frame.values()))
    if total_gt == 0:
        return float("nan"), float("nan"), float("nan")

    # Sort predictions by confidence
    preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
    matched: Dict[Tuple[str, str], np.ndarray] = {
        k: np.zeros((v.shape[0],), dtype=bool) for k, v in gt_by_frame.items()
    }
    tp_flags = []
    fp_flags = []
    for score, frame_key, box in preds_sorted:
        _ = score
        gts = gt_by_frame.get(frame_key)
        if gts is None or gts.size == 0:
            tp_flags.append(0)
            fp_flags.append(1)
            continue
        m = matched[frame_key]
        best_iou = 0.0
        best_j = -1
        for j in range(gts.shape[0]):
            if m[j]:
                continue
            iou = _oriented_bev_iou(box, gts[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            m[best_j] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cum = np.cumsum(tp_flags).astype(np.float64)
    fp_cum = np.cumsum(fp_flags).astype(np.float64)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)

    # Precision envelope + integral
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    precision = float(precisions[-1]) if precisions.size else 0.0
    recall = float(recalls[-1]) if recalls.size else 0.0
    return ap, precision, recall


def _save_det_ap_cache_npz(
    out_path: Path,
    *,
    preds: List[Tuple[float, Tuple[str, str], np.ndarray]],
    gt_by_frame: Dict[Tuple[str, str], np.ndarray],
    iou_thresh: float,
) -> None:
    """Save a compact det cache for offline AP subset scoring.

    We store per-pred TP flags (computed with the same greedy matching policy as
    `compute_ap_bev`) so downstream analyses can avoid recomputing IoUs.
    """

    frame_keys = sorted(gt_by_frame.keys())
    if not frame_keys:
        return

    key_to_idx = {k: i for i, k in enumerate(frame_keys)}
    # Keep global insertion order so offline sorting can reproduce the stable sort
    # behavior of `compute_ap_bev` when scores tie.
    preds_by_idx: List[List[Tuple[float, np.ndarray, int]]] = [[] for _ in frame_keys]
    for global_i, (score, frame_key, box) in enumerate(preds):
        idx = key_to_idx.get(frame_key)
        if idx is None:
            continue
        preds_by_idx[idx].append((float(score), box, int(global_i)))

    frame_seq = np.asarray([k[0] for k in frame_keys], dtype=str)
    frame_frame = np.asarray([k[1] for k in frame_keys], dtype=str)
    gt_counts = np.asarray([int(gt_by_frame[k].shape[0]) for k in frame_keys], dtype=np.int32)

    # Offline cache outputs are stored in the *original* global order (preds list order).
    pred_scores = np.asarray([float(s) for s, _k, _b in preds], dtype=np.float32)
    pred_frame_idx = np.asarray([int(key_to_idx.get(k, -1)) for _s, k, _b in preds], dtype=np.int32)
    pred_tp = np.zeros((pred_scores.shape[0],), dtype=np.int8)

    for idx, k in enumerate(frame_keys):
        gts = gt_by_frame.get(k)
        if gts is None:
            gts = np.zeros((0, 7), dtype=np.float32)
        # Sort by score desc; Python sort is stable so ties preserve insertion order
        # which is consistent with the global stable sort in `compute_ap_bev`.
        preds_sorted = sorted(preds_by_idx[idx], key=lambda x: x[0], reverse=True)
        if not preds_sorted:
            continue
        if gts.size == 0:
            continue

        matched = np.zeros((gts.shape[0],), dtype=bool)
        for _score, box, global_i in preds_sorted:
            best_iou = 0.0
            best_j = -1
            for j in range(gts.shape[0]):
                if matched[j]:
                    continue
                iou = _oriented_bev_iou(box, gts[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            is_tp = int(best_iou >= float(iou_thresh) and best_j >= 0)
            if is_tp:
                matched[best_j] = True
            if 0 <= int(global_i) < int(pred_tp.shape[0]):
                pred_tp[int(global_i)] = int(is_tp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        format_version=np.asarray(["det_ap_cache_v1_tp_flags"], dtype=str),
        iou_thresh=np.asarray([float(iou_thresh)], dtype=np.float32),
        frame_sequence=frame_seq,
        frame_frame=frame_frame,
        gt_counts=gt_counts,
        pred_scores=pred_scores,
        pred_tp=pred_tp,
        pred_frame_idx=pred_frame_idx,
        pred_order=np.arange(int(pred_scores.shape[0]), dtype=np.int32),
    )


def box_to_corners_ego(box: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,l,w,h,yaw] (ego CARLA) to (8,3) corners."""
    x, y, z, l, w, h, yaw = [float(v) for v in box]
    dx = l * 0.5
    dy = w * 0.5
    dz = h * 0.5
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * dx, sy * dy, sz * dz], dtype=np.float32)
                corners.append((rot @ local) + np.array([x, y, z], dtype=np.float32))
    return np.stack(corners, axis=0)


def _checkpoint_to_state_dict(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint file and return the underlying PyTorch state_dict.

    Supports the common checkpoint formats used in this repo:
      - raw state_dict
      - {"model": state_dict}
      - {"state_dict": state_dict}
      - .safetensors
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("safetensors is required to load .safetensors checkpoints") from exc
        ckpt = load_safetensors(str(ckpt_path))
    else:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    state_dict = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format (expected dict state_dict): {ckpt_path}")
    return state_dict  # type: ignore[return-value]


def _load_det_head_only_from_checkpoint(model: torch.nn.Module, det_head_ckpt: str | Path) -> None:
    """Load ONLY det_head weights into an already-initialized mapanything_det model.

    This enables a composed eval where:
      - geometry weights come from one checkpoint
      - det_head weights come from a fixed det/e2e checkpoint
    """
    det_head = getattr(model, "det_head", None)
    if det_head is None:
        raise RuntimeError("Model has no det_head attribute; cannot load det head weights.")

    sd = _checkpoint_to_state_dict(det_head_ckpt)
    det_sd = {k[len("det_head.") :]: v for k, v in sd.items() if k.startswith("det_head.")}
    if not det_sd:
        raise RuntimeError(f"No det_head.* keys found in checkpoint: {det_head_ckpt}")

    missing, unexpected = det_head.load_state_dict(det_sd, strict=False)
    if missing or unexpected:
        print(
            "[WARN] det_head weights loaded with mismatches: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        # Print a few examples for debugging, but keep logs short in long evals.
        if missing:
            print("[WARN] det_head missing keys (first 10):", missing[:10])
        if unexpected:
            print("[WARN] det_head unexpected keys (first 10):", unexpected[:10])


def evaluate_model_on_frames(
    model_name: str,
    checkpoint: str,
    frames: List[FrameInfo],
    images_root: Path,
    depth_root: Path,
    split: str,
    device: torch.device,
    output_dir: Path,
    save_representative: bool = False,
    modes: Sequence[str] = ("single", "coop"),
    pc_metrics_cfg: PCMetricConfig | None = None,
    det_metrics_cfg: DetMetricConfig | None = None,
    keep_camera_poses: bool = False,
    keep_main_agent_poses: bool = False,
    model_task: str = "images_only",
    det_head_cfg: str = "bev_centernet",
    det_head_ckpt: str | None = None,
    model_arch: str = "mapanything",
    data_norm_type: str | None = None,
    vggt_enable_metric_scale_head: bool = False,
    vggt_autocast_dtype: str | None = None,
    gt_scale_contract: dict[tuple[str, str], tuple[float | None, float | None]] | None = None,
    save_det_cache: bool = False,
    log_every: int = 50,
) -> Dict[str, List[EvalResult]]:
    if not modes:
        raise ValueError("At least one evaluation mode must be specified.")
    use_det = bool(det_metrics_cfg and det_metrics_cfg.enabled)

    norm_type = resolve_data_norm_type(model_arch, data_norm_type)

    if model_arch == "vggt":
        model_name = "vggt_det" if use_det else "vggt"
        cfg_overrides = [
            "machine=local_a800",
            "dataset=opv2v_ft_a800",
            f"model={model_name}",
            f"model/det_head={det_head_cfg}" if use_det else None,
            # Avoid HF download in eval loops; checkpoints are expected to carry weights.
            "model.model_config.load_pretrained_weights=false",
            f"model.model_config.enable_metric_scale_head={'true' if vggt_enable_metric_scale_head else 'false'}",
            f"model.model_config.autocast_dtype={vggt_autocast_dtype}" if vggt_autocast_dtype else None,
            "loss=overall_loss",
        ]
    else:
        cfg_overrides = [
            v
            for v in [
                "machine=local_a800",
                "dataset=opv2v_ft_a800",
                "model=mapanything_det" if use_det else "model=mapanything",
                f"model/det_head={det_head_cfg}" if use_det else None,
                f"model/task={model_task}",
                "model.encoder.uses_torch_hub=false",
                "loss=overall_loss",
            ]
            if v
        ]

    cfg = {
        "path": str(REPO_ROOT / "configs/train.yaml"),
        "checkpoint_path": checkpoint,
        "config_overrides": [v for v in cfg_overrides if v],
    }
    model = initialize_mapanything_local(cfg, device)
    if use_det and det_head_ckpt:
        # Composed eval: use geometry from `checkpoint`, but reuse a fixed det_head from another ckpt.
        _load_det_head_only_from_checkpoint(model, det_head_ckpt)
    model.eval()

    per_mode_results: Dict[str, List[EvalResult]] = {mode: [] for mode in modes}
    # Keep only lightweight info for representative selection; rerun inference for the
    # final selected frames to avoid holding GPU tensors for every frame.
    representative_candidates: Dict[str, List[Tuple[float, FrameInfo]]] = {mode: [] for mode in modes}
    det_preds: Dict[str, List[Tuple[float, Tuple[str, str], np.ndarray]]] = {mode: [] for mode in modes}
    det_gts: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {mode: {} for mode in modes}
    # Avoid log spam in large-contract evals: report pose-input policy once per mode.
    pose_policy_logged: set[str] = set()

    t0 = time.time()
    n_total = int(len(frames) * len(modes))
    n_seen = 0
    n_ok = 0

    for info in frames:
        for mode in modes:
            n_seen += 1
            try:
                if mode == "single":
                    raw_views, camera_info = build_single_raw(images_root, depth_root, split, info)
                else:
                    raw_views, camera_info = build_coop_raw(images_root, depth_root, split, info)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Skip {info.sequence}/{info.frame} ({mode}): {exc}")
                continue

            gt_ref_pose_c2e_cv = None
            if raw_views and "camera_poses" in raw_views[0]:
                gt_ref_pose_c2e_cv = np.asarray(raw_views[0]["camera_poses"], dtype=np.float32)

            processed_views = preprocess_inputs(raw_views, norm_type=norm_type)
            gt_depths = [view["depth_z"].clone() for view in processed_views if "depth_z" in view]
            for view in processed_views:
                if "depth_z" in view:
                    view.pop("depth_z")
            if keep_camera_poses:
                if mode not in pose_policy_logged:
                    print(
                        f"[INFO] keep_camera_poses=1 ({mode}): posed inference (GT extrinsics kept)."
                    )
                    pose_policy_logged.add(mode)
            else:
                if keep_main_agent_poses and isinstance(camera_info, list) and camera_info:
                    # Strip only cross-agent camera poses to avoid leaking GT relative vehicle pose,
                    # while still keeping the main-agent rig extrinsics to anchor the ego frame.
                    pose_keys = ("camera_poses", "camera_pose_quats", "camera_pose_trans")
                    # Keep behavior consistent with strip_external_calibration_inputs(): remove cached ray dirs.
                    ray_keys = ("ray_directions", "ray_directions_cam")
                    for view, cam in zip(processed_views, camera_info):
                        for k in ray_keys:
                            view.pop(k, None)

                        # Single-mode camera_info names are like "camera0", so treat them as main-agent.
                        name = cam.get("name") if isinstance(cam, dict) else None
                        is_main = True if mode == "single" else (
                            isinstance(name, str) and name.endswith(f"_{info.main_agent}")
                        )
                        if not is_main:
                            for k in pose_keys:
                                view.pop(k, None)

                    if mode not in pose_policy_logged:
                        print(
                            f"[INFO] keep_camera_poses=0 keep_main_agent_poses=1 ({mode}): "
                            "deployment-like (strip cross-agent poses, keep main-agent rig)."
                        )
                        pose_policy_logged.add(mode)
                else:
                    strip_external_calibration_inputs(processed_views)
                    if mode not in pose_policy_logged:
                        print(
                            f"[INFO] keep_camera_poses=0 ({mode}): unposed inference (external pose inputs stripped)."
                        )
                        pose_policy_logged.add(mode)
            try:
                with torch.no_grad():
                    if model_arch == "vggt":
                        _move_views_to_device(processed_views, device)
                        raw_preds = model(processed_views)
                        predictions = _vggt_preds_to_batch_eval_format(raw_preds, processed_views)
                    else:
                        predictions = model.infer(
                            processed_views, memory_efficient_inference=True
                        )
            except Exception as exc:  # noqa: BLE001
                # Some unstable checkpoints can produce degenerate outputs that make
                # intrinsics recovery singular inside model postprocess. We skip only
                # this frame/mode so long-contract evals still complete.
                print(
                    f"[WARN] Inference failed for {info.sequence}/{info.frame} ({mode}): {exc}"
                )
                continue

            det_num_gt = det_num_pred = det_tp = det_fp = det_fn = None
            det_mean_iou = None
            if use_det and det_metrics_cfg is not None:
                frame_key = (info.sequence, info.frame)
                yaml_path = images_root / split / info.sequence / info.main_agent / f"{info.frame}.yaml"
                try:
                    frame_meta = load_frame_metadata(yaml_path)
                    gt_padded, gt_mask = _extract_vehicle_boxes_in_ego(
                        frame_meta,
                        max_range=det_metrics_cfg.bbox_range,
                        max_num_boxes=det_metrics_cfg.max_num_boxes,
                    )
                    gt_boxes = gt_padded[gt_mask]
                except Exception:  # noqa: BLE001
                    gt_boxes = np.zeros((0, 7), dtype=np.float32)

                det_out = None
                if predictions and isinstance(predictions[0], dict):
                    det_out = predictions[0].get("bev_det")
                pred_boxes = (
                    decode_bev_centernet(det_out, det_metrics_cfg)
                    if isinstance(det_out, dict)
                    else []
                )

                det_num_gt = int(gt_boxes.shape[0])
                det_num_pred = int(len(pred_boxes))
                det_tp, det_fp, det_fn, det_mean_iou = _match_boxes_greedy(
                    pred_boxes, gt_boxes, iou_thresh=float(det_metrics_cfg.iou_thresh)
                )

                det_gts[mode][frame_key] = gt_boxes
                for item in pred_boxes:
                    score = item.get("score")
                    box = item.get("box")
                    if isinstance(score, (float, int)) and isinstance(box, np.ndarray):
                        det_preds[mode].append((float(score), frame_key, box))
            # Point-cloud conversion is expensive and CPU-bound (detaches tensors, runs numpy),
            # so only do it when downstream point-cloud metrics or dumps are requested.
            pred_points_carla = None
            if pc_metrics_cfg is not None and (pc_metrics_cfg.enabled or pc_metrics_cfg.save_dir):
                if gt_ref_pose_c2e_cv is None:
                    pred_points, _ = predictions_to_pointcloud(predictions, colorize=False)
                else:
                    gt_pose_list = None
                    if keep_camera_poses and camera_info:
                        pose_key = "pose_C2E_cv" if "pose_C2E_cv" in camera_info[0] else None
                        if pose_key:
                            gt_pose_list = [
                                np.asarray(cam[pose_key].cpu().numpy(), dtype=np.float32)
                                for cam in camera_info
                            ]
                    pred_points, _ = predictions_to_ego_pointcloud(
                        predictions,
                        gt_ref_pose_c2e_cv,
                        colorize=False,
                        gt_pose_c2e_cv_list=gt_pose_list,
                    )
                pred_points_carla = _points_opencv_to_carla(pred_points)
            pose_abs, pose_rot, pose_ate, pose_ate_rot = _compute_pose_metrics(predictions, camera_info)
            cross_pose_trans = cross_pose_rot = None
            if mode == "coop":
                cross_pose_trans, cross_pose_rot = _compute_cross_agent_pose_metrics(
                    predictions, camera_info
                )
            rig_pose_trans, rig_pose_rot = _compute_intra_agent_rig_metrics(
                predictions, camera_info
            )
            depth_rmse, depth_mae, depth_rel = depth_metrics(predictions, gt_depths)
            gt_scale_factor = None
            if gt_scale_contract is not None:
                cached = gt_scale_contract.get((info.sequence, info.frame))
                if cached is not None:
                    gt_scale_factor = cached[1] if mode == "coop" else cached[0]
            if gt_scale_factor is None:
                gt_scale_factor = _gt_scale_factor_from_raw_views(raw_views)
            pred_scale_fallback = None
            if (
                gt_scale_factor is not None
                and not any(p.get("metric_scaling_factor") is not None for p in predictions)
            ):
                pred_scale_fallback = _pred_scale_factor_from_predictions(predictions, gt_depths)
            scale_details = scale_metric_details(
                predictions,
                gt_scale_factor=gt_scale_factor,
                fallback_scale_factor=pred_scale_fallback,
            )
            detection_metrics = {}
            if pc_metrics_cfg:
                if pred_points_carla is None:
                    raise RuntimeError(
                        "pc_metrics_cfg is set but predicted point cloud is missing; "
                        "this indicates a logic bug in batch_eval.py."
                    )
                if pc_metrics_cfg.save_dir:
                    save_dir = pc_metrics_cfg.save_dir / model_name / mode
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{info.sequence}_{info.frame}.npy"
                    np.save(save_path, pred_points_carla.astype(np.float32))
                if pc_metrics_cfg.enabled:
                    gt_pcd_path = images_root / split / info.sequence / info.main_agent / f"{info.frame}.pcd"
                    if not gt_pcd_path.is_file():
                        print(f"[WARN] GT point cloud missing for detection metrics: {gt_pcd_path}")
                    else:
                        gt_points = load_ascii_pcd_xyz(gt_pcd_path)
                        detection_metrics = compute_detection_metrics(pred_points_carla, gt_points, pc_metrics_cfg)

            per_mode_results[mode].append(
                EvalResult(
                    frame=info,
                    mode=mode,
                    pose_abs=pose_abs,
                    pose_rot=pose_rot,
                    pose_ate=pose_ate,
                    pose_ate_rot=pose_ate_rot,
                    depth_rmse=depth_rmse,
                    depth_mae=depth_mae,
                    depth_rel=depth_rel,
                    cross_agent_pose_trans=cross_pose_trans,
                    cross_agent_pose_rot=cross_pose_rot,
                    intra_agent_rig_pose_trans=rig_pose_trans,
                    intra_agent_rig_pose_rot=rig_pose_rot,
                    scale_err=(
                        scale_details.get("scale_err")
                        if scale_details is not None
                        else None
                    ),
                    scale_log_err=(
                        scale_details.get("scale_log_err")
                        if scale_details is not None
                        else None
                    ),
                    scale_eq_rel_err=(
                        scale_details.get("scale_eq_rel_err")
                        if scale_details is not None
                        else None
                    ),
                    scale_ratio_mean=(
                        scale_details.get("scale_ratio_mean")
                        if scale_details is not None
                        else None
                    ),
                    scale_ratio_median=(
                        scale_details.get("scale_ratio_median")
                        if scale_details is not None
                        else None
                    ),
                    scale_ratio_p90=(
                        scale_details.get("scale_ratio_p90")
                        if scale_details is not None
                        else None
                    ),
                    scale_gt_factor=(gt_scale_factor if gt_scale_factor is not None else None),
                    scale_to_gt_err=(scale_details.get("scale_to_gt_err") if scale_details is not None else None),
                    scale_to_gt_log_err=(scale_details.get("scale_to_gt_log_err") if scale_details is not None else None),
                    scale_to_gt_eq_rel_err=(scale_details.get("scale_to_gt_eq_rel_err") if scale_details is not None else None),
                    scale_to_gt_ratio_mean=(scale_details.get("scale_to_gt_ratio_mean") if scale_details is not None else None),
                    scale_to_gt_ratio_median=(scale_details.get("scale_to_gt_ratio_median") if scale_details is not None else None),
                    scale_to_gt_ratio_p90=(scale_details.get("scale_to_gt_ratio_p90") if scale_details is not None else None),
                    chamfer_pred_to_gt=detection_metrics.get("chamfer_pred_to_gt"),
                    chamfer_gt_to_pred=detection_metrics.get("chamfer_gt_to_pred"),
                    chamfer_filtered_pred_to_gt=detection_metrics.get("chamfer_filtered_pred_to_gt"),
                    chamfer_filtered_gt_to_pred=detection_metrics.get("chamfer_filtered_gt_to_pred"),
                    bev_iou_raw=detection_metrics.get("bev_iou_raw"),
                    bev_iou_filtered=detection_metrics.get("bev_iou_filtered"),
                    det_num_gt=det_num_gt,
                    det_num_pred=det_num_pred,
                    det_tp_iou=det_tp,
                    det_fp_iou=det_fp,
                    det_fn_iou=det_fn,
                    det_mean_iou=det_mean_iou,
                )
            )
            n_ok += 1
            if save_representative:
                representative_candidates[mode].append(
                    (
                        depth_rmse if not math.isnan(depth_rmse) else float("inf"),
                        info,
                    )
                )
            if log_every and (n_seen % int(log_every) == 0 or n_seen == n_total):
                dt = max(time.time() - t0, 1e-6)
                rate = n_seen / dt
                eta_s = (n_total - n_seen) / max(rate, 1e-6)
                print(
                    f"[PROGRESS] seen={n_seen}/{n_total} ok={n_ok} "
                    f"rate={rate:.2f} it/s ETA={eta_s/60.0:.1f} min"
                )

    if use_det and det_metrics_cfg is not None:
        for mode in modes:
            ap, precision, recall = compute_ap_bev(
                det_preds[mode],
                det_gts[mode],
                iou_thresh=float(det_metrics_cfg.iou_thresh),
            )
            for v in per_mode_results[mode]:
                v.det_ap_iou = ap
                v.det_precision_iou = precision
                v.det_recall_iou = recall

        if save_det_cache:
            for mode in modes:
                cache_path = output_dir / model_name / f"det_ap_cache_{mode}.npz"
                _save_det_ap_cache_npz(
                    cache_path,
                    preds=det_preds[mode],
                    gt_by_frame=det_gts[mode],
                    iou_thresh=float(det_metrics_cfg.iou_thresh),
                )

    # Save representative point clouds if requested
    if save_representative:
        def _infer_for_representative(frame_info: FrameInfo, rep_mode: str) -> tuple[list, np.ndarray | None, list[dict]]:
            if rep_mode == "single":
                raw_views, cam_info = build_single_raw(images_root, depth_root, split, frame_info)
            else:
                raw_views, cam_info = build_coop_raw(images_root, depth_root, split, frame_info)

            gt_ref_pose_cv = None
            if raw_views and "camera_poses" in raw_views[0]:
                gt_ref_pose_cv = np.asarray(raw_views[0]["camera_poses"], dtype=np.float32)

            # IMPORTANT: Representative inference must match the main eval protocol,
            # otherwise saved PCDs/boxes can be misleading (e.g., keep_main_agent_poses).
            processed = preprocess_inputs(raw_views, norm_type=norm_type)
            for view in processed:
                view.pop("depth_z", None)
            if not keep_camera_poses:
                if keep_main_agent_poses and isinstance(cam_info, list) and cam_info:
                    # Mirror the main-loop behavior: keep main-agent rig poses, strip only cross-agent poses.
                    pose_keys = ("camera_poses", "camera_pose_quats", "camera_pose_trans")
                    ray_keys = ("ray_directions", "ray_directions_cam")
                    for view, cam in zip(processed, cam_info):
                        for k in ray_keys:
                            view.pop(k, None)
                        name = cam.get("name") if isinstance(cam, dict) else None
                        is_main = True if rep_mode == "single" else (
                            isinstance(name, str) and name.endswith(f"_{frame_info.main_agent}")
                        )
                        if not is_main:
                            for k in pose_keys:
                                view.pop(k, None)
                else:
                    strip_external_calibration_inputs(processed)

            with torch.no_grad():
                if model_arch == "vggt":
                    _move_views_to_device(processed, device)
                    raw_preds = model(processed)
                    preds = _vggt_preds_to_batch_eval_format(raw_preds, processed)
                else:
                    preds = model.infer(processed, memory_efficient_inference=True)
            return preds, gt_ref_pose_cv, cam_info

        for mode, candidates in representative_candidates.items():
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])
            selected = []
            if candidates:
                selected.append(("best", candidates[0]))
            if len(candidates) > 2:
                selected.append(("worst", candidates[-1]))
            if len(candidates) > 4:
                mid = len(candidates) // 2
                selected.append(("median", candidates[mid]))
            rep_dir = output_dir / model_name / f"{mode}_representatives"
            rep_dir.mkdir(parents=True, exist_ok=True)
            for tag, (_, frame_info) in selected:
                preds, gt_ref_pose_cv, cam_info = _infer_for_representative(frame_info, mode)
                if gt_ref_pose_cv is None:
                    pts, _ = predictions_to_pointcloud(preds, colorize=False)
                else:
                    gt_pose_list = None
                    if keep_camera_poses and cam_info:
                        pose_key = "pose_C2E_cv" if "pose_C2E_cv" in cam_info[0] else None
                        if pose_key:
                            gt_pose_list = [
                                np.asarray(cam[pose_key].cpu().numpy(), dtype=np.float32)
                                for cam in cam_info
                            ]
                    pts, _ = predictions_to_ego_pointcloud(
                        preds,
                        gt_ref_pose_cv,
                        colorize=False,
                        gt_pose_c2e_cv_list=gt_pose_list,
                    )
                pts = _points_opencv_to_carla(pts)
                out_path = rep_dir / f"{frame_info.sequence}_{frame_info.frame}_{tag}.pcd"
                save_point_cloud(out_path, pts)
                if use_det and det_metrics_cfg is not None and preds and isinstance(preds[0], dict):
                    det_out = preds[0].get("bev_det")
                    pred_boxes = (
                        decode_bev_centernet(det_out, det_metrics_cfg)
                        if isinstance(det_out, dict)
                        else []
                    )
                    boxes_json = out_path.with_name(out_path.stem + "_pred_boxes.json")
                    serializable = []
                    for item in pred_boxes:
                        score = float(item.get("score", 0.0))
                        box = item.get("box")
                        if not isinstance(box, np.ndarray):
                            continue
                        corners = box_to_corners_ego(box)
                        serializable.append(
                            {
                                "score": score,
                                "box": box.astype(float).tolist(),
                                "corners": corners.astype(float).tolist(),
                            }
                        )
                    with boxes_json.open("w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "sequence": frame_info.sequence,
                                "frame": frame_info.frame,
                                "mode": mode,
                                "model": model_name,
                                "iou_thresh": float(det_metrics_cfg.iou_thresh),
                                "score_thresh": float(det_metrics_cfg.score_thresh),
                                "boxes": serializable,
                            },
                            fh,
                            indent=2,
                        )

    return per_mode_results


def summarize_results(results: Dict[str, List[EvalResult]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for mode, values in results.items():
        if not values:
            continue
        def _attr_mean(attr: str) -> float:
            arr = [getattr(v, attr) for v in values if getattr(v, attr) is not None]
            return float(np.nanmean(arr)) if arr else float("nan")
        summary[mode] = {
            "frames": len(values),
            "pose_abs_mean": float(np.nanmean([v.pose_abs for v in values])),
            "pose_rot_mean": float(np.nanmean([v.pose_rot for v in values])),
            "cross_agent_pose_trans_mean": _attr_mean("cross_agent_pose_trans"),
            "cross_agent_pose_rot_mean": _attr_mean("cross_agent_pose_rot"),
            "intra_agent_rig_pose_trans_mean": _attr_mean("intra_agent_rig_pose_trans"),
            "intra_agent_rig_pose_rot_mean": _attr_mean("intra_agent_rig_pose_rot"),
            "pose_ate_mean": _attr_mean("pose_ate"),
            "pose_ate_rot_mean": _attr_mean("pose_ate_rot"),
            "depth_rmse_mean": float(np.nanmean([v.depth_rmse for v in values])),
            "depth_mae_mean": float(np.nanmean([v.depth_mae for v in values])),
            "depth_rel_mean": float(np.nanmean([v.depth_rel for v in values])),
            "scale_err_mean": float(np.nanmean([v.scale_err for v in values if v.scale_err is not None])) if any(v.scale_err is not None for v in values) else float("nan"),
            "scale_log_err_mean": _attr_mean("scale_log_err"),
            "scale_eq_rel_err_mean": _attr_mean("scale_eq_rel_err"),
            # More human-friendly multiplicative error: exp(mean(|log(s)|)).
            # Per-frame we compute `scale_eq_rel_err = expm1(mean(|log(s)|))`,
            # so `scale_mult_err = scale_eq_rel_err + 1`.
            "scale_mult_err_mean": float("nan"),
            "scale_ratio_mean": _attr_mean("scale_ratio_mean"),
            "scale_ratio_median_mean": _attr_mean("scale_ratio_median"),
            "scale_ratio_p90_mean": _attr_mean("scale_ratio_p90"),
            "scale_gt_factor_mean": _attr_mean("scale_gt_factor"),
            "scale_to_gt_err_mean": _attr_mean("scale_to_gt_err"),
            "scale_to_gt_log_err_mean": _attr_mean("scale_to_gt_log_err"),
            "scale_to_gt_eq_rel_err_mean": _attr_mean("scale_to_gt_eq_rel_err"),
            # Human-friendly multiplicative error: exp(mean(|log(s/g)|)).
            "scale_to_gt_mult_err_mean": float("nan"),
            "scale_to_gt_ratio_mean": _attr_mean("scale_to_gt_ratio_mean"),
            "scale_to_gt_ratio_median_mean": _attr_mean("scale_to_gt_ratio_median"),
            "scale_to_gt_ratio_p90_mean": _attr_mean("scale_to_gt_ratio_p90"),
            "chamfer_pred_to_gt_mean": _attr_mean("chamfer_pred_to_gt"),
            "chamfer_gt_to_pred_mean": _attr_mean("chamfer_gt_to_pred"),
            "chamfer_filtered_pred_to_gt_mean": _attr_mean("chamfer_filtered_pred_to_gt"),
            "chamfer_filtered_gt_to_pred_mean": _attr_mean("chamfer_filtered_gt_to_pred"),
            "bev_iou_raw_mean": _attr_mean("bev_iou_raw"),
            "bev_iou_filtered_mean": _attr_mean("bev_iou_filtered"),
        }
        if any(v.det_ap_iou is not None for v in values):
            summary[mode].update(
                {
                    "det_ap_iou": _attr_mean("det_ap_iou"),
                    "det_precision_iou": _attr_mean("det_precision_iou"),
                    "det_recall_iou": _attr_mean("det_recall_iou"),
                    "det_mean_iou": _attr_mean("det_mean_iou"),
                }
            )

        # Fill the derived mult-err fields if source fields are finite.
        eq_rel = summary[mode].get("scale_eq_rel_err_mean")
        if isinstance(eq_rel, (int, float)) and math.isfinite(float(eq_rel)):
            summary[mode]["scale_mult_err_mean"] = float(eq_rel) + 1.0
        eq_rel_gt = summary[mode].get("scale_to_gt_eq_rel_err_mean")
        if isinstance(eq_rel_gt, (int, float)) and math.isfinite(float(eq_rel_gt)):
            summary[mode]["scale_to_gt_mult_err_mean"] = float(eq_rel_gt) + 1.0
    return summary


def save_metrics_csv(results: Dict[str, List[EvalResult]], out_dir: Path, model_name: str) -> None:
    for mode, values in results.items():
        if not values:
            continue
        csv_path = out_dir / model_name / f"{mode}_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "sequence",
                    "frame",
                    "main_agent",
                    "coop_agents",
                    "baseline_distance_m",
                    "baseline_bucket",
                    "pose_abs_m",
                    "pose_rot_deg",
                    "pose_ate_m",
                    "pose_ate_rot_deg",
                    "cross_agent_pose_trans_m",
                    "cross_agent_pose_rot_deg",
                    "intra_agent_rig_pose_trans_m",
                    "intra_agent_rig_pose_rot_deg",
                    "depth_rmse",
                    "depth_mae",
                    "depth_rel",
                    "scale_err",
                    "scale_log_err",
                    "scale_eq_rel_err",
                    "scale_ratio_mean",
                    "scale_ratio_median",
                    "scale_ratio_p90",
                    "scale_gt_factor",
                    "scale_to_gt_err",
                    "scale_to_gt_log_err",
                    "scale_to_gt_eq_rel_err",
                    "scale_to_gt_ratio_mean",
                    "scale_to_gt_ratio_median",
                    "scale_to_gt_ratio_p90",
                    "chamfer_pred_to_gt",
                    "chamfer_gt_to_pred",
                    "chamfer_filtered_pred_to_gt",
                    "chamfer_filtered_gt_to_pred",
                    "bev_iou_raw",
                    "bev_iou_filtered",
                    "det_num_gt",
                    "det_num_pred",
                    "det_tp_iou",
                    "det_fp_iou",
                    "det_fn_iou",
                    "det_mean_iou",
                ]
            )
            for v in values:
                writer.writerow(
                    [
                        v.frame.sequence,
                        v.frame.frame,
                        v.frame.main_agent,
                        ",".join(v.frame.coop_agents),
                        f"{v.frame.baseline_distance_m:.6f}" if v.frame.baseline_distance_m is not None else "nan",
                        v.frame.baseline_bucket or "",
                        f"{v.pose_abs:.6f}",
                        f"{v.pose_rot:.6f}",
                        f"{v.pose_ate:.6f}" if v.pose_ate is not None else "nan",
                        f"{v.pose_ate_rot:.6f}" if v.pose_ate_rot is not None else "nan",
                        f"{v.cross_agent_pose_trans:.6f}" if v.cross_agent_pose_trans is not None else "nan",
                        f"{v.cross_agent_pose_rot:.6f}" if v.cross_agent_pose_rot is not None else "nan",
                        f"{v.intra_agent_rig_pose_trans:.6f}" if v.intra_agent_rig_pose_trans is not None else "nan",
                        f"{v.intra_agent_rig_pose_rot:.6f}" if v.intra_agent_rig_pose_rot is not None else "nan",
                        f"{v.depth_rmse:.6f}",
                        f"{v.depth_mae:.6f}",
                        f"{v.depth_rel:.6f}",
                        f"{v.scale_err:.6f}" if v.scale_err is not None else "nan",
                        f"{v.scale_log_err:.6f}" if v.scale_log_err is not None else "nan",
                        f"{v.scale_eq_rel_err:.6f}" if v.scale_eq_rel_err is not None else "nan",
                        f"{v.scale_ratio_mean:.6f}" if v.scale_ratio_mean is not None else "nan",
                        f"{v.scale_ratio_median:.6f}" if v.scale_ratio_median is not None else "nan",
                        f"{v.scale_ratio_p90:.6f}" if v.scale_ratio_p90 is not None else "nan",
                        f"{v.scale_gt_factor:.6f}" if v.scale_gt_factor is not None else "nan",
                        f"{v.scale_to_gt_err:.6f}" if v.scale_to_gt_err is not None else "nan",
                        f"{v.scale_to_gt_log_err:.6f}" if v.scale_to_gt_log_err is not None else "nan",
                        f"{v.scale_to_gt_eq_rel_err:.6f}" if v.scale_to_gt_eq_rel_err is not None else "nan",
                        f"{v.scale_to_gt_ratio_mean:.6f}" if v.scale_to_gt_ratio_mean is not None else "nan",
                        f"{v.scale_to_gt_ratio_median:.6f}" if v.scale_to_gt_ratio_median is not None else "nan",
                        f"{v.scale_to_gt_ratio_p90:.6f}" if v.scale_to_gt_ratio_p90 is not None else "nan",
                        f"{v.chamfer_pred_to_gt:.6f}" if v.chamfer_pred_to_gt is not None else "nan",
                        f"{v.chamfer_gt_to_pred:.6f}" if v.chamfer_gt_to_pred is not None else "nan",
                        f"{v.chamfer_filtered_pred_to_gt:.6f}" if v.chamfer_filtered_pred_to_gt is not None else "nan",
                        f"{v.chamfer_filtered_gt_to_pred:.6f}" if v.chamfer_filtered_gt_to_pred is not None else "nan",
                        f"{v.bev_iou_raw:.6f}" if v.bev_iou_raw is not None else "nan",
                        f"{v.bev_iou_filtered:.6f}" if v.bev_iou_filtered is not None else "nan",
                        str(v.det_num_gt) if v.det_num_gt is not None else "nan",
                        str(v.det_num_pred) if v.det_num_pred is not None else "nan",
                        str(v.det_tp_iou) if v.det_tp_iou is not None else "nan",
                        str(v.det_fp_iou) if v.det_fp_iou is not None else "nan",
                        str(v.det_fn_iou) if v.det_fn_iou is not None else "nan",
                        f"{v.det_mean_iou:.6f}" if v.det_mean_iou is not None else "nan",
                    ]
                )


def main() -> None:
    args = parse_args()
    # Determinism: BEV rasterization may randomly subsample points (max_points).
    # Seeding here makes det/e2e AP reproducible across runs under the same contract.
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoid nondeterministic algorithm selection in cuDNN (common source of eval drift).
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if args.device == "cpu" else "cuda")
    modes = tuple(dict.fromkeys(args.modes)) if args.modes else ("single", "coop")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_data_norm_type = resolve_data_norm_type(
        str(args.model_arch),
        str(args.data_norm_type) if args.data_norm_type else None,
    )
    eval_script_md5 = hashlib.md5(Path(__file__).resolve().read_bytes()).hexdigest()

    # Align decoding grid with the selected det-head config by default.
    # This avoids silent metric corruption when using wide BEV heads (e.g., x_range=[-50,120])
    # while the CLI defaults still assume x_range=[0,120].
    try:
        det_head_cfg_path = (
            REPO_ROOT / "configs" / "model" / "det_head" / f"{str(args.det_head_cfg)}.yaml"
        )
        if det_head_cfg_path.is_file():
            det_head_cfg = yaml.safe_load(det_head_cfg_path.read_text(encoding="utf-8")) or {}
            x_range = det_head_cfg.get("x_range")
            y_range = det_head_cfg.get("y_range")
            voxel_size = det_head_cfg.get("voxel_size")
            if (
                isinstance(x_range, (list, tuple))
                and len(x_range) == 2
                and isinstance(y_range, (list, tuple))
                and len(y_range) == 2
            ):
                # Only override when args are still at their CLI defaults.
                if float(args.det_x_min) == 0.0 and float(args.det_x_max) == 120.0:
                    args.det_x_min, args.det_x_max = float(x_range[0]), float(x_range[1])
                if float(args.det_y_min) == -50.0 and float(args.det_y_max) == 50.0:
                    args.det_y_min, args.det_y_max = float(y_range[0]), float(y_range[1])
            if voxel_size is not None and float(args.det_voxel_size) == 0.5:
                args.det_voxel_size = float(voxel_size)
    except Exception:  # noqa: BLE001
        pass

    frames = discover_frames(args.images_root, args.split)
    sampled = load_frames_from_json(args.frames_json) if args.frames_json else sample_frames(frames, args.sample_size, args.seed)
    if not sampled:
        raise RuntimeError("No frames discovered for evaluation.")
    # Record the true evaluated frame count and a stable hash for fairness checks.
    # When --frames_json is used, args.sample_size may not match the actual number of frames.
    evaluated_sample_size = len(sampled)
    # Legacy v1 contract hash: only sequence/frame (kept for backward compatibility).
    frame_items_v1 = sorted({f"{f.sequence}/{f.frame}" for f in sampled})
    frames_hash_md5_v1 = hashlib.md5("\n".join(frame_items_v1).encode("utf-8")).hexdigest()
    # Fair v2 contract hash: include pairing (main + coop_agents order).
    frame_items_v2 = sorted(
        {
            f"{f.sequence}/{f.frame}/main={f.main_agent}/coop={','.join(f.coop_agents)}"
            for f in sampled
        }
    )
    frames_hash_md5_v2 = hashlib.md5("\n".join(frame_items_v2).encode("utf-8")).hexdigest()
    # Keep old key name for existing downstream tools.
    frames_hash_md5 = frames_hash_md5_v1

    gt_scale_contract_hash = None
    gt_scale_contract_hash_v2 = None
    gt_scale_contract_lookup = None
    if args.gt_scale_contract_json is not None:
        gt_scale_contract_hash, gt_scale_contract_hash_v2, gt_scale_contract_lookup = _load_gt_scale_contract(
            args.gt_scale_contract_json
        )
        # Prefer the fair v2 hash when available.
        if gt_scale_contract_hash_v2 and gt_scale_contract_hash_v2 != frames_hash_md5_v2:
            raise ValueError(
                "gt_scale_contract_json hash mismatch: "
                f"contract_v2={gt_scale_contract_hash_v2} eval_v2={frames_hash_md5_v2}. "
                "Pass a matching contract or omit --gt_scale_contract_json."
            )
        if (
            (not gt_scale_contract_hash_v2)
            and gt_scale_contract_hash
            and gt_scale_contract_hash != frames_hash_md5_v1
        ):
            raise ValueError(
                "gt_scale_contract_json hash mismatch: "
                f"contract_v1={gt_scale_contract_hash} eval_v1={frames_hash_md5_v1}. "
                "Pass a matching contract or omit --gt_scale_contract_json."
            )

    model_map = dict(DEFAULT_MODELS)
    if args.models:
        for item in args.models:
            if "=" not in item:
                continue
            name, path = item.split("=", 1)
            model_map[name.strip()] = path.strip()
    if args.model_filter:
        filtered: Dict[str, str] = {}
        missing = []
        for key in args.model_filter:
            key = key.strip()
            if key in model_map:
                filtered[key] = model_map[key]
            else:
                missing.append(key)
        if missing:
            print(f"[WARN] Unknown model keys skipped: {', '.join(missing)}")
        if not filtered:
            raise ValueError("Model filter excluded all checkpoints.")
        model_map = filtered

    pc_cfg = PCMetricConfig(
        enabled=bool(args.pc_metrics),
        z_min=args.pc_filter_z_min,
        z_max=args.pc_filter_z_max,
        radius_max=args.pc_filter_radius,
        bev_range=args.pc_bev_range,
        bev_resolution=args.pc_bev_resolution,
        save_dir=args.pc_save_dir,
    )

    pc_cfg_for_eval = pc_cfg if (pc_cfg.enabled or pc_cfg.save_dir) else None
    det_cfg = DetMetricConfig(
        enabled=bool(args.det_metrics),
        score_thresh=float(args.det_score_thresh),
        nms_iou=float(args.det_nms_iou),
        max_dets=int(args.det_max_dets),
        iou_thresh=float(args.det_iou_thresh),
        min_density=float(args.det_min_density) if args.det_min_density is not None else None,
        min_high_ratio=float(args.det_min_high_ratio) if args.det_min_high_ratio is not None else None,
        min_var_z=float(args.det_min_var_z) if args.det_min_var_z is not None else None,
        bbox_range=float(args.det_bbox_range) if args.det_bbox_range is not None else None,
        max_num_boxes=int(args.det_max_num_boxes),
        x_range=(float(args.det_x_min), float(args.det_x_max)),
        y_range=(float(args.det_y_min), float(args.det_y_max)),
        voxel_size=float(args.det_voxel_size),
    )
    det_cfg_for_eval = det_cfg if det_cfg.enabled else None

    overall_summary = {}
    for name, ckpt in model_map.items():
        print(f"[INFO] Evaluating model '{name}' from {ckpt}")
        results = evaluate_model_on_frames(
            name,
            ckpt,
            sampled,
            args.images_root,
            args.depth_root,
            args.split,
            device,
            output_root,
            save_representative=args.save_representative,
            modes=modes,
            pc_metrics_cfg=pc_cfg_for_eval,
            det_metrics_cfg=det_cfg_for_eval,
            keep_camera_poses=bool(args.keep_camera_poses),
            keep_main_agent_poses=bool(args.keep_main_agent_poses),
            model_task=str(args.model_task),
            det_head_cfg=str(args.det_head_cfg),
            det_head_ckpt=str(args.det_head_ckpt) if args.det_head_ckpt else None,
            model_arch=str(args.model_arch),
            data_norm_type=str(args.data_norm_type) if args.data_norm_type else None,
            vggt_enable_metric_scale_head=bool(args.vggt_enable_metric_scale_head),
            vggt_autocast_dtype=str(args.vggt_autocast_dtype) if args.vggt_autocast_dtype else None,
            gt_scale_contract=gt_scale_contract_lookup,
            save_det_cache=bool(args.save_det_cache),
            log_every=int(args.log_every),
        )
        save_metrics_csv(results, output_root, name)
        overall_summary[name] = summarize_results(results)

    summary_path = output_root / f"summary_{args.split}.json"
    summary_payload = {
        "split": args.split,
        "frames": [f.__dict__ for f in sampled],
        "metrics": overall_summary,
        # Preserve model->checkpoint mapping for reproducibility and post-hoc tracing.
        "model_paths": {k: str(v) for k, v in model_map.items()},
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "sample_size": int(evaluated_sample_size),
            "seed": int(args.seed),
            "modes": list(modes),
            "model_task": str(args.model_task),
            "model_arch": str(args.model_arch),
            # Preserve CLI override (may be null when using defaults).
            "data_norm_type": str(args.data_norm_type) if args.data_norm_type else None,
            # Actual value passed to preprocess_inputs() after applying defaults.
            "resolved_data_norm_type": str(resolved_data_norm_type),
            "vggt_enable_metric_scale_head": bool(args.vggt_enable_metric_scale_head),
            "vggt_autocast_dtype": str(args.vggt_autocast_dtype) if args.vggt_autocast_dtype else None,
            "det_head_cfg": str(args.det_head_cfg),
            # When set, det_head weights are loaded from this checkpoint (composed eval).
            "det_head_ckpt": str(args.det_head_ckpt) if args.det_head_ckpt else None,
            "det_metrics": bool(det_cfg.enabled),
            "save_det_cache": bool(args.save_det_cache) if det_cfg.enabled else False,
            "det_decode_cfg": (
                {
                    "score_thresh": float(det_cfg.score_thresh),
                    "nms_iou": float(det_cfg.nms_iou),
                    "max_dets": int(det_cfg.max_dets),
                    "iou_thresh": float(det_cfg.iou_thresh),
                    "min_density": float(det_cfg.min_density) if det_cfg.min_density is not None else None,
                    "min_high_ratio": float(det_cfg.min_high_ratio) if det_cfg.min_high_ratio is not None else None,
                    "min_var_z": float(det_cfg.min_var_z) if det_cfg.min_var_z is not None else None,
                    "bbox_range": float(det_cfg.bbox_range) if det_cfg.bbox_range is not None else None,
                    "max_num_boxes": int(det_cfg.max_num_boxes),
                    "x_range": [float(det_cfg.x_range[0]), float(det_cfg.x_range[1])],
                    "y_range": [float(det_cfg.y_range[0]), float(det_cfg.y_range[1])],
                    "voxel_size": float(det_cfg.voxel_size),
                }
                if det_cfg.enabled
                else None
            ),
            "keep_camera_poses": bool(args.keep_camera_poses),
            "keep_main_agent_poses": bool(args.keep_main_agent_poses),
            "frames_json": str(args.frames_json) if args.frames_json else None,
            # Legacy v1: only sequence/frame (kept so older dashboards don't break).
            "frames_hash_md5": frames_hash_md5,
            "frames_hash_md5_v2": frames_hash_md5_v2,
            "gt_scale_contract_json": str(args.gt_scale_contract_json) if args.gt_scale_contract_json else None,
            "gt_scale_contract_hash_md5": gt_scale_contract_hash,
            "gt_scale_contract_hash_md5_v2": gt_scale_contract_hash_v2,
            "images_root": str(args.images_root),
            "depth_root": str(args.depth_root),
            "eval_script_md5": str(eval_script_md5),
            # Versions are recorded to help investigate subtle metric drifts across environments.
            "python_version": str(sys.version),
            "python_executable": str(sys.executable),
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_cudnn_version": int(torch.backends.cudnn.version() or 0),
            "torch_device": str(device),
        },
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
