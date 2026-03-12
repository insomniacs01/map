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
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

# Allow imports from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from color_compare import (  # type: ignore  # noqa: E402
    IMAGE_EXTENSIONS,
    preprocess_inputs,
    rescale_intrinsics_to_image,
    strip_external_calibration_inputs,
)
from mapanything.datasets.opv2v import _convert_pose_to_opencv  # type: ignore  # noqa: E402
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local  # noqa: E402
from mapanything.utils.opv2v_pointclouds import predictions_to_pointcloud, save_point_cloud  # noqa: E402

from data_processing.opv2v_pose_utils import (  # type: ignore  # noqa: E402
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


@dataclass
class EvalResult:
    frame: FrameInfo
    mode: str  # "single" or "coop"
    pose_abs: float
    pose_rot: float
    depth_rmse: float
    depth_mae: float
    depth_rel: float
    scale_err: float | None
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


@dataclass
class PCMetricConfig:
    enabled: bool = False
    z_min: float | None = None
    z_max: float | None = None
    radius_max: float | None = None
    bev_range: float = 120.0
    bev_resolution: float = 0.5
    save_dir: Path | None = None




def _checkpoint_model_spec(checkpoint_path: str) -> Tuple[str | None, dict | None]:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if args is None:
        return None, None

    model_cfg = getattr(args, "model", None)
    if model_cfg is None:
        return None, None

    model_str = getattr(model_cfg, "model_str", None)
    model_config = getattr(model_cfg, "model_config", None)
    return model_str, model_config

DEFAULT_MODELS: Dict[str, str] = {
    "pretrain": "/home/qqxluca/map-anything3/checkpoints/facebook_map-anything-local.pth",
    "stage1": "/home/qqxluca/map-anything3/experiments/opv2v_ft_stage1/checkpoint-best.pth",
    "stage2": "/home/qqxluca/map-anything3/experiments/opv2v_coop_stage2_full/checkpoint-best.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="Dataset split (train/validate/test)")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of frames to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--frames_json", type=Path, help="Optional JSON containing explicit frames (uses 'frames' key)")
    parser.add_argument("--images_root", type=Path, default=Path("/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V"))
    parser.add_argument("--depth_root", type=Path, default=Path("/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/opv2v_depth"))
    parser.add_argument("--output_root", type=Path, default=Path("/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/xiongyijin_workspace/opv2v_batch_eval"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
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
    return parser.parse_args()


def discover_frames(images_root: Path, split: str) -> List[FrameInfo]:
    split_root = images_root / split
    frames: List[FrameInfo] = []
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split directory missing: {split_root}")

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
            frames.append(FrameInfo(sequence=seq_dir.name, frame=frame, main_agent=main_agent, coop_agents=coop_agents))
    return frames


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
            frames.append(
                FrameInfo(
                    sequence=item["sequence"],
                    frame=item["frame"],
                    main_agent=item.get("main_agent", "641"),
                    coop_agents=tuple(item.get("coop_agents", (item.get("main_agent", "641"),))),
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
        camera_infos.append({"name": cam_key, "pose_C2W_cv": torch.tensor(pose_world_cv, dtype=torch.float32)})
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


def _compute_pose_metrics(predictions, camera_info_list) -> Tuple[float, float]:
    if not camera_info_list or not predictions:
        return float("nan"), float("nan")
    pose_key = "pose_C2W_cv" if "pose_C2W_cv" in camera_info_list[0] else "pose_C2E_cv"
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
    if not abs_list:
        return float("nan"), float("nan")
    return float(np.mean(abs_list)), float(np.mean(rot_list))


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


def _gt_scale_factor_from_raw_views(raw_views: Sequence[Dict]) -> float | None:
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
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        if fx == 0.0 or fy == 0.0:
            continue

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
    except Exception:
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
        except Exception:
            continue

        if intr.shape != (3, 3) or pose.shape != (4, 4):
            continue

        try:
            T_ref_cur = pose_ref_inv @ pose
        except Exception:
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
        pts = np.stack([x, y, z], axis=0)
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
    vals = []
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
            used_fallback = True
            vals = [float(fallback_scale_factor)] * max(1, len(predictions))
        else:
            return None

    ratios = np.asarray(vals, dtype=np.float64)
    ratios = np.clip(ratios, 1e-8, None)
    out: Dict[str, float] = {}
    if not used_fallback:
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
) -> Dict[str, List[EvalResult]]:
    if not modes:
        raise ValueError("At least one evaluation mode must be specified.")
    checkpoint_model_str, checkpoint_model_config = _checkpoint_model_spec(checkpoint)
    cfg = {
        "path": str(REPO_ROOT / "configs/train.yaml"),
        "checkpoint_path": checkpoint,
        "config_overrides": [
            "machine=local3090",
            "dataset=opv2v_ft",
            "model=mapanything_opv2v_coop_lowmem",
            "model/task=images_only",
            "model.encoder.uses_torch_hub=false",
            "loss=overall_loss",
        ],
    }
    if checkpoint_model_str is not None:
        cfg["model_str"] = checkpoint_model_str
    if checkpoint_model_config is not None:
        cfg["model_config"] = checkpoint_model_config
    model = initialize_mapanything_local(cfg, device)
    model.eval()

    per_mode_results: Dict[str, List[EvalResult]] = {mode: [] for mode in modes}
    representative_candidates: Dict[str, List[Tuple[float, FrameInfo, List]]] = {mode: [] for mode in modes}

    for info in frames:
        for mode in modes:
            try:
                if mode == "single":
                    raw_views, camera_info = build_single_raw(images_root, depth_root, split, info)
                else:
                    raw_views, camera_info = build_coop_raw(images_root, depth_root, split, info)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Skip {info.sequence}/{info.frame} ({mode}): {exc}")
                continue

            processed_views = preprocess_inputs(raw_views)
            gt_depths = [view["depth_z"].clone() for view in processed_views if "depth_z" in view]
            for view in processed_views:
                if "depth_z" in view:
                    view.pop("depth_z")
            strip_external_calibration_inputs(processed_views)
            print(
                f"[INFO] Stripped external intrinsics/poses for {info.sequence}/{info.frame} ({mode}) "
                "so inference remains image-only."
            )
            with torch.no_grad():
                predictions = model.infer(processed_views, memory_efficient_inference=True)
            pred_points, _ = predictions_to_pointcloud(predictions, colorize=True)
            pose_abs, pose_rot = _compute_pose_metrics(predictions, camera_info)
            depth_rmse, depth_mae, depth_rel = depth_metrics(predictions, gt_depths)
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
                if pc_metrics_cfg.save_dir:
                    save_dir = pc_metrics_cfg.save_dir / model_name / mode
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{info.sequence}_{info.frame}.npy"
                    np.save(save_path, pred_points.astype(np.float32))
                if pc_metrics_cfg.enabled:
                    gt_pcd_path = images_root / split / info.sequence / info.main_agent / f"{info.frame}.pcd"
                    if not gt_pcd_path.is_file():
                        print(f"[WARN] GT point cloud missing for detection metrics: {gt_pcd_path}")
                    else:
                        gt_points = load_ascii_pcd_xyz(gt_pcd_path)
                        detection_metrics = compute_detection_metrics(pred_points, gt_points, pc_metrics_cfg)

            per_mode_results[mode].append(
                EvalResult(
                    frame=info,
                    mode=mode,
                    pose_abs=pose_abs,
                    pose_rot=pose_rot,
                    depth_rmse=depth_rmse,
                    depth_mae=depth_mae,
                    depth_rel=depth_rel,
                    scale_err=(scale_details.get("scale_err") if scale_details is not None else None),
                    scale_log_err=(scale_details.get("scale_log_err") if scale_details is not None else None),
                    scale_eq_rel_err=(scale_details.get("scale_eq_rel_err") if scale_details is not None else None),
                    scale_ratio_mean=(scale_details.get("scale_ratio_mean") if scale_details is not None else None),
                    scale_ratio_median=(scale_details.get("scale_ratio_median") if scale_details is not None else None),
                    scale_ratio_p90=(scale_details.get("scale_ratio_p90") if scale_details is not None else None),
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
                )
            )
            representative_candidates[mode].append((depth_rmse if not math.isnan(depth_rmse) else float("inf"), info, predictions))

    # Save representative point clouds if requested
    if save_representative:
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
            for tag, (_, frame_info, preds) in selected:
                pts, _ = predictions_to_pointcloud(preds, colorize=True)
                out_path = rep_dir / f"{frame_info.sequence}_{frame_info.frame}_{tag}.pcd"
                save_point_cloud(out_path, pts)

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
            "depth_rmse_mean": float(np.nanmean([v.depth_rmse for v in values])),
            "depth_mae_mean": float(np.nanmean([v.depth_mae for v in values])),
            "depth_rel_mean": float(np.nanmean([v.depth_rel for v in values])),
            "scale_err_mean": float(np.nanmean([v.scale_err for v in values if v.scale_err is not None])) if any(v.scale_err is not None for v in values) else float("nan"),
            "scale_log_err_mean": _attr_mean("scale_log_err"),
            "scale_eq_rel_err_mean": _attr_mean("scale_eq_rel_err"),
            "scale_ratio_mean": _attr_mean("scale_ratio_mean"),
            "scale_ratio_median_mean": _attr_mean("scale_ratio_median"),
            "scale_ratio_p90_mean": _attr_mean("scale_ratio_p90"),
            "scale_gt_factor_mean": _attr_mean("scale_gt_factor"),
            "scale_to_gt_err_mean": _attr_mean("scale_to_gt_err"),
            "scale_to_gt_log_err_mean": _attr_mean("scale_to_gt_log_err"),
            "scale_to_gt_eq_rel_err_mean": _attr_mean("scale_to_gt_eq_rel_err"),
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
                    "pose_abs_m",
                    "pose_rot_deg",
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
                ]
            )
            for v in values:
                writer.writerow(
                    [
                        v.frame.sequence,
                        v.frame.frame,
                        f"{v.pose_abs:.6f}",
                        f"{v.pose_rot:.6f}",
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
                    ]
                )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" else "cuda")
    modes = tuple(dict.fromkeys(args.modes)) if args.modes else ("single", "coop")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    frames = discover_frames(args.images_root, args.split)
    sampled = load_frames_from_json(args.frames_json) if args.frames_json else sample_frames(frames, args.sample_size, args.seed)
    if not sampled:
        raise RuntimeError("No frames discovered for evaluation.")

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
        )
        save_metrics_csv(results, output_root, name)
        overall_summary[name] = summarize_results(results)

    summary_path = output_root / f"summary_{args.split}.json"
    with summary_path.open("w") as fh:
        json.dump({"split": args.split, "frames": [f.__dict__ for f in sampled], "metrics": overall_summary}, fh, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
