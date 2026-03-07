#!/usr/bin/env python3
"""
Simple point-cloud filtering + BEV/Chamfer metrics for OPV2V predictions.

Usage example:
    python scripts/filter_pointcloud.py \
        --pred visualization_outputs/stage2_coop_000069_pred.pcd \
        --gt /media/.../OPV2V/train/2021_08_16_22_26_54/641/000069.pcd \
        --z_min -2 --z_max 4 --radius_max 120 \
        --save_filtered visualization_outputs/stage2_coop_000069_filtered.pcd
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover - open3d may not be available in tests
    raise ImportError("open3d is required for this script") from exc

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing.opv2v_pose_utils import (  # type: ignore  # noqa: E402
    get_camera_poses_in_ego,
    load_frame_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, help="Predicted point cloud (.pcd/.ply/.xyz)")
    parser.add_argument("--gt", type=Path, help="Ground-truth point cloud for metrics")
    parser.add_argument("--z_min", type=float, default=None, help="Minimum allowed Z (meters)")
    parser.add_argument("--z_max", type=float, default=None, help="Maximum allowed Z (meters)")
    parser.add_argument("--radius_max", type=float, default=None, help="Horizontal radius limit sqrt(x^2+y^2)")
    parser.add_argument("--save_filtered", type=Path, help="Optional path to save filtered cloud")
    parser.add_argument("--bev_range", type=float, default=120.0, help="BEV range in meters for occupancy map")
    parser.add_argument("--bev_resolution", type=float, default=0.5, help="BEV grid resolution (meters)")
    parser.add_argument("--pair_list", type=Path, help="Optional CSV with columns pred,gt for batch processing")
    parser.add_argument(
        "--camera_yaml",
        action="append",
        help="Paths to OPV2V frame YAML files to enable multi-view consistency filtering",
    )
    parser.add_argument("--min_visible_views", type=int, default=1, help="Keep points seen by >= N cameras")
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points)


def filter_points(
    points: np.ndarray,
    z_min: float | None,
    z_max: float | None,
    radius_max: float | None,
    cameras: List[Dict] | None = None,
    min_visible_views: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    if radius_max is not None:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= radius_max
    if cameras and min_visible_views > 1:
        mask &= multi_view_visibility(points, cameras, min_visible_views)
    return points[mask], mask


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"pred_to_gt": float("nan"), "gt_to_pred": float("nan"), "mean": float("nan")}
    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    dist_pred_gt, _ = gt_tree.query(pred, k=1)
    dist_gt_pred, _ = pred_tree.query(gt, k=1)
    return {
        "pred_to_gt": float(np.mean(dist_pred_gt)),
        "gt_to_pred": float(np.mean(dist_gt_pred)),
        "mean": float(np.mean([np.mean(dist_pred_gt), np.mean(dist_gt_pred)])),
    }


def bev_occupancy(points: np.ndarray, bev_range: float, bev_resolution: float) -> np.ndarray:
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


def bev_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def load_cameras_from_yaml(paths: Iterable[Path]) -> List[Dict]:
    cameras: List[Dict] = []
    for path in paths:
        meta = load_frame_metadata(path)
        cam_poses = get_camera_poses_in_ego(meta)
        for cam_key, pose_ego in cam_poses.items():
            intrinsic = np.asarray(meta[cam_key]["intrinsic"], dtype=np.float64)
            width = int(round(intrinsic[0, 2] * 2))
            height = int(round(intrinsic[1, 2] * 2))
            cameras.append(
                {
                    "T_cam_ego": np.linalg.inv(pose_ego),
                    "intrinsic": intrinsic,
                    "width": width,
                    "height": height,
                    "name": f"{path.name}:{cam_key}",
                }
            )
    return cameras


def multi_view_visibility(points: np.ndarray, cameras: List[Dict], min_views: int) -> np.ndarray:
    if not cameras:
        return np.ones(points.shape[0], dtype=bool)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    counts = np.zeros(points.shape[0], dtype=np.int32)
    for cam in cameras:
        T_cam_ego = cam["T_cam_ego"]
        intrinsic = cam["intrinsic"]
        pts_cam = (T_cam_ego @ pts_h.T).T
        z = pts_cam[:, 2]
        in_front = z > 1e-3
        if not np.any(in_front):
            continue
        x = intrinsic[0, 0] * (pts_cam[:, 0] / z) + intrinsic[0, 2]
        y = intrinsic[1, 1] * (pts_cam[:, 1] / z) + intrinsic[1, 2]
        valid = (
            in_front
            & (x >= 0)
            & (x < cam["width"])
            & (y >= 0)
            & (y < cam["height"])
        )
        counts[valid] += 1
    return counts >= min_views


def process_pair(
    pred_path: Path,
    gt_path: Path | None,
    args: argparse.Namespace,
    cameras: List[Dict] | None,
) -> None:
    pred_points = load_points(pred_path)
    filtered_points, mask = filter_points(
        pred_points,
        args.z_min,
        args.z_max,
        args.radius_max,
        cameras=cameras,
        min_visible_views=args.min_visible_views,
    )

    print(f"[INFO] Loaded {pred_points.shape[0]} points from {pred_path}")
    print(
        f"[INFO] Filtered points: {filtered_points.shape[0]} "
        f"({filtered_points.shape[0] / max(1, pred_points.shape[0]):.2%} retained)"
    )

    if args.save_filtered and not args.pair_list:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(filtered_points)
        o3d.io.write_point_cloud(str(args.save_filtered), cloud)
        print(f"[INFO] Saved filtered cloud to {args.save_filtered}")

    if gt_path:
        gt_points = load_points(gt_path)
        chamfer_raw = chamfer_distance(pred_points, gt_points)
        chamfer_filtered = chamfer_distance(filtered_points, gt_points)
        bev_raw = bev_occupancy(pred_points, args.bev_range, args.bev_resolution)
        bev_filtered = bev_occupancy(filtered_points, args.bev_range, args.bev_resolution)
        bev_gt = bev_occupancy(gt_points, args.bev_range, args.bev_resolution)
        print("[METRIC] Chamfer (raw):", chamfer_raw)
        print("[METRIC] Chamfer (filtered):", chamfer_filtered)
        print("[METRIC] BEV IoU (raw):", bev_iou(bev_raw, bev_gt))
        print("[METRIC] BEV IoU (filtered):", bev_iou(bev_filtered, bev_gt))

    if args.save_filtered and args.pair_list:
        out_path = pred_path.with_name(pred_path.stem + "_filtered.pcd")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(filtered_points)
        o3d.io.write_point_cloud(str(out_path), cloud)
        print(f"[INFO] Saved filtered cloud to {out_path}")


def main() -> None:
    args = parse_args()
    if not args.pred and not args.pair_list:
        raise ValueError("Either --pred or --pair_list must be provided.")

    cameras = load_cameras_from_yaml([Path(p) for p in args.camera_yaml]) if args.camera_yaml else None

    if args.pair_list:
        import csv

        with args.pair_list.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pred_path = Path(row["pred"]).expanduser()
                gt_path = Path(row["gt"]).expanduser() if row.get("gt") else None
                print("=" * 60)
                process_pair(pred_path, gt_path, args, cameras)
    else:
        process_pair(args.pred, args.gt, args, cameras)


if __name__ == "__main__":
    main()
