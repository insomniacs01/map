#!/usr/bin/env python3
"""Visualize predicted/ground-truth point clouds with OPV2V vehicle boxes and camera frames."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import List

import numpy as np

from data_processing.opv2v_pose_utils import (
    cords_to_pose,
    get_vehicle_bboxes_in_ego,
    load_frame_metadata,
)
from mapanything.utils.opv2v_pointclouds import OPEN3D_AVAILABLE, create_o3d_geometry, load_point_cloud

try:
    import open3d as o3d  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    o3d = None

logger = logging.getLogger("pcd_with_boxes")

# Carla/UE (X forward, Y right, Z up) -> OpenCV (X right, Y down, Z forward)
_CARLA_TO_OPENCV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_path", type=Path, help="OPV2V frame YAML containing vehicle annotations")
    parser.add_argument("pred_pcd", type=Path, help="Predicted point cloud path")
    parser.add_argument("gt_pcd", type=Path, help="Ground truth point cloud path")
    parser.add_argument(
        "--max_range",
        type=float,
        default=120.0,
        help="Maximum distance (m) for rendering vehicle bounding boxes",
    )
    parser.add_argument(
        "--camera_size",
        type=float,
        default=2.0,
        help="Size of the coordinate frame drawn for each camera",
    )
    parser.add_argument(
        "--pred_coord",
        type=str,
        default="auto",
        choices=("auto", "carla", "opencv"),
        help=(
            "Coordinate convention of the predicted point cloud. "
            "'opencv' points will be converted to CARLA/UE for box overlay."
        ),
    )
    return parser


def _infer_coord_convention(points: np.ndarray) -> str:
    """Heuristic to detect whether points are likely CARLA or OpenCV convention."""
    if points.size == 0:
        return "unknown"
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    span = hi - lo
    # In driving scenes, CARLA's height (Z) span is usually smaller than OpenCV's Y span (since Y_down ~= -Z_up).
    if span[2] <= span[1]:
        return "carla"
    return "opencv"


def _opencv_to_carla(points_cv: np.ndarray) -> np.ndarray:
    """Convert Nx3 points from OpenCV convention to CARLA/UE convention."""
    if points_cv.size == 0:
        return points_cv
    return (points_cv @ _CARLA_TO_OPENCV).astype(np.float32, copy=False)


def _bbox_lines(corners: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    # 8 corners -> 12 edges
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def _camera_axes(frame_meta: dict, size: float) -> List[o3d.geometry.TriangleMesh]:
    lidar_pose = cords_to_pose(frame_meta["lidar_pose"])
    T_ego_world = np.linalg.inv(lidar_pose)
    geoms: List[o3d.geometry.TriangleMesh] = []
    for key, value in frame_meta.items():
        if not key.startswith("camera"):
            continue
        C2W = cords_to_pose(value["cords"])
        C2E = T_ego_world @ C2W
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(C2E)
        geoms.append(frame)
    return geoms


def _visualize(pred_geom, gt_geom, extras: List[o3d.geometry.Geometry]) -> None:
    vis_pred = o3d.visualization.Visualizer()
    vis_gt = o3d.visualization.Visualizer()
    vis_pred.create_window(window_name="Prediction")
    vis_gt.create_window(window_name="Ground Truth")
    for geom in [pred_geom, *extras]:
        vis_pred.add_geometry(geom, reset_bounding_box=False)
    for geom in [gt_geom, *extras]:
        vis_gt.add_geometry(copy.deepcopy(geom), reset_bounding_box=False)

    while True:
        keep_pred = vis_pred.poll_events()
        keep_gt = vis_gt.poll_events()
        if not keep_pred and not keep_gt:
            break
        vis_pred.update_renderer()
        vis_gt.update_renderer()
    vis_pred.destroy_window()
    vis_gt.destroy_window()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _arg_parser()
    args = parser.parse_args()

    if not OPEN3D_AVAILABLE or o3d is None:
        raise ImportError("open3d is required for scripts/pcd_with_boxes.py. Please `pip install open3d`.")

    frame_meta = load_frame_metadata(args.yaml_path)
    vehicles = get_vehicle_bboxes_in_ego(frame_meta, max_range=args.max_range)
    bbox_geoms = []
    for idx, info in vehicles.items():
        seed = abs(hash(idx)) % (2**32)
        color = np.random.default_rng(seed).uniform(0.2, 1.0, size=3)
        bbox_geoms.append(_bbox_lines(info["corners"], color))

    camera_geoms = _camera_axes(frame_meta, args.camera_size)

    pred_points = load_point_cloud(args.pred_pcd)
    pred_coord = str(args.pred_coord).lower()
    if pred_coord == "auto":
        pred_coord = _infer_coord_convention(pred_points)
        logger.info("Detected pred_pcd coordinate convention: %s", pred_coord)
    if pred_coord == "opencv":
        pred_points = _opencv_to_carla(pred_points)
    elif pred_coord not in ("carla", "unknown"):
        raise ValueError(f"Unsupported --pred_coord: {args.pred_coord}")

    pred_geom = create_o3d_geometry(pred_points)
    gt_geom = create_o3d_geometry(load_point_cloud(args.gt_pcd))
    gt_geom.paint_uniform_color([0.4, 0.4, 0.4])

    _visualize(pred_geom, gt_geom, bbox_geoms + camera_geoms)


if __name__ == "__main__":
    main()
