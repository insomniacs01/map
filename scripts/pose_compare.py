#!/usr/bin/env python3
"""Batch process OPV2V YAML configs, run MapAnything, and compare predicted poses/point clouds."""

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.opv2v_pointclouds import (
    OPEN3D_AVAILABLE,
    create_o3d_geometry,
    predictions_to_pointcloud,
    save_point_cloud,
)
from mapanything.utils.opv2v_viz import load_views_from_config, log_camera_pose_errors

if OPEN3D_AVAILABLE:
    import open3d as o3d  # noqa: E402

logger = logging.getLogger("pose_compare")
CV_TO_UE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)


def _iter_configs(args) -> Iterable[Path]:
    if args.config_path:
        yield args.config_path
    if args.config_dir:
        for yaml_file in sorted(args.config_dir.glob("*.yaml")):
            yield yaml_file


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory that stores <stem>_cameraX.png images")
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Single YAML config to process",
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        help="Directory of YAML configs to process",
    )
    parser.add_argument(
        "--model",
        default="facebook/map-anything",
        help="Model identifier or local directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Optional directory to save per-frame predictions (.pcd)",
    )
    parser.add_argument(
        "--aggregate_export",
        type=Path,
        help="Path to save the aggregated prediction cloud",
    )
    parser.add_argument(
        "--export_frame",
        choices=["opencv", "ue"],
        default="opencv",
        help="Coordinate frame for exported points",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory efficient inference",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable Open3D visualization",
    )
    return parser


def _convert_frame(points: np.ndarray, frame: str) -> np.ndarray:
    if frame == "opencv":
        return points
    return points @ CV_TO_UE.T


def _visualize(point_sets: List[np.ndarray]) -> None:
    if not OPEN3D_AVAILABLE:
        logger.warning("open3d not available; skipping visualization")
        return
    geoms = []
    for pts in point_sets:
        geoms.append(create_o3d_geometry(pts))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    o3d.visualization.draw_geometries([*geoms, mesh])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _arg_parser()
    args = parser.parse_args()

    configs = list(_iter_configs(args))
    if not configs:
        raise RuntimeError("No YAML configs were provided")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = MapAnything.from_pretrained(args.model).to(device)
    aggregated_points: List[np.ndarray] = []

    for cfg_path in configs:
        views, camera_info = load_views_from_config(cfg_path, args.image_dir, device)
        with torch.inference_mode():
            predictions = model.infer(views, memory_efficient_inference=args.memory_efficient)
        log_camera_pose_errors(predictions, camera_info)
        pts, _ = predictions_to_pointcloud(predictions)
        pts = _convert_frame(pts, args.export_frame)
        aggregated_points.append(pts)
        if args.save_dir:
            out_path = args.save_dir / f"{cfg_path.stem}_{args.export_frame}.pcd"
            save_point_cloud(out_path, pts)
            logger.info("Saved %s", out_path)

    if args.aggregate_export:
        args.aggregate_export.parent.mkdir(parents=True, exist_ok=True)
        agg = np.concatenate(aggregated_points, axis=0)
        save_point_cloud(args.aggregate_export, agg)
        logger.info("Aggregated cloud saved to %s", args.aggregate_export)

    if not args.no_viz:
        _visualize(aggregated_points)


if __name__ == "__main__":
    main()
