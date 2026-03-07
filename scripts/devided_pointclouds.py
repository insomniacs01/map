#!/usr/bin/env python3
"""Interactive visualization splitting predicted vs ground-truth clouds with camera toggles."""

import argparse
import logging
from pathlib import Path

import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.opv2v_pointclouds import (
    OPEN3D_AVAILABLE,
    create_o3d_geometry,
    load_point_cloud,
    predictions_to_pointcloud,
)

logger = logging.getLogger("devided_pointclouds")

if not OPEN3D_AVAILABLE:
    raise ImportError("open3d is required to run devided_pointclouds.py")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory containing RGB views")
    parser.add_argument("gt_pcd", type=Path, help="Ground-truth point cloud")
    parser.add_argument(
        "--model",
        default="facebook/map-anything",
        help="Model identifier or path for MapAnything.from_pretrained",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory efficient inference",
    )
    return parser


def _make_camera_frustums(predictions) -> list:
    if not OPEN3D_AVAILABLE:
        return []
    import open3d as o3d  # noqa: WPS433

    geoms = []
    for pred in predictions:
        pose = pred["camera_poses"][0].detach().cpu().numpy()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)
        frame.transform(pose)
        geoms.append(frame)
    return geoms


def _visualize(pred_geom, gt_geom, camera_geoms) -> None:
    if not OPEN3D_AVAILABLE:
        logger.warning("open3d missing; skipping visualization")
        return
    import open3d as o3d  # noqa: WPS433

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Predicted (color) vs Ground Truth (gray)")
    vis.add_geometry(pred_geom)
    vis.add_geometry(gt_geom)
    visible = {"cameras": False}

    def _toggle(vis_obj):  # noqa: WPS430
        visible["cameras"] = not visible["cameras"]
        logger.info("Camera axes %s", "ON" if visible["cameras"] else "OFF")
        for geom in camera_geoms:
            if visible["cameras"]:
                vis_obj.add_geometry(geom, reset_bounding_box=False)
            else:
                vis_obj.remove_geometry(geom, reset_bounding_box=False)
        return False

    vis.register_key_callback(ord("C"), _toggle)
    logger.info("Press 'C' to toggle camera coordinate frames")
    vis.run()
    vis.destroy_window()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _arg_parser()
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)
    if not args.gt_pcd.is_file():
        raise FileNotFoundError(args.gt_pcd)

    device = torch.device(args.device)
    model = MapAnything.from_pretrained(args.model).to(device)
    views = load_images(str(args.image_dir))
    if not views:
        raise RuntimeError(f"No images under {args.image_dir}")

    with torch.inference_mode():
        predictions = model.infer(views, memory_efficient_inference=args.memory_efficient)
    pred_points, pred_colors = predictions_to_pointcloud(predictions)
    pred_geom = create_o3d_geometry(pred_points, pred_colors)
    gt_points = load_point_cloud(args.gt_pcd)
    gt_geom = create_o3d_geometry(gt_points, None)
    gt_geom.paint_uniform_color([0.5, 0.5, 0.5])

    camera_geoms = _make_camera_frustums(predictions)
    _visualize(pred_geom, gt_geom, camera_geoms)


if __name__ == "__main__":
    main()
