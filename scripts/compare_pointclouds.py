#!/usr/bin/env python3
"""Quick point-cloud comparison utility for MapAnything predictions."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.opv2v_pointclouds import (
    OPEN3D_AVAILABLE,
    create_o3d_geometry,
    load_point_cloud,
    predictions_to_pointcloud,
    save_point_cloud,
)

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected to be available but guard anyway
    cKDTree = None

logger = logging.getLogger("compare_pointclouds")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory containing multi-view RGBs")
    parser.add_argument("gt_pcd", type=Path, help="Ground-truth point cloud to compare against")
    parser.add_argument(
        "--model",
        default="facebook/map-anything",
        help="Model identifier or local directory for MapAnything.from_pretrained",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on",
    )
    parser.add_argument(
        "--save_pred",
        type=Path,
        help="Optional path to save the predicted point cloud (.pcd/.ply/.npy)",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip Open3D visualization even if the library is installed",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use the memory efficient inference mode",
    )
    return parser


def _nearest_neighbor_stats(pred: np.ndarray, gt: np.ndarray) -> Optional[dict]:
    if cKDTree is None:
        logger.warning("scipy not available; skipping distance metrics")
        return None
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": float(np.percentile(dists, 90)),
    }


def _visualize(pred_points: np.ndarray, gt_points: np.ndarray, colors: Optional[np.ndarray], title: str) -> None:
    if not OPEN3D_AVAILABLE:
        logger.warning("open3d not installed; cannot launch viewer")
        return
    pred_geom = create_o3d_geometry(pred_points, colors)
    gt_geom = create_o3d_geometry(gt_points, None)
    gt_geom.paint_uniform_color([0.6, 0.6, 0.6])
    import open3d as o3d  # noqa: WPS433

    o3d.visualization.draw_geometries(
        [pred_geom, gt_geom],
        window_name=title,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.gt_pcd.is_file():
        raise FileNotFoundError(f"Ground-truth PCD not found: {args.gt_pcd}")

    device = torch.device(args.device)
    logger.info("Loading model %s on %s", args.model, device)
    model = MapAnything.from_pretrained(args.model).to(device)

    logger.info("Loading multi-view inputs from %s", args.image_dir)
    views = load_images(str(args.image_dir))
    if not views:
        raise RuntimeError(f"No images found under {args.image_dir}")

    logger.info("Running inference on %d views", len(views))
    with torch.inference_mode():
        predictions = model.infer(
            views,
            memory_efficient_inference=args.memory_efficient,
        )
    pred_points, pred_colors = predictions_to_pointcloud(predictions)
    logger.info("Predicted %d points", pred_points.shape[0])

    gt_points = load_point_cloud(args.gt_pcd)
    logger.info("Ground truth contains %d points", gt_points.shape[0])

    stats = _nearest_neighbor_stats(pred_points, gt_points)
    if stats:
        logger.info(
            "NN distance stats (m): mean=%.4f median=%.4f p90=%.4f",
            stats["mean"],
            stats["median"],
            stats["p90"],
        )

    if args.save_pred:
        save_point_cloud(args.save_pred, pred_points, pred_colors)
        logger.info("Saved predicted point cloud to %s", args.save_pred)

    if not args.no_viz:
        _visualize(pred_points, gt_points, pred_colors, "MapAnything vs Ground Truth")


if __name__ == "__main__":
    main()
