"""Helper utilities for converting MapAnything predictions to point clouds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

from mapanything.utils.geometry import depthmap_to_world_frame

try:  # Optional dependency used only for visualization or IO helpers
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in headless CI
    o3d = None
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)


def _mask_from_prediction(pred: dict) -> np.ndarray:
    mask = pred["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
    return mask


def predictions_to_pointcloud(
    predictions: Iterable[dict],
    *,
    colorize: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert MapAnything predictions into stacked XYZ (and optional RGB) arrays."""
    point_list = []
    color_list = []
    use_colors = colorize
    for pred in predictions:
        depth = pred["depth_z"][0].squeeze(-1)
        intrinsics = pred["intrinsics"][0]
        pose = pred["camera_poses"][0]
        pts3d_world, valid_mask = depthmap_to_world_frame(depth, intrinsics, pose)
        valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        mask_np = _mask_from_prediction(pred)
        final_mask = mask_np & valid_mask_np
        pts_np = pts3d_world.detach().cpu().numpy()
        point_list.append(pts_np[final_mask])
        if use_colors:
            colors = pred.get("img_no_norm")
            if colors is None:
                use_colors = False
            else:
                colors_np = colors[0].detach().cpu().numpy()
                color_list.append(colors_np[final_mask])
    if not point_list:
        raise ValueError("No prediction entries were provided")
    points = np.concatenate(point_list, axis=0)
    if use_colors and color_list:
        colors = np.concatenate(color_list, axis=0)
        colors = np.clip(colors, 0.0, 1.0)
        return points, colors
    return points, None


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load a point cloud from .pcd, .ply, .bin, or .npy."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".bin":
        data = np.fromfile(path, dtype=np.float32)
        return data.reshape(-1, 4)[:, :3]
    if suffix in {".pcd", ".ply"}:
        if OPEN3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points)
        raise ImportError(
            "open3d is required to load .pcd/.ply files. Install open3d or use .npy/.bin instead."
        )
    raise ValueError(f"Unsupported point cloud format: {path}")


def save_point_cloud(
    path: str | Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, points)
        return
    if suffix == ".pcd" and not OPEN3D_AVAILABLE:
        # Write a minimal ASCII PCD header
        header = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {points.shape[0]}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {points.shape[0]}",
            "DATA ascii",
        ]
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, points, fmt="%.6f")
        return
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d is required to save point clouds to this format")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(str(path), pcd)


def create_o3d_geometry(points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Create an Open3D geometry from numpy arrays."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d is not available")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd
