from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

CV_TO_UE_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    if colors.size == 0:
        return colors.astype(np.float32)
    colors = colors.astype(np.float32, copy=False)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def predictions_to_pointcloud(
    predictions: Iterable[dict], colorize: bool = False
) -> Tuple[np.ndarray, np.ndarray | None]:
    aggregated_points = []
    aggregated_colors = []

    for pred in predictions:
        if "pts3d" not in pred:
            continue

        pts3d = pred["pts3d"]
        if pts3d is None:
            continue
        pts_np = pts3d[0].detach().cpu().numpy().reshape(-1, 3)

        mask_tensor = pred.get("mask")
        if mask_tensor is not None:
            mask_np = mask_tensor[0].detach().cpu().numpy().astype(bool).reshape(-1)
            if mask_np.shape[0] == pts_np.shape[0]:
                pts_np = pts_np[mask_np]
            else:
                mask_np = None
        else:
            mask_np = None

        if pts_np.size == 0:
            continue

        finite_mask = np.isfinite(pts_np).all(axis=1)
        pts_np = pts_np[finite_mask]
        if pts_np.size == 0:
            continue

        pts_ue = pts_np @ CV_TO_UE_ROTATION.T
        aggregated_points.append(pts_ue.astype(np.float32, copy=False))

        if colorize:
            img = pred.get("img_no_norm")
            if img is not None:
                colors_np = img[0].detach().cpu().numpy().reshape(-1, 3)
                if mask_np is not None and mask_np.shape[0] == colors_np.shape[0]:
                    colors_np = colors_np[mask_np]
                if finite_mask.shape[0] == colors_np.shape[0]:
                    colors_np = colors_np[finite_mask]
                else:
                    colors_np = np.tile(
                        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                        (pts_ue.shape[0], 1),
                    )
            else:
                colors_np = np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    (pts_ue.shape[0], 1),
                )
            aggregated_colors.append(_normalize_colors(colors_np))

    if not aggregated_points:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, (empty.copy() if colorize else None)

    points = np.concatenate(aggregated_points, axis=0)
    if not colorize:
        return points, None

    if aggregated_colors:
        colors = np.concatenate(aggregated_colors, axis=0)
    else:
        colors = np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            (points.shape[0], 1),
        )
    return points, colors.astype(np.float32, copy=False)


def _pack_rgb_float(colors: np.ndarray) -> np.ndarray:
    rgb_uint8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint32)
    packed = (rgb_uint8[:, 0] << 16) | (rgb_uint8[:, 1] << 8) | rgb_uint8[:, 2]
    return packed.view(np.float32)


def save_point_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    if path.suffix.lower() == ".npy":
        np.save(path, points)
        return

    if colors is not None:
        colors = _normalize_colors(np.asarray(colors))
        if colors.shape[0] != points.shape[0]:
            colors = None

    if path.suffix.lower() == ".pcd":
        if colors is None:
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {points.shape[0]}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {points.shape[0]}
DATA ascii
"""
            with path.open("w", encoding="utf-8") as fh:
                fh.write(header)
                for x, y, z in points:
                    fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            return

        rgb_floats = _pack_rgb_float(colors)
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {points.shape[0]}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {points.shape[0]}
DATA ascii
"""
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            for (x, y, z), rgb in zip(points, rgb_floats):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb:.8e}\n")
        return

    if path.suffix.lower() == ".ply":
        with path.open("w", encoding="utf-8") as fh:
            fh.write("ply\n")
            fh.write("format ascii 1.0\n")
            fh.write(f"element vertex {points.shape[0]}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            if colors is not None:
                fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write("end_header\n")
            if colors is None:
                for x, y, z in points:
                    fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            else:
                rgb_uint8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
                for (x, y, z), (r, g, b) in zip(points, rgb_uint8):
                    fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        return

    raise ValueError(f"Unsupported point cloud extension: {path.suffix}")
