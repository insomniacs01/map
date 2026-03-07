#!/usr/bin/env python3
"""Project ASCII/PLY point clouds into camera depth images using OPV2V metadata."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from data_processing.opv2v_pose_utils import cords_to_pose, load_frame_metadata
from mapanything.utils.opv2v_pointclouds import load_point_cloud


def _project(points: np.ndarray, pose: np.ndarray, intr: np.ndarray, resolution) -> np.ndarray:
    width, height = resolution
    homo = np.c_[points, np.ones(points.shape[0])]
    W2C = np.linalg.inv(pose)
    pts_cam = (W2C @ homo.T).T
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pixels = (intr @ pts_cam[:, :3].T).T
    pixels[:, 0] /= pixels[:, 2]
    pixels[:, 1] /= pixels[:, 2]
    depth_img = np.full((height, width), np.inf, dtype=np.float32)
    for u, v, z in pixels:
        x = int(round(u))
        y = int(round(v))
        if 0 <= x < width and 0 <= y < height and z > 0:
            if z < depth_img[y, x]:
                depth_img[y, x] = z
    depth_img[~np.isfinite(depth_img)] = 0.0
    return depth_img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd_path", type=Path)
    parser.add_argument("yaml_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("depth_outputs"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 540))
    args = parser.parse_args()

    points = load_point_cloud(args.pcd_path)
    meta = load_frame_metadata(args.yaml_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cam_name, cam_cfg in meta.items():
        if not cam_name.startswith("camera"):
            continue
        pose = cords_to_pose(cam_cfg["cords"])
        intr = np.asarray(cam_cfg["intrinsic"], dtype=np.float32)
        depth = _project(points, pose, intr, args.resolution)
        np.save(args.output_dir / f"{cam_name}.npy", depth)
        depth_png = (depth * 1000.0).astype(np.uint16)
        Image.fromarray(depth_png).save(args.output_dir / f"{cam_name}.png")


if __name__ == "__main__":
    main()
