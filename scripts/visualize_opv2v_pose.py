#!/usr/bin/env python3
"""
Visualize OPV2V point clouds together with transformed camera poses and boxes.

Example:
    python scripts/visualize_opv2v_pose.py \
        --dataset_root /media/.../OPV2V \
        --split train \
        --sequence 2021_08_16_22_26_54 \
        --agent 641 \
        --frame 000069 \
        --output outputs/opv2v_pose_vis.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data_processing.opv2v_pose_utils import load_opv2v_frame

EDGE_IDX: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def plot_bbox(ax, corners: np.ndarray, color: str = "lime") -> None:
    """Draw wireframe bounding box on the matplotlib axis."""
    for start, end in EDGE_IDX:
        xs = [corners[start, 0], corners[end, 0]]
        ys = [corners[start, 1], corners[end, 1]]
        zs = [corners[start, 2], corners[end, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1.0, alpha=0.9)


def plot_camera_pose(ax, pose: np.ndarray, name: str, axis_length: float = 1.5) -> None:
    """Draw camera origin and axes in ego coordinates."""
    origin = pose[:3, 3]
    axes = pose[:3, :3]
    colors = ("r", "g", "b")
    for axis_vec, color in zip(axes.T, colors):
        axis_vec = axis_vec / np.linalg.norm(axis_vec)
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_vec[0],
            axis_vec[1],
            axis_vec[2],
            color=color,
            linewidth=1.5,
            length=axis_length,
            arrow_length_ratio=0.2,
        )
    ax.text(
        origin[0],
        origin[1],
        origin[2],
        name,
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
    )


def set_axes_equal(ax) -> None:
    """Set 3D axes to equal scale to avoid distortion."""
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    span = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_span = max(span)
    for center, axis in zip(centers, "xyz"):
        getattr(ax, f"set_{axis}lim")(center - max_span / 2, center + max_span / 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OPV2V frame geometry.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/opv2v_pose_vis.png",
        help="Path to save the rendered figure.",
    )
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--bbox_range", type=float, default=120.0)
    parser.add_argument("--view_elev", type=float, default=90.0)
    parser.add_argument("--view_azim", type=float, default=-90.0)
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 11.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 12 11)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output image DPI/resolution (default: 400)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_dir = (
        Path(args.dataset_root) / args.split / args.sequence / args.agent
    )
    frame = load_opv2v_frame(frame_dir, args.frame, bbox_range=args.bbox_range)

    points = frame.points
    if args.max_points is not None and len(points) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[idx]

    fig = plt.figure(figsize=tuple(args.figsize))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=0.5,
        c=points[:, 0],
        cmap="viridis",
        alpha=0.6,
    )

    for cam_name, pose in frame.camera_poses.items():
        plot_camera_pose(ax, pose, cam_name)

    for _, bbox in frame.vehicle_bboxes.items():
        plot_bbox(ax, bbox["corners"])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"{args.sequence}/{args.agent}/{args.frame} — cameras & boxes in ego frame"
    )
    ax.view_init(elev=args.view_elev, azim=args.view_azim)
    set_axes_equal(ax)
    ax.dist = 10

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
