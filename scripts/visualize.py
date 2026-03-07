#!/usr/bin/env python3
"""Minimal Open3D viewer for point clouds."""

import argparse
import logging

from mapanything.utils.opv2v_pointclouds import OPEN3D_AVAILABLE, create_o3d_geometry, load_point_cloud

if not OPEN3D_AVAILABLE:
    raise ImportError("open3d is required for scripts/visualize.py")

import open3d as o3d  # noqa: E402

logger = logging.getLogger("visualize")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd_path", help="Point cloud file (.pcd/.ply/.npy/.bin)")
    parser.add_argument(
        "--coord_size",
        type=float,
        default=5.0,
        help="Size of origin coordinate frame",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _arg_parser()
    args = parser.parse_args()

    points = load_point_cloud(args.pcd_path)
    logger.info("Loaded %d points", points.shape[0])
    geom = create_o3d_geometry(points)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.coord_size)
    o3d.visualization.draw_geometries([geom, axis])


if __name__ == "__main__":
    main()
