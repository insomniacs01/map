"""
Utilities for converting OPV2V poses into ego-centric coordinates.

The OPV2V metadata stores camera, LiDAR and vehicle poses as
``[x, y, z, roll, yaw, pitch]`` with angles expressed in degrees in the CARLA
right-handed coordinate system (X forward, Y right, Z up). The angles follow
an internal ordering, so we convert them into rotation matrices using the
empirically verified mapping defined below and then build 4x4 homogeneous
transforms for convenient chaining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# Carla/UE uses X-forward, Y-right, Z-up. OpenCV expects X-right, Y-down, Z-forward.
# This matrix converts vectors expressed in Carla coordinates into OpenCV coordinates.
CARLA_TO_CAMERA_CV = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _angles_to_rotation(angles_deg: Iterable[float]) -> np.ndarray:
    """Convert OPV2V angle triplet to a rotation matrix."""
    angles_deg = list(angles_deg)
    if len(angles_deg) != 3:
        raise ValueError(f"Expected 3 angles, got {len(angles_deg)}")

    roll_deg, yaw_deg, pitch_deg = angles_deg
    # OPV2V / Carla store orientation as yaw(Z) -> pitch(Y) -> roll(X).
    rot = Rotation.from_euler("zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True)
    return rot.as_matrix()


def cords_to_pose(cords: Iterable[float]) -> np.ndarray:
    """Transform OPV2V ``cords`` arrays into 4x4 homogeneous matrices."""
    cords = list(cords)
    if len(cords) != 6:
        raise ValueError(f"Expected 6 values for pose, received {len(cords)}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _angles_to_rotation(cords[3:])
    pose[:3, 3] = np.asarray(cords[:3], dtype=np.float64)
    return pose


def get_camera_poses_in_ego(frame_meta: Dict) -> Dict[str, np.ndarray]:
    """
    Convert all camera poses from world to ego (LiDAR) coordinates.

    Args:
        frame_meta: Parsed YAML dictionary for a single OPV2V sample.

    Returns:
        Dict mapping camera names to 4x4 poses expressed in the ego frame.
    """
    T_world_ego = cords_to_pose(frame_meta["lidar_pose"])
    T_ego_world = np.linalg.inv(T_world_ego)
    camera_poses = {}
    for cam_key in ("camera0", "camera1", "camera2", "camera3"):
        cam_pose = cords_to_pose(frame_meta[cam_key]["cords"])
        camera_poses[cam_key] = T_ego_world @ cam_pose
    return camera_poses


def get_vehicle_bboxes_in_ego(
    frame_meta: Dict, max_range: float | None = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert all vehicle bounding boxes into the ego coordinate frame.

    Args:
        frame_meta: Parsed YAML for the frame.
        max_range: Optional distance threshold (meters). Bounding boxes with
            centers farther than this range from the ego origin are skipped.

    Returns:
        Dict mapping vehicle ids to dictionaries containing:
            - ``center`` (3,): bbox center in ego coordinates
            - ``corners`` (8, 3): bbox corners in ego coordinates
    """
    T_world_ego = cords_to_pose(frame_meta["lidar_pose"])
    T_ego_world = np.linalg.inv(T_world_ego)
    vehicle_bboxes = {}
    for veh_id, veh_data in frame_meta["vehicles"].items():
        T_world_vehicle = cords_to_pose((*veh_data["location"], *veh_data["angle"]))
        center_local = np.asarray(veh_data["center"], dtype=np.float64)
        extent = np.asarray(veh_data["extent"], dtype=np.float64)

        pose_center = T_world_vehicle.copy()
        pose_center[:3, 3] = (
            T_world_vehicle[:3, 3] + T_world_vehicle[:3, :3] @ center_local
        )
        T_ego_bbox = T_ego_world @ pose_center
        bbox_center = T_ego_bbox[:3, 3]

        if max_range is not None and np.linalg.norm(bbox_center) > max_range:
            continue

        signs = np.array(
            [
                [sx, sy, sz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        corners_local = signs * extent
        corners_world = (pose_center @ np.c_[corners_local, np.ones(8)].T).T[:, :3]
        corners_ego = (T_ego_world @ np.c_[corners_world, np.ones(8)].T).T[:, :3]

        vehicle_bboxes[str(veh_id)] = {
            "center": bbox_center,
            "corners": corners_ego,
        }
    return vehicle_bboxes


def load_frame_metadata(yaml_path: str | Path) -> Dict:
    """Load a single OPV2V frame YAML file."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.UnsafeLoader)


@dataclass
class PCDFrame:
    """Container holding all assets for a single OPV2V sample."""

    points: np.ndarray
    camera_poses: Dict[str, np.ndarray]
    vehicle_bboxes: Dict[str, Dict[str, np.ndarray]]


def load_ascii_pcd_xyz(pcd_path: str | Path) -> np.ndarray:
    """
    Load XYZ coordinates from an ASCII .pcd file.

    Returns:
        numpy array of shape (N, 3)
    """
    pcd_path = Path(pcd_path)
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    return points


def load_opv2v_frame(
    frame_dir: str | Path,
    frame_id: str,
    bbox_range: float = 120.0,
) -> PCDFrame:
    """
    Load a frame directory consisting of ``.pcd`` and ``.yaml`` files.

    Args:
        frame_dir: Directory containing ``<frame_id>.pcd`` and ``.yaml`` files.
        frame_id: Frame identifier (e.g., ``000069``).
        bbox_range: Bounding boxes farther than this radius (meters) are
            discarded to keep the visualization readable.
    """
    frame_dir = Path(frame_dir)
    yaml_path = frame_dir / f"{frame_id}.yaml"
    pcd_path = frame_dir / f"{frame_id}.pcd"

    meta = load_frame_metadata(yaml_path)
    camera_poses = get_camera_poses_in_ego(meta)
    vehicle_bboxes = get_vehicle_bboxes_in_ego(meta, max_range=bbox_range)
    points = load_ascii_pcd_xyz(pcd_path)

    return PCDFrame(points=points, camera_poses=camera_poses, vehicle_bboxes=vehicle_bboxes)
