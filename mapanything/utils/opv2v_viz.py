"""Utilities shared across OPV2V visualization scripts."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from mapanything.utils.image import preprocess_inputs

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")


def _carla_basis(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def ue_c2w_to_opencv_c2w(c2w_ue: torch.Tensor) -> torch.Tensor:
    """Convert a camera pose from Carla/UE coordinates into OpenCV coordinates."""
    if c2w_ue.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {c2w_ue.shape}")
    S = _carla_basis(c2w_ue.device)
    r_cv = S @ c2w_ue[:3, :3] @ S.t()
    t_cv = S @ c2w_ue[:3, 3]
    c2w_cv = torch.eye(4, dtype=torch.float32, device=c2w_ue.device)
    c2w_cv[:3, :3] = r_cv
    c2w_cv[:3, 3] = t_cv
    return c2w_cv


def opencv_c2w_to_ue_c2w(c2w_cv: torch.Tensor) -> torch.Tensor:
    """Invert :func:`ue_c2w_to_opencv_c2w`."""
    if c2w_cv.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {c2w_cv.shape}")
    S = _carla_basis(c2w_cv.device)
    r_ue = S.t() @ c2w_cv[:3, :3] @ S
    t_ue = S.t() @ c2w_cv[:3, 3]
    c2w_ue = torch.eye(4, dtype=torch.float32, device=c2w_cv.device)
    c2w_ue[:3, :3] = r_ue
    c2w_ue[:3, 3] = t_ue
    return c2w_ue


def rescale_intrinsics_to_image(
    intrinsics: torch.Tensor, img_w: int, img_h: int
) -> torch.Tensor:
    """Rescale a pinhole intrinsic matrix to match a particular resolution."""
    intrinsics = intrinsics.clone()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()
    base_w = max(1.0, 2.0 * cx)
    base_h = max(1.0, 2.0 * cy)
    sx = img_w / base_w
    sy = img_h / base_h
    if abs(img_w - base_w) > 1.5 or abs(img_h - base_h) > 1.5:
        intrinsics[0, 0] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[0, 2] *= sx
        intrinsics[1, 2] *= sy
        logger.info(
            "Rescaled intrinsics to (%d x %d) with factors sx=%.4f sy=%.4f",
            img_w,
            img_h,
            sx,
            sy,
        )
    return intrinsics


def pose_list_to_matrix(pose: Sequence[float], device: torch.device) -> torch.Tensor:
    """Convert ``[x, y, z, roll, yaw, pitch]`` (degrees) into a 4x4 SE(3) matrix."""
    if len(pose) < 6:
        raise ValueError(f"Pose must contain 6 values, got {pose}")
    x, y, z, roll_deg, yaw_deg, pitch_deg = pose[:6]
    roll = math.radians(roll_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    c_y = math.cos(yaw)
    s_y = math.sin(yaw)
    c_r = math.cos(roll)
    s_r = math.sin(roll)
    c_p = math.cos(pitch)
    s_p = math.sin(pitch)
    mat = torch.eye(4, dtype=torch.float32, device=device)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    mat[0, 0] = c_y * c_p
    mat[0, 1] = c_y * s_p * s_r - s_y * c_r
    mat[0, 2] = c_y * s_p * c_r + s_y * s_r
    mat[1, 0] = s_y * c_p
    mat[1, 1] = s_y * s_p * s_r + c_y * c_r
    mat[1, 2] = s_y * s_p * c_r - c_y * s_r
    mat[2, 0] = -s_p
    mat[2, 1] = c_p * s_r
    mat[2, 2] = c_p * c_r
    return mat


def load_views_from_config(
    config_path: Path,
    image_dir: Path,
    device: torch.device,
    *,
    camera_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a Carla/OPV2V YAML and build MapAnything-ready views."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)
    if "lidar_pose" not in config_data:
        raise ValueError(f"'lidar_pose' missing from {config_path}")

    ego_pose_ue = pose_list_to_matrix(config_data["lidar_pose"], device)
    ego_pose_cv = ue_c2w_to_opencv_c2w(ego_pose_ue)
    world_to_ego_cv = torch.inverse(ego_pose_cv)

    config_stem = config_path.stem
    camera_keys = sorted(
        [k for k in config_data.keys() if k.startswith("camera")],
        key=lambda name: int(name.replace("camera", "")),
    )
    if camera_filter:
        camera_filter_set = {f"camera{cid}" if isinstance(cid, int) else str(cid) for cid in camera_filter}
        camera_keys = [key for key in camera_keys if key in camera_filter_set]
    if not camera_keys:
        raise ValueError(f"No camera entries found inside {config_path}")

    raw_views: List[Dict[str, Any]] = []
    camera_info_list: List[Dict[str, Any]] = []

    for cam_name in camera_keys:
        cam_cfg = config_data[cam_name]
        img_path: Optional[Path] = None
        basename = f"{config_stem}_{cam_name}"
        for ext in IMAGE_EXTENSIONS:
            candidate = image_dir / f"{basename}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            logger.warning("Image for %s not found under %s.*", basename, image_dir)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to open %s: %s", img_path, exc)
            continue

        W, H = img.size
        if "intrinsic" not in cam_cfg:
            logger.warning("%s missing intrinsics, skipping.", cam_name)
            continue

        intrinsics = torch.tensor(cam_cfg["intrinsic"], dtype=torch.float32, device=device)
        intrinsics = rescale_intrinsics_to_image(intrinsics, W, H)

        if "cords" not in cam_cfg:
            logger.warning("%s missing 'cords' (world pose). Skipping.", cam_name)
            continue
        cam_pose_list = cam_cfg["cords"]
        if len(cam_pose_list) != 6:
            logger.warning(
                "%s has invalid cords length (%d). Skipping.", cam_name, len(cam_pose_list)
            )
            continue

        c2w_ue = pose_list_to_matrix(cam_pose_list, device)
        c2w_cv = ue_c2w_to_opencv_c2w(c2w_ue)
        c2e_cv = world_to_ego_cv @ c2w_cv

        raw_views.append(
            {
                "img": img,
                "intrinsics": intrinsics.cpu().numpy(),
                "camera_poses": c2e_cv.cpu().numpy(),
            }
        )
        camera_info_list.append(
            {
                "name": cam_name,
                "pose_C2W_cv": c2w_cv.cpu(),
                "pose_C2E_cv": c2e_cv.cpu(),
                "image_path": str(img_path),
            }
        )

    if not raw_views:
        raise ValueError(f"No valid camera views could be constructed from {config_path}")

    processed_views = preprocess_inputs(raw_views)
    return processed_views, camera_info_list


def _rotation_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    r_rel = R_gt.T @ R_pred
    trace_val = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(trace_val))


def _compute_pose_error_metrics(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, float]:
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]
    abs_trans = np.linalg.norm(pred_t - gt_t)
    gt_trans_norm = max(np.linalg.norm(gt_t), 1e-6)
    rel_trans = abs_trans / gt_trans_norm
    abs_rot_deg = _rotation_angle_deg(pred_R, gt_R)
    rel_rot = abs_rot_deg / 180.0
    return {
        "abs_trans": float(abs_trans),
        "rel_trans": float(rel_trans),
        "abs_rot_deg": float(abs_rot_deg),
        "rel_rot": float(rel_rot),
    }


def log_camera_pose_errors(
    predictions: List[Dict[str, torch.Tensor]],
    camera_info_list: List[Dict[str, Any]],
    *,
    prefer_world_frame: bool = True,
) -> None:
    pose_key = "pose_C2W_cv" if prefer_world_frame else "pose_C2E_cv"
    if pose_key not in camera_info_list[0]:
        pose_key = "pose_C2E_cv"
    gt_ref_pose = camera_info_list[0][pose_key].cpu().numpy()
    gt_ref_pose_inv = np.linalg.inv(gt_ref_pose)

    pred_ref_pose = None
    if predictions:
        pred_ref_tensor = predictions[0].get("camera_poses")
        if pred_ref_tensor is not None:
            pred_ref_pose = pred_ref_tensor[0].detach().cpu().numpy()
            pred_ref_pose = np.linalg.inv(pred_ref_pose)

    logger.info(
        "%s",
        f"{'View':<12}{'AbsTrans(m)':>12}{'RelTrans(%)':>14}{'AbsRot(deg)':>14}{'RelRot(%)':>12}",
    )
    for idx, cam_info in enumerate(camera_info_list):
        if idx >= len(predictions):
            logger.warning(
                "Skipping pose error calc for %s: missing prediction entry (idx=%d).",
                cam_info.get("name", f"view_{idx}"),
                idx,
            )
            break
        pred = predictions[idx]
        pred_pose_tensor = pred.get("camera_poses")
        if pred_pose_tensor is None:
            logger.warning(
                "Prediction %s missing 'camera_poses'; cannot compute pose error.",
                cam_info.get("name", f"view_{idx}"),
            )
            continue

        pred_pose_np = pred_pose_tensor[0].detach().cpu().numpy()
        gt_pose_np = cam_info[pose_key].cpu().numpy()
        gt_pose_rel = gt_ref_pose_inv @ gt_pose_np
        if pred_ref_pose is not None:
            pred_pose_rel = pred_ref_pose @ pred_pose_np
        else:
            pred_pose_rel = pred_pose_np
        metrics = _compute_pose_error_metrics(pred_pose_rel, gt_pose_rel)
        logger.info(
            f"{cam_info.get('name', f'view_{idx}'):<12}"
            f"{metrics['abs_trans']:>12.4f}"
            f"{metrics['rel_trans'] * 100:>14.2f}"
            f"{metrics['abs_rot_deg']:>14.2f}"
            f"{metrics['rel_rot'] * 100:>12.2f}"
        )
