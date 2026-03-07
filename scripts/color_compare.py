
# scripts/color_compare.py (Fixed)

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local
from mapanything.utils.image import load_images, rgb, preprocess_inputs

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_OVERRIDES = [
    "machine=aws",
    "model=mapanything",
    "model/task=images_only",
    "model.encoder.uses_torch_hub=false",
]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def strip_external_calibration_inputs(
    views: List[Dict[str, Any]],
    *,
    strip_intrinsics: bool = False,
) -> None:
    """Remove calibration-related inputs from preprocessed views in-place.

    This is useful when evaluating the ``images_only`` task: we still keep the
    calibration/pose information separately for metric computation, but we do
    not allow the model to consume ground-truth intrinsics/extrinsics.
    """

    keys = {
        "camera_poses",
        "camera_pose_quats",
        "camera_pose_trans",
        "ray_directions",
        "ray_directions_cam",
    }
    if strip_intrinsics:
        keys.add("intrinsics")

    for view in views:
        for key in keys:
            view.pop(key, None)


def ue_c2w_to_opencv_c2w(C2W_ue: torch.Tensor) -> torch.Tensor:
    """
    Convert a Carla/UE camera-to-frame transform into the OpenCV (right-handed) convention.
    Matches the helper used inside scripts/depth_ge.py.
    """
    if C2W_ue.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {C2W_ue.shape}")

    S = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=C2W_ue.device,
    )

    R_ue = C2W_ue[:3, :3]
    t_ue = C2W_ue[:3, 3]

    R_cv = S @ R_ue @ S.t()
    t_cv = S @ t_ue

    C2W_cv = torch.eye(4, dtype=torch.float32, device=C2W_ue.device)
    C2W_cv[:3, :3] = R_cv
    C2W_cv[:3, 3] = t_cv
    return C2W_cv


def rescale_intrinsics_to_image(K: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Rescale the pinhole intrinsics if the current image resolution differs from the calibration resolution.
    """
    K = K.clone()
    cx = K[0, 2].item()
    cy = K[1, 2].item()
    W0 = max(1.0, 2.0 * cx)
    H0 = max(1.0, 2.0 * cy)
    sx = img_w / W0
    sy = img_h / H0

    if abs(img_w - W0) > 1.5 or abs(img_h - H0) > 1.5:
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        logger.info(
            "Rescaled intrinsics to match resolution (%d x %d) with factors sx=%.4f sy=%.4f",
            img_w,
            img_h,
            sx,
            sy,
        )
    else:
        logger.info("Intrinsics already match current resolution; no rescale applied.")

    return K


def pose_list_to_matrix(pose: List[float], device: torch.device) -> torch.Tensor:
    """
    Convert [x, y, z, roll, yaw, pitch] (degrees) to a 4x4 SE(3) matrix following the Carla convention (Z-Y-X order).
    """
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
    config_path: Path, image_dir: Path, device: torch.device
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse a Carla/OPV2V-style YAML to build MapAnything-ready views plus per-camera pose info.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

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
        except Exception as exc:
            logger.warning("Failed to open %s: %s", img_path, exc)
            continue

        W, H = img.size

        if "intrinsic" not in cam_cfg:
            logger.warning("%s missing intrinsics, skipping.", cam_name)
            continue

        intrinsics = torch.tensor(
            cam_cfg["intrinsic"], dtype=torch.float32, device=device
        )
        intrinsics = rescale_intrinsics_to_image(intrinsics, W, H)

        if "cords" not in cam_cfg:
            logger.warning("%s missing 'cords' (world pose). Skipping.", cam_name)
            continue

        cam_pose_list = cam_cfg["cords"]
        if len(cam_pose_list) != 6:
            logger.warning("%s has invalid cords length (%d). Skipping.", cam_name, len(cam_pose_list))
            continue

        C2W_ue = pose_list_to_matrix(cam_pose_list, device)
        C2W_cv = ue_c2w_to_opencv_c2w(C2W_ue)
        C2E_cv = world_to_ego_cv @ C2W_cv

        raw_views.append(
            {
                "img": img,
                "intrinsics": intrinsics.cpu().numpy(),
                "camera_poses": C2E_cv.cpu().numpy(),
            }
        )
        camera_info_list.append(
            {
                "name": cam_name,
                "pose_C2W_cv": C2W_cv.cpu(),
                "pose_C2E_cv": C2E_cv.cpu(),
                "image_path": str(img_path),
            }
        )

    if not raw_views:
        raise ValueError(f"No valid camera views could be constructed from {config_path}")

    processed_views = preprocess_inputs(raw_views)
    return processed_views, camera_info_list


def _rotation_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute the geodesic distance (in degrees) between two rotation matrices.
    """
    # Relative rotation from gt -> pred
    R_rel = R_gt.T @ R_pred
    trace_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(trace_val))


def _compute_pose_error_metrics(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, float]:
    """
    Compute absolute/relative translation (meters) and rotation (degrees) errors.
    """
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]

    abs_trans = np.linalg.norm(pred_t - gt_t)
    gt_trans_norm = max(np.linalg.norm(gt_t), 1e-6)
    rel_trans = abs_trans / gt_trans_norm

    abs_rot_deg = _rotation_angle_deg(pred_R, gt_R)
    rel_rot = abs_rot_deg / 180.0  # relative to the maximum possible geodesic distance

    return {
        "abs_trans": float(abs_trans),
        "rel_trans": float(rel_trans),
        "abs_rot_deg": float(abs_rot_deg),
        "rel_rot": float(rel_rot),
    }


def log_camera_pose_errors(
    predictions: List[Dict[str, torch.Tensor]], camera_info_list: List[Dict[str, Any]]
) -> None:
    """
    Compare predicted camera poses against YAML-derived poses (both expressed
    relative to the first view to remove arbitrary global offsets) and log
    error statistics.
    """
    rows: List[Dict[str, float]] = []
    header = f"{'View':<12}{'AbsTrans(m)':>12}{'RelTrans(%)':>14}{'AbsRot(deg)':>14}{'RelRot(%)':>12}"
    logger.info("Camera pose errors (predicted vs YAML, relative to view_0):")
    logger.info(header)

    pose_key = "pose_C2W_cv" if "pose_C2W_cv" in camera_info_list[0] else "pose_C2E_cv"
    gt_ref_pose = camera_info_list[0][pose_key].cpu().numpy()
    gt_ref_pose_inv = np.linalg.inv(gt_ref_pose)

    pred_ref_pose = None
    if predictions:
        pred_ref_tensor = predictions[0].get("camera_poses")
        if pred_ref_tensor is not None:
            pred_ref_pose = pred_ref_tensor[0].detach().cpu().numpy()
            pred_ref_pose = np.linalg.inv(pred_ref_pose)

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
        rows.append(metrics)

        logger.info(
            f"{cam_info.get('name', f'view_{idx}'):<12}"
            f"{metrics['abs_trans']:>12.4f}"
            f"{metrics['rel_trans'] * 100:>14.2f}"
            f"{metrics['abs_rot_deg']:>14.3f}"
            f"{metrics['rel_rot'] * 100:>12.2f}"
        )

    if rows:
        avg_abs_trans = float(np.mean([r["abs_trans"] for r in rows]))
        avg_rel_trans = float(np.mean([r["rel_trans"] for r in rows]))
        avg_abs_rot = float(np.mean([r["abs_rot_deg"] for r in rows]))
        avg_rel_rot = float(np.mean([r["rel_rot"] for r in rows]))
        logger.info("-" * len(header))
        logger.info(
            f"{'Average':<12}"
            f"{avg_abs_trans:>12.4f}"
            f"{avg_rel_trans * 100:>14.2f}"
            f"{avg_abs_rot:>14.3f}"
            f"{avg_rel_rot * 100:>12.2f}"
        )


def read_pcd_with_packed_rgb(file_path):
    """
    Reads a .pcd file with RGB data packed into a single float.
    This logic is adapted from visualize.py to handle the specific format.
    """
    # This assumes an ASCII PCD file with a standard 11-line header.
    try:
        with open(file_path, 'r') as f:
            header = [next(f) for _ in range(11)]
        data = np.loadtxt(file_path, skiprows=11)
    except Exception as e:
        logger.error(f"Failed to load .pcd file with numpy from {file_path}: {e}")
        logger.warning("Falling back to standard Open3D reader.")
        return o3d.io.read_point_cloud(str(file_path))

    points = data[:, :3]
    
    # Check if color information is present
    if data.shape[1] < 4:
        logger.warning(f"No color data in {file_path}. Returning colorless point cloud.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    # Extract and convert packed RGB float to three-channel color
    rgb_float = data[:, 3].copy()
    rgb_int_view = rgb_float.view(np.int32)
    r = (rgb_int_view >> 16) & 0xFF
    g = (rgb_int_view >> 8) & 0xFF
    b = rgb_int_view & 0xFF
    colors = np.vstack([r, g, b]).T / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    """
    Main function to run model inference and visualize point clouds.
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Error: open3d library is not installed. Please run 'pip install open3d' to install it.")
        return

    parser = argparse.ArgumentParser(description="Compare predicted point cloud with ground truth from a .pcd file using MapAnything.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument("--config_path", type=str, default=None, help="Optional YAML config describing calibrated camera poses and intrinsics.")
    parser.add_argument("--gt_pcd_path", type=str, required=True, help="Path to the ground truth point cloud file (.pcd format).")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient mode for inference (slower).")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint-best.pth", help="Local MapAnything checkpoint to load (.pth or .safetensors).")
    parser.add_argument("--hydra_config_path", type=str, default="configs/train.yaml", help="Hydra config used to instantiate the model.")
    parser.add_argument("--config_json_path", type=str, default="scripts/local_models/config.json", help="Optional model config JSON describing encoder/heads.")
    parser.add_argument(
        "--config_overrides",
        nargs="*",
        default=None,
        help="Optional Hydra overrides (defaults target the released MapAnything model).",
    )
    parser.add_argument("--strict_load", action="store_true", help="Enable strict checkpoint loading.")
    parser.add_argument("--no_viz", action="store_true", help="Disable Open3D visualization (useful for headless runs).")
    parser.add_argument("--save_pred_path", type=str, default=None, help="Optional path to save aggregated predicted point cloud (.pcd/.ply).")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_pcd_path = Path(args.gt_pcd_path)
    checkpoint_path = Path(args.checkpoint_path)
    hydra_config_path = Path(args.hydra_config_path)
    config_json_path = Path(args.config_json_path)
    config_path = Path(args.config_path) if args.config_path else None

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not gt_pcd_path.is_file():
        raise FileNotFoundError(f"Ground truth .pcd file not found: {gt_pcd_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not hydra_config_path.is_file():
        raise FileNotFoundError(f"Hydra config not found: {hydra_config_path}")
    if not config_json_path.is_file():
        raise FileNotFoundError(f"Config JSON not found: {config_json_path}")

    if config_path and not config_path.is_file():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    # --- Model Loading and Inference ---
    logger.info("Loading MapAnything model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.config_overrides:
        config_overrides = args.config_overrides
    else:
        config_overrides = DEFAULT_CONFIG_OVERRIDES

    local_config = {
        "path": str(hydra_config_path),
        "checkpoint_path": str(checkpoint_path),
        "config_overrides": config_overrides,
        "config_json_path": str(config_json_path),
        "strict": args.strict_load,
    }

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model = initialize_mapanything_local(local_config, device)
    model.eval()

    if config_path:
        logger.info("Loading calibrated multi-view inputs from %s", config_path)
        views, camera_info_list = load_views_from_config(config_path, image_dir, device)
        logger.info("Loaded %d calibrated views using external poses.", len(views))
    else:
        camera_info_list = None
        logger.info(f"Loading images from '{image_dir}'...")
        views = load_images(str(image_dir))
    using_external_poses = camera_info_list is not None and len(camera_info_list) > 0

    logger.info(f"Loaded {len(views)} images, starting inference (image-only mode)...")
    with torch.no_grad():
        predictions = model.infer(views, memory_efficient_inference=args.memory_efficient)
    logger.info("Inference complete.")

    if using_external_poses:
        log_camera_pose_errors(predictions, camera_info_list)

    # --- Extract Prediction Results (aggregate point cloud) ---
    aggregated_points: List[np.ndarray] = []
    aggregated_colors: List[np.ndarray] = []

    T_CV_TO_UE = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    R_CV_TO_UE = T_CV_TO_UE[:3, :3]

    if using_external_poses:
        if len(camera_info_list) != len(predictions):
            logger.warning(
                "Camera info count (%d) does not match predictions (%d).",
                len(camera_info_list),
                len(predictions),
            )

        for idx, pred in enumerate(predictions):
            if idx >= len(camera_info_list):
                logger.warning("No camera pose info for prediction %d. Stopping.", idx)
                break

            cam_info = camera_info_list[idx]
            view_label = cam_info.get("name", f"view_{idx}")

            if "pts3d_cam" not in pred:
                logger.warning("Prediction %s missing 'pts3d_cam'; skipping.", view_label)
                continue

            pts_cam = pred["pts3d_cam"][0].cpu().numpy()
            mask_tensor = pred.get("mask")
            if mask_tensor is None:
                logger.warning("Prediction %s missing mask; skipping.", view_label)
                continue
            mask_np = mask_tensor[0].cpu().numpy().astype(bool).squeeze(-1)

            pts_flat = pts_cam.reshape(-1, 3)
            mask_flat = mask_np.reshape(-1)
            if mask_flat.shape[0] != pts_flat.shape[0]:
                logger.warning(
                    "Mask/point mismatch for %s (mask=%d, pts=%d).",
                    view_label,
                    mask_flat.shape[0],
                    pts_flat.shape[0],
                )
                continue

            pts_masked = pts_flat[mask_flat]
            if pts_masked.size == 0:
                logger.warning("No valid points for %s after masking.", view_label)
                continue

            if "img_no_norm" in pred:
                img_np = pred["img_no_norm"][0].cpu().numpy()
            else:
                view_tensor = views[idx]["img"]
                norm_type = views[idx]["data_norm_type"][0]
                img_np = rgb(view_tensor, norm_type=norm_type)[0]

            colors_flat = img_np.reshape(-1, 3)
            colors_masked = colors_flat[mask_flat]

            pts_cam_hom = np.hstack(
                (pts_masked, np.ones((pts_masked.shape[0], 1), dtype=pts_masked.dtype))
            )
            pose_c2e_cv = cam_info["pose_C2E_cv"].numpy()
            pts_ego_cv = (pose_c2e_cv @ pts_cam_hom.T).T[:, :3]

            if colors_masked.shape[0] == pts_ego_cv.shape[0]:
                if colors_masked.max() > 1.0:
                    colors_masked = colors_masked / 255.0
                logger.info(
                    "Applied colors to fused ego point chunk for %s (%d points).",
                    view_label,
                    pts_ego_cv.shape[0],
                )
            else:
                logger.warning(
                    "Color mismatch for %s (points=%d, colors=%d). Using blue.",
                    view_label,
                    pts_ego_cv.shape[0],
                    colors_masked.shape[0],
                )
                colors_masked = np.tile(np.array([[0.0, 0.0, 1.0]]), (pts_ego_cv.shape[0], 1))

            pts_ego_cv_hom = np.hstack(
                (pts_ego_cv, np.ones((pts_ego_cv.shape[0], 1), dtype=pts_ego_cv.dtype))
            )
            pts_ego_ue = (T_CV_TO_UE @ pts_ego_cv_hom.T).T[:, :3]
            aggregated_points.append(pts_ego_ue)
            aggregated_colors.append(colors_masked)
    else:
        for i, pred in enumerate(predictions):
            pts3d_world = pred["pts3d"][0].cpu().numpy()  # shape (H, W, 3)
            points_flat = pts3d_world.reshape(-1, 3)

            mask = pred.get("mask")
            if mask is not None:
                mask_np = mask[0].cpu().numpy().astype(bool).squeeze(-1)
                mask_flat = mask_np.reshape(-1)
                if mask_flat.shape[0] != points_flat.shape[0]:
                    logger.warning(
                        "Mask size mismatch for view %d (mask=%d, points=%d). Ignoring mask.",
                        i,
                        mask_flat.shape[0],
                        points_flat.shape[0],
                    )
                    mask_flat = None
            else:
                mask_flat = None

            points_filtered = (
                points_flat if mask_flat is None else points_flat[mask_flat]
            )
            if points_filtered.size == 0:
                logger.warning("No valid points for view %d. Skipping point cloud.", i)
                continue

            if "img_no_norm" in pred:
                img_np = pred["img_no_norm"][0].cpu().numpy()
            else:
                view_tensor = views[i]["img"]
                norm_type = views[i]["data_norm_type"][0]
                img_np = rgb(view_tensor, norm_type=norm_type)[0]

            colors_flat = img_np.reshape(-1, 3)
            colors_filtered = (
                colors_flat if mask_flat is None else colors_flat[mask_flat]
            )

            if colors_filtered.shape[0] == points_filtered.shape[0]:
                if colors_filtered.max() > 1.0:
                    colors_filtered = colors_filtered / 255.0
                logger.info(
                    "Applied colors to predicted point cloud for view %d (%d points).",
                    i,
                    points_filtered.shape[0],
                )
            else:
                logger.warning(
                    "Color/point mismatch for view %d (points=%d, colors=%d). Using uniform blue.",
                    i,
                    points_filtered.shape[0],
                    colors_filtered.shape[0],
                )
                colors_filtered = np.tile(
                    np.array([[0.0, 0.0, 1.0]]), (points_filtered.shape[0], 1)
                )

            # Rotate OpenCV-style world predictions into the UE/ego convention
            points_filtered_ue = points_filtered @ R_CV_TO_UE.T
            aggregated_points.append(points_filtered_ue)
            aggregated_colors.append(colors_filtered)

    if not aggregated_points:
        logger.error("No predicted point data were generated. Exiting.")
        return

    points_pred = np.concatenate(aggregated_points, axis=0)
    colors_pred = np.concatenate(aggregated_colors, axis=0)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    if colors_pred.shape[0] == points_pred.shape[0]:
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
    else:
        logger.warning(
            "Aggregated color count (%d) != point count (%d). Using uniform blue.",
            colors_pred.shape[0],
            points_pred.shape[0],
        )
        pcd_pred.paint_uniform_color([0, 0, 1])

    # --- Load Ground Truth Point Cloud (.pcd) ---
    logger.info(f"Loading ground truth point cloud from '{gt_pcd_path}'...")
    if args.save_pred_path:
        save_path = Path(args.save_pred_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(
            str(save_path),
            pcd_pred,
            write_ascii=True,
            compressed=False,
            print_progress=True,
        )
        if ok:
            logger.info("Predicted point cloud saved to '%s'.", save_path.resolve())
        else:
            logger.error("Failed to write predicted point cloud to '%s'.", save_path)

    pcd_gt = read_pcd_with_packed_rgb(str(gt_pcd_path))
    if pcd_gt is None or not pcd_gt.has_points():
        logger.error("Failed to load ground truth point cloud or it is empty.")
        return
    
    if not pcd_gt.has_colors():
        logger.warning("Ground truth point cloud does not have colors. Painting it green.")
        pcd_gt.paint_uniform_color([0, 1, 0]) # Green

    logger.info(f"Successfully loaded ground truth point cloud with {len(pcd_gt.points)} points.")

    if args.no_viz:
        logger.info("Skipping visualization as requested (--no_viz).")
        return

    # --- Visualization ---
    logger.info("Creating visualization windows (predicted vs ground truth)...")

    vis_pred = o3d.visualization.Visualizer()
    vis_pred.create_window(
        window_name="Predicted Point Cloud (Aggregated)",
        width=960,
        height=1080,
        left=0,
        top=40,
    )
    vis_pred.add_geometry(pcd_pred)

    vis_gt = o3d.visualization.Visualizer()
    vis_gt.create_window(
        window_name="Ground Truth Point Cloud",
        width=960,
        height=1080,
        left=960,
        top=40,
    )
    vis_gt.add_geometry(pcd_gt)

    logger.info("Windows created. Close both windows to exit the program.")

    try:
        while True:
            if not vis_pred.poll_events() or not vis_gt.poll_events():
                break
            vis_pred.update_renderer()
            vis_gt.update_renderer()
    finally:
        vis_pred.destroy_window()
        vis_gt.destroy_window()
        logger.info("Visualization windows closed.")


if __name__ == "__main__":
    main()
