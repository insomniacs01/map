
# scripts/color_compare.py (Fixed)

import argparse
import logging
import math
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise ImportError(
        "scipy is required for pose conversions. Please install it via 'pip install scipy'."
    ) from exc

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
YAML_EXTENSIONS = (".yaml", ".yml")
CARLA_TO_CAMERA_CV = torch.tensor(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)
CV_TO_UE_TRANSFORM = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
CV_TO_UE_ROTATION = CV_TO_UE_TRANSFORM[:3, :3]


class PoseLogSink:
    """Simple sink that mirrors pose-related logs into a file."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._fh: Optional[TextIO] = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def log_lines(self, lines: Iterable[str]) -> None:
        if not self._fh:
            return
        for line in lines:
            self._fh.write(f"{line}\n")
        self._fh.flush()

    def log_run_header(self, command_line: str) -> None:
        if not self._fh:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        separator = "=" * 80
        self.log_lines(
            [
                "",
                separator,
                f"Run started: {timestamp}",
                f"Command: {command_line}",
                separator,
            ]
        )

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def _maybe_log_lines(pose_log_sink: Optional[PoseLogSink], lines: Iterable[str]) -> None:
    if pose_log_sink is not None:
        pose_log_sink.log_lines(lines)


CALIBRATION_KEYS_TO_DROP = {
    "intrinsics",
    "camera_poses",
    "camera_pose_quats",
    "camera_pose_trans",
    "ray_directions",
    "ray_directions_cam",
}


def strip_external_calibration_inputs(views: List[Dict[str, Any]]) -> None:
    """
    Remove externally provided calibration (poses/intrinsics/ray dirs) so inference stays image-only.
    Operates in-place on the provided list of view dictionaries.
    """
    for view in views:
        for key in CALIBRATION_KEYS_TO_DROP:
            if key in view:
                view.pop(key)


def ue_c2w_to_opencv_c2w(C2W_ue: torch.Tensor) -> torch.Tensor:
    """
    Convert a Carla/UE camera-to-frame transform into the OpenCV (right-handed) convention
    using the CARLA→OpenCV change-of-basis matrix.
    """
    if C2W_ue.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {C2W_ue.shape}")

    converter = CARLA_TO_CAMERA_CV.to(C2W_ue.device)
    converter_inv = converter.t()
    return converter @ C2W_ue @ converter_inv


def _angles_to_rotation(angles_deg: Iterable[float]) -> np.ndarray:
    """Convert [roll, yaw, pitch] angles (degrees) into a Carla-style rotation matrix."""
    angles_deg = list(angles_deg)
    if len(angles_deg) != 3:
        raise ValueError(f"Expected 3 angles, got {len(angles_deg)}")

    roll_deg, yaw_deg, pitch_deg = angles_deg
    # Carla stores rotations as yaw(Z) → pitch(Y) → roll(X). The equivalent SciPy call
    # is intrinsic order "zyx" with angles [yaw, pitch, roll].
    rot = Rotation.from_euler("zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True)
    return rot.as_matrix().astype(np.float32)


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


def cords_to_pose_matrix(cords: Iterable[float], device: torch.device) -> torch.Tensor:
    """
    Transform OPV2V ``cords`` arrays ([x, y, z, roll, yaw, pitch]) into 4x4 homogeneous matrices.
    """
    values = list(cords)
    if len(values) != 6:
        raise ValueError(f"Expected 6 values for pose, received {len(values)}")

    pose_np = np.eye(4, dtype=np.float32)
    pose_np[:3, :3] = _angles_to_rotation(values[3:])
    pose_np[:3, 3] = np.asarray(values[:3], dtype=np.float32)
    return torch.from_numpy(pose_np).to(device)


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

    ego_pose_ue = cords_to_pose_matrix(config_data["lidar_pose"], device)
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
        view_label = f"{config_stem}_{cam_name}"
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

        C2W_ue = cords_to_pose_matrix(cam_pose_list, device)
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
                "name": view_label,
                "pose_C2W_cv": C2W_cv.cpu(),
                "pose_C2E_cv": C2E_cv.cpu(),
                "image_path": str(img_path),
                "config_name": config_stem,
                "config_path": str(config_path),
                "ego_pose_C2W_cv": ego_pose_cv.cpu(),
            }
        )

    if not raw_views:
        raise ValueError(f"No valid camera views could be constructed from {config_path}")

    processed_views = preprocess_inputs(raw_views)
    return processed_views, camera_info_list


def collect_config_paths(config_path: Path) -> List[Path]:
    """
    Expand a file or directory argument into a sorted list of YAML config paths.
    """
    if config_path.is_file():
        if config_path.suffix.lower() not in YAML_EXTENSIONS:
            raise ValueError(f"Unsupported config extension for {config_path}")
        return [config_path]

    if config_path.is_dir():
        yaml_files = sorted(
            [
                path
                for path in config_path.rglob("*")
                if path.is_file() and path.suffix.lower() in YAML_EXTENSIONS
            ]
        )
        if not yaml_files:
            raise FileNotFoundError(
                f"No YAML config files found inside directory: {config_path}"
            )
        return yaml_files

    raise FileNotFoundError(f"Config path not found: {config_path}")


def load_views_from_configs(
    config_paths: Iterable[Path], image_dir: Path, device: torch.device
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load and concatenate views/camera metadata from multiple YAML config files.
    """
    all_views: List[Dict[str, Any]] = []
    all_camera_info: List[Dict[str, Any]] = []

    for cfg_path in config_paths:
        logger.info("Loading calibrated multi-view inputs from %s", cfg_path)
        views, camera_info = load_views_from_config(cfg_path, image_dir, device)
        all_views.extend(views)
        all_camera_info.extend(camera_info)

    if not all_views:
        raise ValueError("No valid views could be loaded from the provided configs.")

    return all_views, all_camera_info


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


def _format_pose_matrix(matrix: np.ndarray, precision: int = 4) -> str:
    """Pretty-print a pose matrix with consistent precision."""
    formatter = {"float_kind": lambda val, p=precision: f"{val: .{p}f}"}
    return np.array2string(matrix, formatter=formatter)


def _log_pose_matrices(
    view_label: str,
    frame_label: str,
    pred_abs: np.ndarray,
    gt_abs: np.ndarray,
    pred_rel: np.ndarray,
    gt_rel: np.ndarray,
    gt_ref_label: str,
    pred_rel_desc: str,
    pose_log_sink: Optional[PoseLogSink] = None,
) -> None:
    """Log absolute and relative pose matrices for easier inspection."""
    logger.info("Pose matrices for %s (%s frame, OpenCV convention):", view_label, frame_label)
    logger.info("  Predicted (absolute):\n%s", _format_pose_matrix(pred_abs))
    logger.info("  Ground truth (absolute):\n%s", _format_pose_matrix(gt_abs))
    logger.info("  Predicted (%s):\n%s", pred_rel_desc, _format_pose_matrix(pred_rel))
    logger.info(
        "  Ground truth (relative to %s):\n%s",
        gt_ref_label,
        _format_pose_matrix(gt_rel),
    )
    _maybe_log_lines(
        pose_log_sink,
        [
            f"Pose matrices for {view_label} ({frame_label} frame, OpenCV convention):",
            f"  Predicted (absolute):\n{_format_pose_matrix(pred_abs)}",
            f"  Ground truth (absolute):\n{_format_pose_matrix(gt_abs)}",
            f"  Predicted ({pred_rel_desc}):\n{_format_pose_matrix(pred_rel)}",
            f"  Ground truth (relative to {gt_ref_label}):\n{_format_pose_matrix(gt_rel)}",
            "",
        ],
    )


def log_camera_pose_errors(
    predictions: List[Dict[str, torch.Tensor]],
    camera_info_list: List[Dict[str, Any]],
    pose_log_sink: Optional[PoseLogSink] = None,
) -> None:
    """
    Compare predicted camera poses against YAML-derived poses (both expressed
    relative to the first view to remove arbitrary global offsets) and log
    error statistics.
    """
    rows: List[Dict[str, float]] = []
    header = f"{'View':<12}{'AbsTrans(m)':>12}{'RelTrans(%)':>14}{'AbsRot(deg)':>14}{'RelRot(%)':>12}"
    banner = "Camera pose errors (predicted vs YAML, relative to view_0):"
    logger.info(banner)
    logger.info(header)
    _maybe_log_lines(pose_log_sink, [banner, header])

    pose_key = "pose_C2W_cv" if "pose_C2W_cv" in camera_info_list[0] else "pose_C2E_cv"
    pose_frame_label = "C2W" if pose_key == "pose_C2W_cv" else "C2E"
    ref_view_label = camera_info_list[0].get("name", "view_0")
    gt_ref_pose = camera_info_list[0][pose_key].cpu().numpy()
    gt_ref_pose_inv = np.linalg.inv(gt_ref_pose)

    pred_ref_pose = None
    if predictions:
        pred_ref_tensor = predictions[0].get("camera_poses")
        if pred_ref_tensor is not None:
            pred_ref_pose = pred_ref_tensor[0].detach().cpu().numpy()
            pred_ref_pose = np.linalg.inv(pred_ref_pose)
    pred_rel_desc = (
        f"relative to {ref_view_label}" if pred_ref_pose is not None else "model output frame"
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
        rows.append(metrics)

        row = (
            f"{cam_info.get('name', f'view_{idx}'):<12}"
            f"{metrics['abs_trans']:>12.4f}"
            f"{metrics['rel_trans'] * 100:>14.2f}"
            f"{metrics['abs_rot_deg']:>14.3f}"
            f"{metrics['rel_rot'] * 100:>12.2f}"
        )
        logger.info(row)
        _maybe_log_lines(pose_log_sink, [row])

        _log_pose_matrices(
            cam_info.get("name", f"view_{idx}"),
            pose_frame_label,
            pred_pose_np,
            gt_pose_np,
            pred_pose_rel,
            gt_pose_rel,
            ref_view_label,
            pred_rel_desc,
            pose_log_sink,
        )

    if rows:
        avg_abs_trans = float(np.mean([r["abs_trans"] for r in rows]))
        avg_rel_trans = float(np.mean([r["rel_trans"] for r in rows]))
        avg_abs_rot = float(np.mean([r["abs_rot_deg"] for r in rows]))
        avg_rel_rot = float(np.mean([r["rel_rot"] for r in rows]))
        separator = "-" * len(header)
        summary = (
            f"{'Average':<12}"
            f"{avg_abs_trans:>12.4f}"
            f"{avg_rel_trans * 100:>14.2f}"
            f"{avg_abs_rot:>14.3f}"
            f"{avg_rel_rot * 100:>12.2f}"
        )
        logger.info(separator)
        logger.info(summary)
        _maybe_log_lines(pose_log_sink, [separator, summary, ""])


def _log_predicted_pose_only(
    view_label: str,
    pose_abs: np.ndarray,
    pose_rel: np.ndarray,
    rel_ref_label: Optional[str],
    pose_log_sink: Optional[PoseLogSink] = None,
) -> None:
    logger.info("Pose matrices for %s (model output, OpenCV convention):", view_label)
    logger.info("  Predicted (absolute):\n%s", _format_pose_matrix(pose_abs))
    rel_desc = f"relative to {rel_ref_label}" if rel_ref_label else "model output frame"
    logger.info("  Predicted (%s):\n%s", rel_desc, _format_pose_matrix(pose_rel))
    _maybe_log_lines(
        pose_log_sink,
        [
            f"Pose matrices for {view_label} (model output, OpenCV convention):",
            f"  Predicted (absolute):\n{_format_pose_matrix(pose_abs)}",
            f"  Predicted ({rel_desc}):\n{_format_pose_matrix(pose_rel)}",
            "",
        ],
    )


def log_predicted_camera_poses(
    predictions: List[Dict[str, torch.Tensor]],
    pose_log_sink: Optional[PoseLogSink] = None,
) -> None:
    banner = "Predicted camera pose matrices from model outputs (no external config provided):"
    logger.info(banner)
    _maybe_log_lines(pose_log_sink, [banner])

    if not predictions:
        warning_msg = "No predictions were returned; unable to log camera poses."
        logger.warning(warning_msg)
        _maybe_log_lines(pose_log_sink, [warning_msg, ""])
        return

    pred_ref_pose = None
    if predictions[0].get("camera_poses") is not None:
        pred_ref_pose = predictions[0]["camera_poses"][0].detach().cpu().numpy()
        try:
            pred_ref_pose = np.linalg.inv(pred_ref_pose)
        except np.linalg.LinAlgError as exc:
            warning_msg = f"Failed to invert reference pose for relative logging: {exc}."
            logger.warning(warning_msg)
            _maybe_log_lines(pose_log_sink, [warning_msg])
            pred_ref_pose = None
    else:
        warning_msg = (
            "Prediction view_0 missing 'camera_poses'; relative predicted poses will "
            "be reported in the model output frame."
        )
        logger.warning(warning_msg)
        _maybe_log_lines(pose_log_sink, [warning_msg])

    for idx, pred in enumerate(predictions):
        pose_tensor = pred.get("camera_poses")
        view_label = pred.get("view_name", f"view_{idx}")
        if pose_tensor is None:
            msg = f"Prediction {view_label} missing 'camera_poses'; skipping."
            logger.warning(msg)
            _maybe_log_lines(pose_log_sink, [msg])
            continue

        pose_abs = pose_tensor[0].detach().cpu().numpy()
        if pred_ref_pose is not None:
            pose_rel = pred_ref_pose @ pose_abs
            rel_label = predictions[0].get("view_name", "view_0")
        else:
            pose_rel = pose_abs
            rel_label = None

        _log_predicted_pose_only(view_label, pose_abs, pose_rel, rel_label, pose_log_sink)


def _run_color_compare(
    args: argparse.Namespace,
    image_dir: Path,
    gt_pcd_path: Path,
    checkpoint_path: Path,
    hydra_config_path: Path,
    config_json_path: Path,
    config_input_path: Optional[Path],
    pose_log_sink: PoseLogSink,
) -> None:
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

    config_paths: List[Path] = []
    if config_input_path is not None:
        config_paths = collect_config_paths(config_input_path)

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

    if config_paths:
        logger.info("Loading calibrated multi-view inputs from %d config(s).", len(config_paths))
        views, camera_info_list = load_views_from_configs(config_paths, image_dir, device)
        logger.info(
            "Loaded %d calibrated views using %d external config(s).",
            len(views),
            len(config_paths),
        )
        strip_external_calibration_inputs(views)
        logger.info(
            "Stripped external intrinsics/pose inputs from views to keep inference strictly image-only."
        )
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
        log_camera_pose_errors(predictions, camera_info_list, pose_log_sink)
    else:
        log_predicted_camera_poses(predictions, pose_log_sink)

    # --- Extract Prediction Results (aggregate point cloud) ---
    aggregated_points: List[np.ndarray] = []
    aggregated_colors: List[np.ndarray] = []

    T_CV_TO_UE = CV_TO_UE_TRANSFORM
    R_CV_TO_UE = CV_TO_UE_ROTATION

    use_world_frame = False
    if config_paths:
        if args.aggregate_frame == "world":
            use_world_frame = True
        elif args.aggregate_frame == "ego":
            use_world_frame = False
        else:
            use_world_frame = len(config_paths) > 1

    ref_world_to_ego_cv: Optional[np.ndarray] = None
    ref_config_label: Optional[str] = None
    if use_world_frame and camera_info_list:
        for info in camera_info_list:
            ego_pose_tensor = info.get("ego_pose_C2W_cv")
            if ego_pose_tensor is None:
                continue
            ego_pose_np = ego_pose_tensor.detach().cpu().numpy()
            if ego_pose_np.shape != (4, 4):
                continue
            ref_world_to_ego_cv = np.linalg.inv(ego_pose_np)
            ref_config_label = info.get("config_name") or Path(info.get("config_path", "")).stem
            break
        if ref_world_to_ego_cv is not None:
            logger.info(
                "Anchoring fused world points to ego frame from config '%s'.",
                ref_config_label or "unknown",
            )
        else:
            logger.warning("Failed to find ego pose for anchoring. Keeping global world coordinates.")

    if using_external_poses:
        frame_desc = "world" if use_world_frame else "ego"
        logger.info("Aggregating predicted points in %s frame.", frame_desc)

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

            pts_cam_hom = np.hstack((pts_masked, np.ones((pts_masked.shape[0], 1), dtype=pts_masked.dtype)))
            pose_matrix = (
                cam_info["pose_C2W_cv"].numpy() if use_world_frame else cam_info["pose_C2E_cv"].numpy()
            )
            pts_target_cv = (pose_matrix @ pts_cam_hom.T).T[:, :3]
            if use_world_frame and ref_world_to_ego_cv is not None:
                pts_target_cv_hom = np.hstack(
                    (pts_target_cv, np.ones((pts_target_cv.shape[0], 1), dtype=pts_target_cv.dtype))
                )
                pts_target_cv = (ref_world_to_ego_cv @ pts_target_cv_hom.T).T[:, :3]

            if colors_masked.shape[0] == pts_target_cv.shape[0]:
                if colors_masked.max() > 1.0:
                    colors_masked = colors_masked / 255.0
                if use_world_frame:
                    if ref_world_to_ego_cv is not None:
                        chunk_label = f"ref-ego({ref_config_label})" if ref_config_label else "ref-ego"
                    else:
                        chunk_label = "global"
                else:
                    chunk_label = "ego"
                logger.info(
                    "Applied colors to fused %s point chunk for %s (%d points).",
                    chunk_label,
                    view_label,
                    pts_target_cv.shape[0],
                )
            else:
                logger.warning(
                    "Color mismatch for %s (points=%d, colors=%d). Using blue.",
                    view_label,
                    pts_target_cv.shape[0],
                    colors_masked.shape[0],
                )
                colors_masked = np.tile(np.array([[0.0, 0.0, 1.0]]), (pts_target_cv.shape[0], 1))

            pts_target_cv_hom = np.hstack(
                (pts_target_cv, np.ones((pts_target_cv.shape[0], 1), dtype=pts_target_cv.dtype))
            )
            pts_target_ue = (T_CV_TO_UE @ pts_target_cv_hom.T).T[:, :3]
            aggregated_points.append(pts_target_ue)
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

            points_filtered = points_flat if mask_flat is None else points_flat[mask_flat]
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
            colors_filtered = colors_flat if mask_flat is None else colors_flat[mask_flat]

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
                colors_filtered = np.tile(np.array([[0.0, 0.0, 1.0]]), (points_filtered.shape[0], 1))

            # Rotate OpenCV-style world predictions into the UE/ego convention
            points_filtered_ue = points_filtered @ R_CV_TO_UE.T
            aggregated_points.append(points_filtered_ue)
            aggregated_colors.append(colors_filtered)

    if not aggregated_points:
        logger.error("No predicted point data were generated. Exiting.")
        return

    points_pred = np.concatenate(aggregated_points, axis=0)
    colors_pred = np.concatenate(aggregated_colors, axis=0)

    if args.max_height is not None:
        height_mask = points_pred[:, 2] <= args.max_height
        removed_points = int((~height_mask).sum())
        if removed_points > 0:
            logger.info(
                "Removed %d predicted points above max height %.2f m.",
                removed_points,
                args.max_height,
            )
        points_pred = points_pred[height_mask]
        if colors_pred.shape[0] == height_mask.shape[0]:
            colors_pred = colors_pred[height_mask]

    if points_pred.size == 0:
        logger.error("All predicted points were filtered out. Nothing to save or visualize.")
        return

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
        pcd_gt.paint_uniform_color([0, 1, 0])  # Green

    logger.info(f"Successfully loaded ground truth point cloud with {len(pcd_gt.points)} points.")

    if args.no_viz:
        logger.info("Skipping visualization as requested (--no_viz).")
        return

    camera_pose_visuals, camera_pose_counts = build_camera_pose_visuals(predictions, camera_info_list)
    camera_pose_geoms = camera_pose_visuals.get("predicted", []) + camera_pose_visuals.get(
        "ground_truth", []
    )
    if camera_pose_counts.get("predicted"):
        logger.info(
            "Visualizing %d predicted camera pose(s) (blue frustums).",
            camera_pose_counts["predicted"],
        )
    elif predictions:
        logger.warning("Model output did not include camera poses to visualize.")

    if camera_pose_counts.get("ground_truth"):
        logger.info(
            "Visualizing %d ground truth camera pose(s) (green frustums).",
            camera_pose_counts["ground_truth"],
        )
    elif camera_info_list:
        logger.warning(
            "Camera configs were provided but ground truth poses were missing; only predicted poses will be shown."
        )

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
    for geom in camera_pose_geoms:
        vis_pred.add_geometry(geom)

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
def _extract_pose_numpy(pose_value: Any) -> Optional[np.ndarray]:
    """Convert a tensor/array pose into a float64 numpy matrix."""
    if pose_value is None:
        return None

    if torch.is_tensor(pose_value):
        pose_arr = pose_value.detach().cpu().numpy()
    elif isinstance(pose_value, np.ndarray):
        pose_arr = np.array(pose_value, copy=True)
    elif isinstance(pose_value, (list, tuple)):
        pose_arr = np.asarray(pose_value, dtype=np.float64)
    else:
        return None

    if pose_arr.ndim == 3:
        if pose_arr.shape[0] == 0:
            return None
        pose_arr = pose_arr[0]

    if pose_arr.shape != (4, 4):
        logger.warning("Pose has invalid shape %s (expected 4x4).", pose_arr.shape)
        return None

    return pose_arr.astype(np.float64)


def _align_predicted_camera_poses(
    pred_pose_list: List[np.ndarray], gt_pose_list: List[np.ndarray]
) -> List[np.ndarray]:
    """Align predicted poses with ground truth when available."""
    if not pred_pose_list:
        return []

    if not gt_pose_list:
        return list(pred_pose_list)

    try:
        alignment = gt_pose_list[0] @ np.linalg.inv(pred_pose_list[0])
    except np.linalg.LinAlgError as exc:
        logger.warning("Failed to align predicted camera poses: %s", exc)
        return list(pred_pose_list)

    return [alignment @ pose for pose in pred_pose_list]


def _cv_pose_to_ue(pose: np.ndarray) -> np.ndarray:
    """Convert a camera pose from OpenCV coordinates into the UE convention."""
    return (CV_TO_UE_TRANSFORM @ pose).astype(np.float64)


def _create_frustum_geometry(
    pose: np.ndarray,
    color: Tuple[float, float, float],
    scale: float,
    marker_radius: float,
) -> List[Any]:
    """Create a frustum plus marker for a single camera pose."""
    if not OPEN3D_AVAILABLE:
        return []

    half_w = 0.35 * scale
    half_h = 0.25 * scale
    depth = 0.8 * scale

    points_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    homog = np.hstack((points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)))
    points_world = (pose @ homog.T).T[:, :3]

    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    line_colors = np.tile(np.asarray(color, dtype=np.float64), (lines.shape[0], 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(line_colors)

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
    marker.paint_uniform_color(color)
    marker.translate(points_world[0])

    return [frustum, marker]


def _build_camera_pose_geoms(
    pose_list: List[np.ndarray],
    color: Tuple[float, float, float],
    scale: float,
    marker_radius: float,
) -> List[Any]:
    visuals: List[Any] = []
    for pose in pose_list:
        visuals.extend(_create_frustum_geometry(pose, color, scale, marker_radius))
    return visuals


def build_camera_pose_visuals(
    predictions: List[Dict[str, torch.Tensor]],
    camera_info_list: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, List[Any]], Dict[str, int]]:
    """Prepare Open3D geometries that visualize predicted/ground-truth camera poses."""
    visuals: Dict[str, List[Any]] = {"predicted": [], "ground_truth": []}
    counts = {"predicted": 0, "ground_truth": 0}

    if not OPEN3D_AVAILABLE:
        return visuals, counts

    pred_pose_list: List[np.ndarray] = []
    for pred in predictions:
        pose_np = _extract_pose_numpy(pred.get("camera_poses"))
        if pose_np is not None:
            pred_pose_list.append(pose_np)
    counts["predicted"] = len(pred_pose_list)

    gt_pose_list: List[np.ndarray] = []
    if camera_info_list:
        for cam_info in camera_info_list:
            pose_value = cam_info.get("pose_C2E_cv")
            if pose_value is None:
                pose_value = cam_info.get("pose_C2W_cv")
            pose_np = _extract_pose_numpy(pose_value)
            if pose_np is not None:
                gt_pose_list.append(pose_np)
    counts["ground_truth"] = len(gt_pose_list)

    pred_pose_aligned = _align_predicted_camera_poses(pred_pose_list, gt_pose_list)

    pred_pose_ue = [_cv_pose_to_ue(pose) for pose in pred_pose_aligned]
    gt_pose_ue = [_cv_pose_to_ue(pose) for pose in gt_pose_list]

    if pred_pose_ue:
        visuals["predicted"] = _build_camera_pose_geoms(
            pred_pose_ue, (0.2, 0.4, 1.0), 1.1, 0.08
        )
    if gt_pose_ue:
        visuals["ground_truth"] = _build_camera_pose_geoms(
            gt_pose_ue, (0.1, 0.8, 0.3), 0.9, 0.07
        )

    return visuals, counts


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
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a YAML config file or a directory containing multiple YAML configs.",
    )
    parser.add_argument("--gt_pcd_path", type=str, required=True, help="Path to the ground truth point cloud file (.pcd format).")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient mode for inference (slower).")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint-best.pth", help="Local MapAnything checkpoint to load (.pth or .safetensors).")
    parser.add_argument("--hydra_config_path", type=str, default="configs/train.yaml", help="Hydra config used to instantiate the model.")
    parser.add_argument("--config_json_path", type=str, default="scripts/local_models/config.json", help="Optional model config JSON describing encoder/heads.")
    parser.add_argument(
        "--aggregate_frame",
        choices=["auto", "ego", "world"],
        default="auto",
        help=(
            "Coordinate frame to aggregate predicted points when external poses are provided. "
            "'auto' keeps the old ego-centric behavior for a single config and switches to the "
            "global world frame when multiple configs are used."
        ),
    )
    parser.add_argument(
        "--config_overrides",
        nargs="*",
        default=None,
        help="Optional Hydra overrides (defaults target the released MapAnything model).",
    )
    parser.add_argument("--strict_load", action="store_true", help="Enable strict checkpoint loading.")
    parser.add_argument("--no_viz", action="store_true", help="Disable Open3D visualization (useful for headless runs).")
    parser.add_argument("--save_pred_path", type=str, default=None, help="Optional path to save aggregated predicted point cloud (.pcd/.ply).")
    parser.add_argument(
        "--max_height",
        type=float,
        default=2.0,
        help="Discard predicted points above this Z height (in meters) before saving or visualizing.",
    )
    parser.add_argument(
        "--pose_log_path",
        type=str,
        default="color_compare_pose_log.txt",
        help="File to store pose matrices/errors. Provide an empty string to disable file logging.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_pcd_path = Path(args.gt_pcd_path)
    checkpoint_path = Path(args.checkpoint_path)
    hydra_config_path = Path(args.hydra_config_path)
    config_json_path = Path(args.config_json_path)
    config_input_path = Path(args.config_path).expanduser() if args.config_path else None
    pose_log_path = Path(args.pose_log_path).expanduser() if args.pose_log_path else None

    pose_log_sink = PoseLogSink(pose_log_path)
    command_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    logger.info("记录当前运行命令：%s", command_line)
    if pose_log_path:
        logger.info("位姿日志将附加写入：%s", pose_log_path)
    else:
        logger.info("未设置 --pose_log_path，位姿信息仅会在终端输出。")
    pose_log_sink.log_run_header(command_line)
    try:
        _run_color_compare(
            args,
            image_dir,
            gt_pcd_path,
            checkpoint_path,
            hydra_config_path,
            config_json_path,
            config_input_path,
            pose_log_sink,
        )
    finally:
        pose_log_sink.close()


if __name__ == "__main__":
    main()
