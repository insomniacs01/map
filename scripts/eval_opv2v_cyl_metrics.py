#!/usr/bin/env python3
"""
Compute quick metrics for OPV2V cylindrical cooperative scenes.

This script is meant to make "visualization looks good" and "numbers look good"
consistent by enforcing the same:
  - agent selection (num_views, max_agent_distance, agent_selection_policy)
  - point cloud crop (z_min/z_max/radius_max)

Metrics:
  - pose_err_rel_mean: mean relative SE(3) error (translation meters / rotation degrees)
    computed on inv(T0)@Ti against GT, averaged over non-reference views.
  - Chamfer mean + BEV IoU against GT LiDAR (main or fused across selected agents)
    for both PredPose points and GTPose points (pose-decoupled).
  - pts-in-box median for sanity checking vehicle density.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/eval_opv2v_cyl_metrics.py \
    --checkpoint experiments/opv2v_cyl_geom_quick/checkpoint-validate_2021_09_09_19_27_35_2314_geom_lidar_v1.pth \
    --split validate --sequence 2021_09_11_00_33_16 --main_agent 1016 \
    --frames 000269,000271,000273,000275,000277 --num_views 4 \
    --max_agent_distance 60 --agent_selection_policy nearest --gt_lidar_mode fused \
    --radius_max 60 --out_json eval_runs/demo_alignment/metrics_r60.json
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from data_processing.opv2v_pose_utils import cords_to_pose, load_frame_metadata
from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.utils.geometry import quaternion_to_rotation_matrix
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local


REPO_ROOT = Path(__file__).resolve().parents[1]

_CARLA_TO_CV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    try:
        with pcd_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("DATA"):
                    break
            points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    except (UnicodeDecodeError, ValueError):
        points = _load_binary_pcd_xyz(pcd_path)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points


def _load_binary_pcd_xyz(pcd_path: Path) -> np.ndarray:
    try:
        import open3d as o3d  # local import to keep optional dependency
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Binary PCD parsing requires open3d. Install open3d or export ASCII PCDs."
        ) from exc
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd is None:
        raise RuntimeError(f"Failed to load PCD: {pcd_path}")
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return points


def _transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    pts_h = np.concatenate(
        [points.astype(np.float64, copy=False), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    pts_out = (T_dst_src @ pts_h.T).T[:, :3]
    return pts_out.astype(np.float32)


def _load_gt_lidar_points(
    *,
    lidar_root: Path,
    split: str,
    sequence: str,
    frame: str,
    main_agent: str,
    agent_ids: List[str],
    mode: str,
) -> np.ndarray:
    """Load GT LiDAR points in the main-agent ego frame (CARLA coordinates)."""

    if mode not in ("main", "fused"):
        raise ValueError(f"Unknown --gt_lidar_mode '{mode}' (expected: main|fused)")

    lidar_root = lidar_root.expanduser().resolve()
    main_yaml = lidar_root / split / sequence / main_agent / f"{frame}.yaml"
    if not main_yaml.is_file():
        raise FileNotFoundError(main_yaml)

    meta_main = load_frame_metadata(main_yaml)
    T_world_main = cords_to_pose(meta_main["lidar_pose"])
    T_main_world = np.linalg.inv(T_world_main)

    selected_agents = [main_agent] if mode == "main" else sorted(set([main_agent, *agent_ids]))

    all_points: List[np.ndarray] = []
    for agent in selected_agents:
        pcd_path = lidar_root / split / sequence / agent / f"{frame}.pcd"
        yaml_path = lidar_root / split / sequence / agent / f"{frame}.yaml"
        if not pcd_path.is_file() or not yaml_path.is_file():
            continue
        pts_agent = _load_ascii_pcd_xyz(pcd_path)
        if agent == main_agent:
            pts_main = pts_agent
        else:
            meta_agent = load_frame_metadata(yaml_path)
            T_world_agent = cords_to_pose(meta_agent["lidar_pose"])
            T_main_agent = T_main_world @ T_world_agent
            pts_main = _transform_points(pts_agent, T_main_agent)
        all_points.append(pts_main)

    if not all_points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(all_points, axis=0)


def _filter_points(
    points: np.ndarray,
    *,
    z_min: float | None,
    z_max: float | None,
    radius_max: float | None,
) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= points[:, 2] <= float(z_max)
    if radius_max is not None and float(radius_max) > 0:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= float(radius_max)
    return points[mask]


def _rotation_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    r_rel = R_gt.T @ R_pred
    trace_val = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))


def _pose_error(pred_pose: np.ndarray, gt_pose: np.ndarray) -> tuple[float, float]:
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    pred_R = pred_pose[:3, :3]
    pred_t = pred_pose[:3, 3]
    abs_trans = float(np.linalg.norm(pred_t - gt_t))
    abs_rot_deg = _rotation_angle_deg(pred_R, gt_R)
    return abs_trans, abs_rot_deg


def _rigid_align_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (no scale) that maps src -> dst."""
    if src.shape[0] < 2:
        raise ValueError("Need at least 2 points for rigid alignment.")
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (src_c.T @ dst_c) / float(src.shape[0])
    U, _, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def _pose_matrix_from_quat_trans(quat_xyzw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    quat_xyzw = torch.nan_to_num(quat_xyzw, nan=0.0, posinf=0.0, neginf=0.0)
    trans = torch.nan_to_num(trans, nan=0.0, posinf=0.0, neginf=0.0)
    eps = 1e-6
    norms = quat_xyzw.norm(dim=1, keepdim=True)
    if torch.any(norms < eps):
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat_xyzw.device, dtype=quat_xyzw.dtype).unsqueeze(0)
        quat_xyzw = torch.where(norms < eps, ident, quat_xyzw)
    rot = quaternion_to_rotation_matrix(quat_xyzw)
    mat = torch.eye(4, device=rot.device, dtype=rot.dtype).unsqueeze(0).repeat(rot.shape[0], 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


def _chamfer_mean(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    if pred.size == 0 or gt.size == 0:
        return float("nan"), float("nan"), float("nan")
    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    dist_pred_gt, _ = gt_tree.query(pred, k=1)
    dist_gt_pred, _ = pred_tree.query(gt, k=1)
    a = float(np.mean(dist_pred_gt))
    b = float(np.mean(dist_gt_pred))
    return a, b, float((a + b) * 0.5)


def _bev_occupancy(points: np.ndarray, bev_range: float, bev_resolution: float) -> np.ndarray:
    grid_size = int(np.ceil((2 * bev_range) / bev_resolution))
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    if points.size == 0:
        return grid
    coords = points[:, :2]
    mask = np.all(np.abs(coords) <= bev_range, axis=1)
    coords = coords[mask]
    if coords.size == 0:
        return grid
    idx = ((coords + bev_range) / bev_resolution).astype(int)
    idx = np.clip(idx, 0, grid_size - 1)
    grid[idx[:, 0], idx[:, 1]] = True
    return grid


def _bev_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def _count_points_in_box(points: np.ndarray, box: np.ndarray) -> int:
    x, y, z, l, w, h, yaw = [float(v) for v in box]
    rel = points.astype(np.float64) - np.array([x, y, z], dtype=np.float64)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    xloc = c * rel[:, 0] - s * rel[:, 1]
    yloc = s * rel[:, 0] + c * rel[:, 1]
    zloc = rel[:, 2]
    inside = (np.abs(xloc) <= l / 2.0) & (np.abs(yloc) <= w / 2.0) & (np.abs(zloc) <= h / 2.0)
    return int(inside.sum())


def _aggregate_pred_points(
    views: List[dict],
    preds: List[dict],
    *,
    mask_thresh: float | None,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return (predpose_points_carla, gtpose_points_carla, weights_per_point)."""

    pts_cv_pred_list = []
    pts_cv_gtpose_list = []
    weights_list = []

    for view, pred in zip(views, preds):
        pts3d = pred.get("pts3d")
        pts3d_cam = pred.get("pts3d_cam")
        if pts3d is None or pts3d_cam is None:
            continue
        if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
            continue
        if pts3d_cam.ndim != 4 or pts3d_cam.shape[-1] != 3:
            continue

        pts_cv_pred_list.append(pts3d.reshape(-1, 3))

        weights = pred.get("non_ambiguous_mask")
        if weights is None or not torch.is_tensor(weights):
            weights_list.append(torch.ones((pts3d.numel() // 3,), device=pts3d.device, dtype=torch.float32))
        else:
            weights_list.append(weights.reshape(-1).float())

        pose_gt = view.get("camera_pose")
        if pose_gt is not None and torch.is_tensor(pose_gt) and pose_gt.ndim == 3 and pose_gt.shape[-2:] == (4, 4):
            rot = pose_gt[:, :3, :3]
            trans = pose_gt[:, :3, 3]
            pts_gtpose = torch.einsum("bij,bhwj->bhwi", rot, pts3d_cam) + trans[:, None, None, :]
            pts_cv_gtpose_list.append(pts_gtpose.reshape(-1, 3))

    if not pts_cv_pred_list:
        raise RuntimeError("No valid predicted points found in model outputs.")

    pts_cv_pred = torch.cat(pts_cv_pred_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    pts_cv_gtpose = torch.cat(pts_cv_gtpose_list, dim=0) if pts_cv_gtpose_list else None

    if mask_thresh is None or float(mask_thresh) < 0:
        keep = torch.ones_like(weights, dtype=torch.bool)
    else:
        keep = weights > float(mask_thresh)

    pts_carla_pred = pts_cv_pred[keep].detach().cpu().numpy().astype(np.float32) @ _CARLA_TO_CV
    pts_carla_gtpose = None
    if pts_cv_gtpose is not None and pts_cv_gtpose.shape[0] == keep.shape[0]:
        pts_carla_gtpose = pts_cv_gtpose[keep].detach().cpu().numpy().astype(np.float32) @ _CARLA_TO_CV
    return pts_carla_pred, pts_carla_gtpose, weights.detach().cpu().numpy().astype(np.float32)


def _scene_indices_for_frames(
    dataset: OPV2VCoopCylindricalDataset,
    *,
    sequence: str | None,
    frames: Sequence[str],
    main_agent: str | None,
    frame_specs: Sequence[Tuple[str, str]] | None = None,
) -> List[int]:
    wanted: List[int] = []
    frames_set = {str(f) for f in frames}
    allowed = {(str(seq), str(frame)) for seq, frame in frame_specs} if frame_specs else None
    for idx, scene in enumerate(dataset.scenes):
        scene_seq = str(scene.get("sequence"))
        scene_frame = str(scene.get("frame"))
        if allowed is not None:
            if (scene_seq, scene_frame) not in allowed:
                continue
        else:
            if sequence is not None and scene_seq != str(sequence):
                continue
            if frames_set and scene_frame not in frames_set:
                continue
        if main_agent is not None and str(main_agent) not in list(scene.get("agents") or []):
            continue
        wanted.append(int(idx))
    if not wanted:
        raise RuntimeError(
            f"No scenes found for seq={sequence} frames={frames} main_agent={main_agent}"
        )
    return wanted


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _load_frames_from_summary(path: Path) -> Tuple[str | None, List[str], List[str], List[Tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames") or []
    sequences = []
    main_agents = []
    frame_ids = []
    frame_specs: List[Tuple[str, str]] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        seq = str(item.get("sequence")) if item.get("sequence") is not None else None
        frame = str(item.get("frame")) if item.get("frame") is not None else None
        agent = str(item.get("main_agent")) if item.get("main_agent") is not None else None
        if seq and frame:
            sequences.append(seq)
            frame_ids.append(frame)
            frame_specs.append((seq, frame))
        if agent:
            main_agents.append(agent)
    sequence = sequences[0] if sequences and len(set(sequences)) == 1 else None
    main_agent = main_agents[0] if main_agents and len(set(main_agents)) == 1 else None
    return sequence, frame_ids, main_agents, frame_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated frame ids (e.g. 000269,000271)")
    parser.add_argument("--frames_json", type=Path, default=None, help="Optional JSON containing explicit frames.")
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument(
        "--task",
        type=str,
        default="images_only",
        help="Model task config under `configs/model/task/` (e.g. images_only, posed_sfm, mvs).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=200_000, help="Max points per cloud for Chamfer")
    parser.add_argument("--radius_max", type=float, default=60.0)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--bev_resolution", type=float, default=0.5)
    parser.add_argument(
        "--gt_lidar_mode",
        type=str,
        default="main",
        choices=("main", "fused"),
        help="Which GT LiDAR to use: main-agent only or fused across selected agents.",
    )
    parser.add_argument(
        "--max_agent_distance",
        type=float,
        default=None,
        help="Only sample cooperative agents whose GT ego distance to main is <= this XY radius (meters).",
    )
    parser.add_argument(
        "--agent_selection_policy",
        type=str,
        default="random",
        choices=("random", "nearest"),
        help="How to select agents when more than num_views are available.",
    )
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument(
        "--use_torch_hub",
        action="store_true",
        help="Allow loading encoder weights via torch hub if missing from checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    ckpt_path = (REPO_ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    local_cfg = {
        "path": str((REPO_ROOT / "configs" / "train.yaml").resolve()),
        "checkpoint_path": str(ckpt_path),
        "config_overrides": [
            "machine=local_a800",
            "model=mapanything",
            f"model.encoder.uses_torch_hub={'true' if args.use_torch_hub else 'false'}",
            f"model/task={args.task}",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)
    model.eval()

    frames = _parse_csv_list(args.frames)
    sequence = args.sequence
    main_agent = args.main_agent
    frame_specs = None
    if args.frames_json is not None:
        summary_path = (REPO_ROOT / args.frames_json).resolve() if not args.frames_json.is_absolute() else args.frames_json
        seq_from_json, frames_from_json, main_agents, frame_specs = _load_frames_from_summary(summary_path)
        if sequence is None:
            sequence = seq_from_json
        if not frames:
            frames = frames_from_json
        if main_agent is None and main_agents:
            if len(set(main_agents)) == 1:
                main_agent = main_agents[0]

    dataset = OPV2VCoopCylindricalDataset(
        split=args.split,
        ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
        depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
        camera_ids=(0, 1, 2, 3),
        num_views=int(args.num_views),
        variable_num_views=False,
        resolution=(1008, 252),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=777,
        include_vehicle_boxes=True,
        max_num_boxes=128,
        bbox_range=120.0,
        min_agents=2,
        min_num_views=int(args.num_views),
        max_num_views=int(args.num_views),
        panorama_resolution=(1008, 252),
        panorama_vertical_fov_deg=90.0,
        panorama_elevation_center_deg=0.0,
        view_selection_margin=0.05,
        max_num_retries=0,
        main_agent=main_agent,
        ensure_main_agent_first=True,
        max_agent_distance=float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        agent_selection_policy=str(args.agent_selection_policy),
    )

    wanted = _scene_indices_for_frames(
        dataset,
        sequence=str(sequence) if sequence is not None else None,
        frames=frames,
        main_agent=str(main_agent) if main_agent is not None else None,
        frame_specs=frame_specs,
    )
    wanted = sorted(wanted, key=lambda i: str(dataset.scenes[i].get("frame", "")))

    per_frame: List[Dict[str, Any]] = []
    all_pose_trans: List[float] = []
    all_pose_rot: List[float] = []
    all_pose_ate_trans: List[float] = []
    all_pose_ate_rot: List[float] = []
    all_chamfer_pred: List[float] = []
    all_chamfer_gtpose: List[float] = []
    all_bev_pred: List[float] = []
    all_bev_gtpose: List[float] = []
    all_pts_in_box_pred: List[float] = []
    all_pts_in_box_gtpose: List[float] = []
    all_pts_in_box_gt: List[float] = []

    bev_range = float(args.radius_max) if args.radius_max is not None and args.radius_max > 0 else 120.0

    for scene_idx in wanted:
        subset = torch.utils.data.Subset(dataset, [int(scene_idx)])
        loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
        views = next(iter(loader))
        for view in views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device)

        scene = dataset.scenes[int(scene_idx)]
        sequence_id = str(scene.get("sequence"))
        frame_id = str(scene.get("frame"))
        main_agent_id = str(views[0].get("main_agent_id", "")) if views else ""
        if not main_agent_id and main_agent is not None:
            main_agent_id = str(main_agent)
        agent_ids = [str(v.get("agent_id")[0]) if isinstance(v.get("agent_id"), (list, tuple)) and v.get("agent_id") else str(v.get("agent_id")) for v in views]
        agent_ids = [a for a in agent_ids if a]
        if main_agent_id and main_agent_id not in agent_ids and agent_ids:
            main_agent_id = agent_ids[0]

        with torch.no_grad():
            preds = model(views)

        # Pose errors (relative to view0).
        gt_poses_cv = []
        pred_poses_cv = []
        for view, pred in zip(views, preds):
            pose_gt = view.get("camera_pose")
            if pose_gt is not None and torch.is_tensor(pose_gt) and pose_gt.ndim == 3 and pose_gt.shape[-2:] == (4, 4):
                gt_poses_cv.append(pose_gt[0].detach().cpu().numpy().astype(np.float64))
            else:
                gt_poses_cv.append(None)

            quat = pred.get("cam_quats")
            trans = pred.get("cam_trans")
            if quat is not None and trans is not None:
                pose_pred = _pose_matrix_from_quat_trans(quat, trans)
                pred_poses_cv.append(pose_pred[0].detach().cpu().numpy().astype(np.float64))
            else:
                pred_poses_cv.append(None)

        pose_trans = []
        pose_rot = []
        pose_ate_trans = []
        pose_ate_rot = []
        if gt_poses_cv and pred_poses_cv and gt_poses_cv[0] is not None and pred_poses_cv[0] is not None:
            gt_ref_inv = np.linalg.inv(gt_poses_cv[0])
            pred_ref_inv = np.linalg.inv(pred_poses_cv[0])
            for i in range(1, min(len(gt_poses_cv), len(pred_poses_cv))):
                gt_pose = gt_poses_cv[i]
                pred_pose = pred_poses_cv[i]
                if gt_pose is None or pred_pose is None:
                    continue
                gt_rel = gt_ref_inv @ gt_pose
                pred_rel = pred_ref_inv @ pred_pose
                abs_trans, abs_rot = _pose_error(pred_rel, gt_rel)
                pose_trans.append(abs_trans)
                pose_rot.append(abs_rot)

            valid_pairs = [
                (pred, gt)
                for pred, gt in zip(pred_poses_cv, gt_poses_cv)
                if pred is not None and gt is not None
            ]
            if len(valid_pairs) >= 2:
                src = np.stack([p[:3, 3] for p, _ in valid_pairs], axis=0)
                dst = np.stack([g[:3, 3] for _, g in valid_pairs], axis=0)
                R_align, t_align = _rigid_align_umeyama(src, dst)
                for pred_pose, gt_pose in valid_pairs:
                    pred_aligned = pred_pose.copy()
                    pred_aligned[:3, :3] = R_align @ pred_pose[:3, :3]
                    pred_aligned[:3, 3] = R_align @ pred_pose[:3, 3] + t_align
                    abs_trans, abs_rot = _pose_error(pred_aligned, gt_pose)
                    pose_ate_trans.append(abs_trans)
                    pose_ate_rot.append(abs_rot)

        # Point clouds and metrics.
        gt_points = _load_gt_lidar_points(
            lidar_root=REPO_ROOT / "data" / "opv2v",
            split=str(args.split),
            sequence=str(sequence_id),
            frame=str(frame_id),
            main_agent=str(main_agent_id),
            agent_ids=agent_ids,
            mode=str(args.gt_lidar_mode),
        )

        gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
        gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
        gt_boxes = gt_boxes[gt_mask]

        pred_points, gtpose_points, _weights = _aggregate_pred_points(
            views, preds, mask_thresh=float(args.mask_thresh) if args.mask_thresh is not None else None
        )

        gt_points_f = _filter_points(gt_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
        pred_points_f = _filter_points(pred_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
        gtpose_points_f = (
            _filter_points(gtpose_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
            if gtpose_points is not None
            else None
        )

        gt_points_s = _downsample(gt_points_f, int(args.max_points), rng)
        pred_points_s = _downsample(pred_points_f, int(args.max_points), rng)
        gtpose_points_s = _downsample(gtpose_points_f, int(args.max_points), rng) if gtpose_points_f is not None else None

        chamfer_pred = _chamfer_mean(pred_points_s, gt_points_s)
        chamfer_gtpose = _chamfer_mean(gtpose_points_s, gt_points_s) if gtpose_points_s is not None else (float("nan"), float("nan"), float("nan"))

        bev_gt = _bev_occupancy(gt_points_f, bev_range, float(args.bev_resolution))
        bev_pred = _bev_occupancy(pred_points_f, bev_range, float(args.bev_resolution))
        bev_pred_iou = _bev_iou(bev_pred, bev_gt)
        bev_gtpose_iou = float("nan")
        if gtpose_points_f is not None:
            bev_gtpose = _bev_occupancy(gtpose_points_f, bev_range, float(args.bev_resolution))
            bev_gtpose_iou = _bev_iou(bev_gtpose, bev_gt)

        gt_counts = [_count_points_in_box(gt_points_f, box) for box in gt_boxes] if gt_boxes.size else []
        pred_counts = [_count_points_in_box(pred_points_f, box) for box in gt_boxes] if gt_boxes.size else []
        gtpose_counts = (
            [_count_points_in_box(gtpose_points_f, box) for box in gt_boxes] if (gtpose_points_f is not None and gt_boxes.size) else []
        )
        gt_median = float(np.median(gt_counts)) if gt_counts else float("nan")
        pred_median = float(np.median(pred_counts)) if pred_counts else float("nan")
        gtpose_median = float(np.median(gtpose_counts)) if gtpose_counts else float("nan")

        row = {
            "sequence": sequence_id,
            "frame": frame_id,
            "main_agent": main_agent_id,
            "agent_ids": agent_ids,
            "pose_err_rel_trans_mean": float(np.mean(pose_trans)) if pose_trans else float("nan"),
            "pose_err_rel_rot_mean": float(np.mean(pose_rot)) if pose_rot else float("nan"),
            "pose_ate_trans_mean": float(np.mean(pose_ate_trans)) if pose_ate_trans else float("nan"),
            "pose_ate_rot_mean": float(np.mean(pose_ate_rot)) if pose_ate_rot else float("nan"),
            "chamfer_predpose_mean": chamfer_pred[2],
            "chamfer_predpose_pred_to_gt": chamfer_pred[0],
            "chamfer_predpose_gt_to_pred": chamfer_pred[1],
            "chamfer_gtpose_mean": chamfer_gtpose[2],
            "chamfer_gtpose_pred_to_gt": chamfer_gtpose[0],
            "chamfer_gtpose_gt_to_pred": chamfer_gtpose[1],
            "bev_iou_predpose": float(bev_pred_iou),
            "bev_iou_gtpose": float(bev_gtpose_iou),
            "pts_in_box_gt_median": gt_median,
            "pts_in_box_predpose_median": pred_median,
            "pts_in_box_gtpose_median": gtpose_median,
        }
        per_frame.append(row)

        if pose_trans:
            all_pose_trans.extend(pose_trans)
        if pose_rot:
            all_pose_rot.extend(pose_rot)
        if pose_ate_trans:
            all_pose_ate_trans.extend(pose_ate_trans)
        if pose_ate_rot:
            all_pose_ate_rot.extend(pose_ate_rot)
        if math.isfinite(chamfer_pred[2]):
            all_chamfer_pred.append(chamfer_pred[2])
        if math.isfinite(chamfer_gtpose[2]):
            all_chamfer_gtpose.append(chamfer_gtpose[2])
        if math.isfinite(bev_pred_iou):
            all_bev_pred.append(float(bev_pred_iou))
        if math.isfinite(bev_gtpose_iou):
            all_bev_gtpose.append(float(bev_gtpose_iou))
        if math.isfinite(gt_median):
            all_pts_in_box_gt.append(gt_median)
        if math.isfinite(pred_median):
            all_pts_in_box_pred.append(pred_median)
        if math.isfinite(gtpose_median):
            all_pts_in_box_gtpose.append(gtpose_median)

        print(
            f"[{frame_id}] pose_err_rel_mean={row['pose_err_rel_trans_mean']:.3f}m/{row['pose_err_rel_rot_mean']:.3f}° "
            f"| chamfer(predpose)={row['chamfer_predpose_mean']:.3f} "
            f"| bev_iou(predpose)={row['bev_iou_predpose']:.3f} "
            f"| pts_in_box med GT/Pred/GTPose={gt_median:.0f}/{pred_median:.0f}/{gtpose_median:.0f}"
        )

    summary = {
        "checkpoint": str(ckpt_path),
        "split": str(args.split),
        "sequence": str(sequence) if sequence is not None else None,
        "frames": frames,
        "frame_specs": frame_specs,
        "main_agent": str(main_agent) if main_agent is not None else None,
        "num_views": int(args.num_views),
        "max_agent_distance": float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        "agent_selection_policy": str(args.agent_selection_policy),
        "gt_lidar_mode": str(args.gt_lidar_mode),
        "z_min": float(args.z_min) if args.z_min is not None else None,
        "z_max": float(args.z_max) if args.z_max is not None else None,
        "radius_max": float(args.radius_max) if args.radius_max is not None else None,
        "bev_resolution": float(args.bev_resolution),
        "pose_err_rel_trans_mean": float(np.mean(all_pose_trans)) if all_pose_trans else float("nan"),
        "pose_err_rel_rot_mean": float(np.mean(all_pose_rot)) if all_pose_rot else float("nan"),
        "pose_ate_trans_mean": float(np.mean(all_pose_ate_trans)) if all_pose_ate_trans else float("nan"),
        "pose_ate_rot_mean": float(np.mean(all_pose_ate_rot)) if all_pose_ate_rot else float("nan"),
        "chamfer_predpose_mean": float(np.mean(all_chamfer_pred)) if all_chamfer_pred else float("nan"),
        "chamfer_gtpose_mean": float(np.mean(all_chamfer_gtpose)) if all_chamfer_gtpose else float("nan"),
        "bev_iou_predpose_mean": float(np.mean(all_bev_pred)) if all_bev_pred else float("nan"),
        "bev_iou_gtpose_mean": float(np.mean(all_bev_gtpose)) if all_bev_gtpose else float("nan"),
        "pts_in_box_gt_median_mean": float(np.mean(all_pts_in_box_gt)) if all_pts_in_box_gt else float("nan"),
        "pts_in_box_predpose_median_mean": float(np.mean(all_pts_in_box_pred)) if all_pts_in_box_pred else float("nan"),
        "pts_in_box_gtpose_median_mean": float(np.mean(all_pts_in_box_gtpose)) if all_pts_in_box_gtpose else float("nan"),
        "per_frame": per_frame,
    }

    print(
        "[SUMMARY] "
        f"pose_err_rel_mean={summary['pose_err_rel_trans_mean']:.3f}m/{summary['pose_err_rel_rot_mean']:.3f}° "
        f"| chamfer(predpose)={summary['chamfer_predpose_mean']:.3f} "
        f"| chamfer(gtpose)={summary['chamfer_gtpose_mean']:.3f} "
        f"| bev_iou(predpose)={summary['bev_iou_predpose_mean']:.3f} "
        f"| bev_iou(gtpose)={summary['bev_iou_gtpose_mean']:.3f}"
    )

    if args.out_json is not None:
        out_path = args.out_json
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
