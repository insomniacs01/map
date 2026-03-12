#!/usr/bin/env python3
"""Render correct coop point-cloud HTML for OPV2V with PredPose/GTPose side-by-side.

This script is meant for debugging the exact issue discussed in `opv2v_multicar_conversation_20260312.md`:
- compare coop prediction against fused coop GT (not ego-only GT)
- separate pose errors from geometry errors
- keep RGB point colors so obviously-wrong geometry is easier to spot

Definitions
-----------
- PredPose: use `pred["pts3d"]` (model's own predicted placement)
- GTPose:  use GT camera pose on `pred["pts3d_cam"]` (geometry only)
- GT:      OPV2V LiDAR in main-agent ego frame, with optional fused coop agents
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import batch_eval as be  # type: ignore
from data_processing.opv2v_pose_utils import cords_to_pose, load_ascii_pcd_xyz, load_frame_metadata


_CARLA_TO_CV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

_AGENT_PALETTE = np.array(
    [
        [0.121, 0.466, 0.705],
        [1.000, 0.498, 0.054],
        [0.172, 0.627, 0.172],
        [0.839, 0.152, 0.156],
        [0.580, 0.404, 0.741],
        [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761],
        [0.498, 0.498, 0.498],
    ],
    dtype=np.float32,
)


def _patch_torch_hub_for_local_dinov2(repo_dir: Optional[Path]) -> None:
    if repo_dir is None:
        return
    repo_dir = repo_dir.expanduser().resolve()
    original_load = torch.hub.load

    def _patched_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs = dict(kwargs)
            kwargs["source"] = "local"
            repo_or_dir = str(repo_dir)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = _patched_load  # type: ignore[assignment]


def _points_opencv_to_carla(points_cv: np.ndarray) -> np.ndarray:
    if points_cv.size == 0:
        return points_cv
    basis = _CARLA_TO_CV.astype(points_cv.dtype, copy=False)
    return points_cv @ basis


def _transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    pts_h = np.concatenate(
        [points.astype(np.float64, copy=False), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (T_dst_src @ pts_h.T).T[:, :3]
    return out.astype(np.float32, copy=False)


def _load_gt_lidar_points(
    *,
    lidar_root: Path,
    split: str,
    sequence: str,
    frame: str,
    main_agent: str,
    coop_agents: Sequence[str],
    mode: str,
) -> np.ndarray:
    if mode not in ("main", "fused"):
        raise ValueError(f"unknown gt lidar mode: {mode}")

    main_yaml = lidar_root / split / sequence / main_agent / f"{frame}.yaml"
    if not main_yaml.is_file():
        raise FileNotFoundError(main_yaml)

    meta_main = load_frame_metadata(main_yaml)
    T_world_main = cords_to_pose(meta_main["lidar_pose"])
    T_main_world = np.linalg.inv(T_world_main)

    selected_agents = [main_agent] if mode == "main" else list(dict.fromkeys([main_agent, *[str(a) for a in coop_agents]]))

    clouds: List[np.ndarray] = []
    for agent in selected_agents:
        pcd_path = lidar_root / split / sequence / agent / f"{frame}.pcd"
        yaml_path = lidar_root / split / sequence / agent / f"{frame}.yaml"
        if not pcd_path.is_file() or not yaml_path.is_file():
            continue
        pts = load_ascii_pcd_xyz(pcd_path).astype(np.float32, copy=False)
        if str(agent) == str(main_agent):
            pts_main = pts
        else:
            meta_agent = load_frame_metadata(yaml_path)
            T_world_agent = cords_to_pose(meta_agent["lidar_pose"])
            T_main_agent = T_main_world @ T_world_agent
            pts_main = _transform_points(pts, T_main_agent)
        clouds.append(pts_main)

    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0).astype(np.float32, copy=False)


def _boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    corners_all = []
    for box in boxes:
        x, y, z, l, w, h, yaw = [float(v) for v in box]
        dx = l / 2.0
        dy = w / 2.0
        dz = h / 2.0
        local = np.array(
            [
                [dx, dy, dz],
                [dx, -dy, dz],
                [-dx, -dy, dz],
                [-dx, dy, dz],
                [dx, dy, -dz],
                [dx, -dy, -dz],
                [-dx, -dy, -dz],
                [-dx, dy, -dz],
            ],
            dtype=np.float32,
        )
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        corners = local @ rot.T + np.array([x, y, z], dtype=np.float32)
        corners_all.append(corners)
    if not corners_all:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return np.stack(corners_all, axis=0)


def _iter_box_edges() -> Iterable[Tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def _add_boxes_3d(
    fig: go.Figure,
    corners: np.ndarray,
    *,
    row: int,
    col: int,
    color: str,
    name: str,
    showlegend: bool,
) -> None:
    if corners.size == 0:
        return
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    for corner in corners:
        for a, b in _iter_box_edges():
            xs.extend([float(corner[a, 0]), float(corner[b, 0]), None])
            ys.extend([float(corner[a, 1]), float(corner[b, 1]), None])
            zs.extend([float(corner[a, 2]), float(corner[b, 2]), None])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name=name,
            line=dict(color=color, width=4),
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


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


def _downsample_pair(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    points_ds = points[idx]
    colors_ds = colors[idx] if colors is not None else None
    return points_ds, colors_ds


def _filter_points_and_colors(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    *,
    z_min: Optional[float],
    z_max: Optional[float],
    radius_max: Optional[float],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if points.size == 0:
        return points, colors
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= points[:, 2] <= float(z_max)
    if radius_max is not None and float(radius_max) > 0:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= float(radius_max)
    out_points = points[mask]
    out_colors = colors[mask] if colors is not None else None
    return out_points, out_colors


def _filter_points_only(
    points: np.ndarray,
    *,
    z_min: Optional[float],
    z_max: Optional[float],
    radius_max: Optional[float],
) -> np.ndarray:
    points_out, _ = _filter_points_and_colors(points, None, z_min=z_min, z_max=z_max, radius_max=radius_max)
    return points_out


def _rgb_strings(colors: np.ndarray) -> List[str]:
    colors = np.clip(colors, 0.0, 1.0)
    colors_u8 = np.rint(colors * 255.0).astype(np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors_u8]


def _extract_agent_ids_from_views(views: Sequence[dict], main_agent: str) -> List[str]:
    agents: List[str] = []
    for view in views:
        agent_raw = view.get("agent_id")
        if torch.is_tensor(agent_raw):
            if agent_raw.numel() > 0:
                agent = str(agent_raw.flatten()[0].item())
            else:
                continue
        elif isinstance(agent_raw, (list, tuple)):
            if not agent_raw:
                continue
            agent = str(agent_raw[0])
        elif agent_raw is None:
            continue
        else:
            agent = str(agent_raw)
        if agent not in agents:
            agents.append(agent)
    if main_agent not in agents:
        agents.insert(0, main_agent)
    return agents


def _agent_color_for_points(agent_index: int, count: int) -> np.ndarray:
    color = _AGENT_PALETTE[agent_index % len(_AGENT_PALETTE)]
    return np.repeat(color[None, :], count, axis=0)


def _build_color_mode(
    *,
    rgb_points: Optional[np.ndarray],
    agent_point_colors: np.ndarray,
    points: np.ndarray,
    mode: str,
    solid_rgb: Tuple[float, float, float],
) -> Optional[np.ndarray]:
    if mode == "rgb" and rgb_points is not None:
        return rgb_points
    if mode == "agent":
        return agent_point_colors
    if mode == "solid":
        return np.repeat(np.array(solid_rgb, dtype=np.float32)[None, :], points.shape[0], axis=0)
    if mode == "z":
        z = points[:, 2]
        z_min = float(np.min(z)) if z.size else 0.0
        z_max = float(np.max(z)) if z.size else 1.0
        denom = max(z_max - z_min, 1e-6)
        zn = ((z - z_min) / denom).astype(np.float32)
        return np.stack([zn, 1.0 - np.abs(zn - 0.5) * 2.0, 1.0 - zn], axis=1)
    return None


def _resolved_norm_type(model_arch: str, data_norm_type: Optional[str]) -> str:
    if data_norm_type:
        return str(data_norm_type)
    return "identity" if model_arch == "vggt" else "dinov2"


def _init_model(args: argparse.Namespace, device: torch.device):
    if args.model_arch == "vggt":
        model_name = "vggt"
        overrides = [
            f"machine={args.machine}",
            f"dataset={args.dataset_cfg}",
            f"model={model_name}",
            "model.model_config.load_pretrained_weights=false",
            f"model.model_config.enable_metric_scale_head={'true' if args.vggt_enable_metric_scale_head else 'false'}",
            f"model.model_config.autocast_dtype={args.vggt_autocast_dtype}" if args.vggt_autocast_dtype else None,
            "loss=overall_loss",
        ]
    else:
        overrides = [
            f"machine={args.machine}",
            f"dataset={args.dataset_cfg}",
            "model=mapanything",
            f"model/task={args.model_task}",
            "model.encoder.uses_torch_hub=false",
            "loss=overall_loss",
        ]
    cfg = {
        "path": str(REPO_ROOT / "configs" / "train.yaml"),
        "checkpoint_path": str(args.checkpoint),
        "config_overrides": [v for v in overrides if v],
    }
    model = be.initialize_mapanything_local(cfg, device)
    model.eval()
    return model


def _infer_predictions(
    args: argparse.Namespace,
    model,
    device: torch.device,
    raw_views: List[dict],
):
    norm_type = _resolved_norm_type(args.model_arch, args.data_norm_type)
    processed = be.preprocess_inputs(raw_views, norm_type=norm_type)
    for view in processed:
        view.pop("depth_z", None)
    if not args.keep_camera_poses:
        be.strip_external_calibration_inputs(processed)
    with torch.no_grad():
        if args.model_arch == "vggt":
            be._move_views_to_device(processed, device)
            raw_preds = model(processed)
            preds = be._vggt_preds_to_batch_eval_format(raw_preds, processed)
        else:
            preds = model.infer(processed, memory_efficient_inference=True)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument("--main_agent", type=str, required=True)
    parser.add_argument(
        "--coop_agents",
        type=str,
        required=True,
        help="Comma-separated agent ids used in coop inference; include main agent or it will be auto-added.",
    )
    parser.add_argument("--images_root", type=Path, default=REPO_ROOT / "data" / "opv2v")
    parser.add_argument("--depth_root", type=Path, default=REPO_ROOT / "data" / "opv2v_depth")
    parser.add_argument("--machine", type=str, default="local_a800")
    parser.add_argument("--dataset_cfg", type=str, default="opv2v_coop_ft")
    parser.add_argument("--model_arch", type=str, default="mapanything", choices=("mapanything", "vggt"))
    parser.add_argument("--model_task", type=str, default="calibrated_sfm")
    parser.add_argument("--data_norm_type", type=str, default=None)
    parser.add_argument("--vggt_enable_metric_scale_head", action="store_true")
    parser.add_argument("--vggt_autocast_dtype", type=str, default=None)
    parser.add_argument("--keep_camera_poses", action="store_true")
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--radius_max", type=float, default=60.0)
    parser.add_argument("--gt_lidar_mode", type=str, default="fused", choices=("main", "fused"))
    parser.add_argument("--pred_color_mode", type=str, default="rgb", choices=("rgb", "agent", "solid", "z"))
    parser.add_argument("--include_plotlyjs", type=str, default="include", choices=("include", "cdn", "directory"))
    parser.add_argument("--dinov2_hub_repo_dir", type=Path, default=None)
    parser.add_argument("--out_html", type=Path, required=True)
    args = parser.parse_args()

    _patch_torch_hub_for_local_dinov2(args.dinov2_hub_repo_dir)

    rng = np.random.default_rng(int(args.seed))
    coop_agents = [x.strip() for x in str(args.coop_agents).split(",") if x.strip()]
    if args.main_agent not in coop_agents:
        coop_agents = [args.main_agent, *coop_agents]
    info = be.FrameInfo(
        sequence=str(args.sequence),
        frame=str(args.frame),
        main_agent=str(args.main_agent),
        coop_agents=tuple(coop_agents),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _init_model(args, device)

    raw_views, cam_infos = be.build_coop_raw(args.images_root, args.depth_root, args.split, info)
    preds = _infer_predictions(args, model, device, raw_views)

    gt_ref_pose_cv = np.asarray(raw_views[0]["camera_poses"], dtype=np.float32)
    gt_points = _load_gt_lidar_points(
        lidar_root=args.images_root,
        split=args.split,
        sequence=args.sequence,
        frame=args.frame,
        main_agent=args.main_agent,
        coop_agents=_extract_agent_ids_from_views(raw_views, args.main_agent),
        mode=args.gt_lidar_mode,
    )

    meta = load_frame_metadata(args.images_root / args.split / args.sequence / args.main_agent / f"{args.frame}.yaml")
    gt_boxes = np.zeros((0, 7), dtype=np.float32)
    if hasattr(be, "_extract_vehicle_boxes_in_ego"):
        gt_padded, gt_mask = be._extract_vehicle_boxes_in_ego(meta, max_range=120.0, max_num_boxes=128)
        gt_boxes = gt_padded[gt_mask.astype(bool)]
    gt_corners = _boxes_to_corners(gt_boxes)

    pred_pts_cv_list: List[np.ndarray] = []
    gtpose_pts_cv_list: List[np.ndarray] = []
    pred_rgb_list: List[np.ndarray] = []
    gtpose_rgb_list: List[np.ndarray] = []
    pred_agent_color_list: List[np.ndarray] = []
    gtpose_agent_color_list: List[np.ndarray] = []
    pose_summary_parts: List[str] = []

    gt_ref_inv = np.linalg.inv(np.asarray(cam_infos[0]["pose_C2E_cv"].cpu().numpy(), dtype=np.float64))
    pred_ref_inv: Optional[np.ndarray] = None
    pred_ref_pose_tensor = preds[0].get("camera_poses") if preds else None
    if pred_ref_pose_tensor is not None:
        pred_ref_inv = np.linalg.inv(pred_ref_pose_tensor[0].detach().cpu().numpy().astype(np.float64))

    for view_idx, (raw_view, cam_info, pred) in enumerate(zip(raw_views, cam_infos, preds)):
        pts3d = pred.get("pts3d")
        pts3d_cam = pred.get("pts3d_cam")
        if pts3d is None or pts3d_cam is None:
            continue
        if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
            continue
        if pts3d_cam.ndim != 4 or pts3d_cam.shape[-1] != 3:
            continue

        valid = torch.isfinite(pts3d).all(dim=-1) & torch.isfinite(pts3d_cam).all(dim=-1)
        pred_mask = pred.get("mask")
        if pred_mask is not None:
            valid &= pred_mask[0].squeeze(-1).to(dtype=torch.bool)
        if args.mask_thresh is not None and float(args.mask_thresh) >= 0:
            conf_mask = pred.get("non_ambiguous_mask")
            if conf_mask is not None:
                valid &= conf_mask.reshape(valid.shape)[0] > float(args.mask_thresh)

        pose_gt = cam_info.get("pose_C2E_cv")
        if pose_gt is None:
            continue
        pose_gt = pose_gt[0].detach() if torch.is_tensor(pose_gt) and pose_gt.ndim == 3 else pose_gt.detach() if torch.is_tensor(pose_gt) else pose_gt
        if torch.is_tensor(pose_gt):
            pose_gt_t = pose_gt.to(device=pts3d_cam.device, dtype=pts3d_cam.dtype)
        else:
            pose_gt_t = torch.tensor(np.asarray(pose_gt), device=pts3d_cam.device, dtype=pts3d_cam.dtype)
        rot = pose_gt_t[:3, :3]
        trans = pose_gt_t[:3, 3]
        pts_gtpose = torch.einsum("ij,hwj->hwi", rot, pts3d_cam[0]) + trans[None, None, :]

        pred_sel = pts3d[0][valid[0]].detach().cpu().numpy().astype(np.float32, copy=False)
        gtpose_sel = pts_gtpose[valid[0]].detach().cpu().numpy().astype(np.float32, copy=False)
        if pred_sel.size == 0:
            continue

        color_tensor = pred.get("img_no_norm")
        if color_tensor is not None and color_tensor.ndim == 4 and color_tensor.shape[-1] == 3:
            rgb_sel = color_tensor[0][valid[0]].detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            rgb_sel = None

        agent_raw = raw_view.get("agent_id")
        if isinstance(agent_raw, (list, tuple)) and agent_raw:
            agent_id = str(agent_raw[0])
        elif agent_raw is None:
            agent_id = cam_info.get("name", f"view{view_idx}")
            agent_id = str(agent_id)
        else:
            agent_id = str(agent_raw)
        agent_color = _agent_color_for_points(view_idx, pred_sel.shape[0])

        pred_pts_cv_list.append(pred_sel)
        gtpose_pts_cv_list.append(gtpose_sel)
        pred_agent_color_list.append(agent_color)
        gtpose_agent_color_list.append(agent_color)
        if rgb_sel is not None:
            pred_rgb_list.append(rgb_sel)
            gtpose_rgb_list.append(rgb_sel)

        pred_pose_tensor = pred.get("camera_poses")
        if pred_pose_tensor is not None and pred_ref_inv is not None:
            pred_pose = pred_pose_tensor[0].detach().cpu().numpy().astype(np.float64)
            gt_pose = np.asarray(cam_info["pose_C2E_cv"].cpu().numpy(), dtype=np.float64)
            if gt_pose.ndim == 3:
                gt_pose = gt_pose[0]
            gt_rel = gt_ref_inv @ gt_pose
            pred_rel = pred_ref_inv @ pred_pose
            abs_trans, abs_rot = _pose_error(pred_rel, gt_rel)
            pose_summary_parts.append(f"{agent_id}:t={abs_trans:.1f}m,r={abs_rot:.1f}°")

    if not pred_pts_cv_list:
        raise RuntimeError("No valid predicted points found in model outputs.")

    pred_pts = _points_opencv_to_carla(np.concatenate(pred_pts_cv_list, axis=0))
    gtpose_pts = _points_opencv_to_carla(np.concatenate(gtpose_pts_cv_list, axis=0))
    pred_rgb = np.concatenate(pred_rgb_list, axis=0) if pred_rgb_list else None
    gtpose_rgb = np.concatenate(gtpose_rgb_list, axis=0) if gtpose_rgb_list else None
    pred_agent_colors = np.concatenate(pred_agent_color_list, axis=0)
    gtpose_agent_colors = np.concatenate(gtpose_agent_color_list, axis=0)

    pred_colors = _build_color_mode(
        rgb_points=pred_rgb,
        agent_point_colors=pred_agent_colors,
        points=pred_pts,
        mode=args.pred_color_mode,
        solid_rgb=(0.0, 0.447, 0.741),
    )
    gtpose_colors = _build_color_mode(
        rgb_points=gtpose_rgb,
        agent_point_colors=gtpose_agent_colors,
        points=gtpose_pts,
        mode=args.pred_color_mode,
        solid_rgb=(0.0, 0.62, 0.45),
    )

    gt_points = _filter_points_only(gt_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
    pred_pts, pred_colors = _filter_points_and_colors(pred_pts, pred_colors, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
    gtpose_pts, gtpose_colors = _filter_points_and_colors(gtpose_pts, gtpose_colors, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)

    pred_pts, pred_colors = _downsample_pair(pred_pts, pred_colors, int(args.max_points), rng)
    gtpose_pts, gtpose_colors = _downsample_pair(gtpose_pts, gtpose_colors, int(args.max_points), rng)
    gt_points, _ = _downsample_pair(gt_points, None, int(args.max_points), rng)

    pred_marker = dict(size=1.2)
    gtpose_marker = dict(size=1.2)
    if pred_colors is not None:
        pred_marker["color"] = _rgb_strings(pred_colors)
    else:
        pred_marker["color"] = "rgba(0,114,178,0.55)"
    if gtpose_colors is not None:
        gtpose_marker["color"] = _rgb_strings(gtpose_colors)
    else:
        gtpose_marker["color"] = "rgba(0,158,115,0.55)"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "PredPose (model pose) vs GT",
            "GTPose (GT camera pose) vs GT",
        ),
    )

    for col in (1, 2):
        fig.add_trace(
            go.Scatter3d(
                x=gt_points[:, 0],
                y=gt_points[:, 1],
                z=gt_points[:, 2],
                mode="markers",
                name=f"GT LiDAR ({args.gt_lidar_mode})",
                marker=dict(size=1.0, color="rgba(140,140,140,0.26)"),
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
        _add_boxes_3d(
            fig,
            gt_corners,
            row=1,
            col=col,
            color="rgba(0,0,0,0.9)",
            name="GT boxes",
            showlegend=(col == 1),
        )

    fig.add_trace(
        go.Scatter3d(
            x=pred_pts[:, 0],
            y=pred_pts[:, 1],
            z=pred_pts[:, 2],
            mode="markers",
            name=f"PredPose ({args.pred_color_mode})",
            marker=pred_marker,
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=gtpose_pts[:, 0],
            y=gtpose_pts[:, 1],
            z=gtpose_pts[:, 2],
            mode="markers",
            name=f"GTPose ({args.pred_color_mode})",
            marker=gtpose_marker,
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    scene_common = dict(
        xaxis=dict(title="X", range=[-20, 80], showbackground=False),
        yaxis=dict(title="Y", range=[-60, 60], showbackground=False),
        zaxis=dict(title="Z", range=[-3, 5], showbackground=False),
        aspectratio=dict(x=1.2, y=1.4, z=0.35),
        camera=dict(eye=dict(x=1.55, y=-1.65, z=0.8)),
        aspectmode="manual",
    )
    fig.update_layout(
        title=(
            f"OPV2V coop point cloud | seq={args.sequence}/{args.frame} | main={args.main_agent} | "
            f"agents={','.join(coop_agents)} | GT={args.gt_lidar_mode} | color={args.pred_color_mode}<br>"
            f"pred_pts={pred_pts.shape[0]} | gtpose_pts={gtpose_pts.shape[0]} | gt_pts={gt_points.shape[0]}"
            + (f" | pose_err_rel {'; '.join(pose_summary_parts)}" if pose_summary_parts else "")
        ),
        scene=scene_common,
        scene2=scene_common,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_path = args.out_html if args.out_html.is_absolute() else (REPO_ROOT / args.out_html).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=args.include_plotlyjs, full_html=True)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
