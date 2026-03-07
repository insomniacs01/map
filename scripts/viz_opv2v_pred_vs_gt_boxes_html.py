#!/usr/bin/env python3
"""
Visualize MapAnything predicted point cloud vs OPV2V GT LiDAR + GT vehicle boxes (offline HTML).

This is intended for training-time sanity checks: as finetuning progresses, predicted vehicle points should line up
with GT boxes similarly to the GT baseline pages produced by `scripts/viz_opv2v_gt_pcd_boxes_html.py`.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_pred_vs_gt_boxes_html.py \
    --checkpoint checkpoints/facebook_map-anything-v1.pth \
    --split validate --index 0 --num_views 4 \
    --out_html eval_runs/pred_viz/validate_idx0.html
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from plotly.subplots import make_subplots

from mapanything.datasets.opv2v import OPV2VCoopDataset
from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local
from mapanything.utils.geometry import quaternion_to_rotation_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from batch_eval import DetMetricConfig, decode_bev_centernet  # type: ignore


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
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points


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
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def _add_boxes_3d(
    fig: go.Figure,
    corners: np.ndarray,
    *,
    name: str,
    color: str,
    row: int,
    col: int,
    showlegend: bool,
    legendgroup: str,
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
            legendgroup=legendgroup,
            showlegend=showlegend,
            line=dict(color=color, width=4),
        ),
        row=row,
        col=col,
    )


def _add_boxes_bev(
    fig: go.Figure,
    boxes: np.ndarray,
    *,
    name: str,
    color: str,
    row: int,
    col: int,
    showlegend: bool,
    legendgroup: str,
) -> None:
    if boxes.size == 0:
        return
    xs: List[float | None] = []
    ys: List[float | None] = []
    for box in boxes:
        x, y, _z, l, w, _h, yaw = [float(v) for v in box]
        dx = l / 2.0
        dy = w / 2.0
        local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]], dtype=np.float32)
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        poly = local @ rot.T + np.array([x, y], dtype=np.float32)
        xs.extend([float(v) for v in poly[:, 0]] + [None])
        ys.extend([float(v) for v in poly[:, 1]] + [None])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            line=dict(color=color, width=2),
        ),
        row=row,
        col=col,
    )


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


def _load_det_cfg(det_head_cfg: str | None) -> dict:
    if not det_head_cfg:
        return {}
    cfg_path = REPO_ROOT / "configs" / "model" / "det_head" / f"{det_head_cfg}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _scalar_str(value) -> str:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return ""
        return str(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return str(value[0])
    return str(value)


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


def _pose_matrix_from_quat_trans(quat_xyzw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Build (B,4,4) cam2world pose from quaternion+translation (OpenCV convention)."""
    rot = quaternion_to_rotation_matrix(quat_xyzw)
    mat = torch.eye(4, device=rot.device, dtype=rot.dtype).unsqueeze(0).repeat(rot.shape[0], 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cyl",
        choices=("cyl", "pinhole"),
        help="Dataset type: cylindrical panoramas (cyl) or pinhole (pinhole).",
    )
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--index", type=int, default=0, help="Cooperative scene index (optional if using --sequence/--frame)")
    parser.add_argument("--sequence", type=str, default=None, help="Optional: pick scene by sequence name")
    parser.add_argument("--frame", type=str, default=None, help="Optional: pick scene by frame id (e.g. 000585)")
    parser.add_argument("--main_agent", type=str, default=None, help="Optional: force main agent id")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(448, 252),
        help="(pinhole) Resolution (width height) for OPV2VCoopDataset.",
    )
    parser.add_argument(
        "--pair_agents",
        action="store_true",
        help="(pinhole) Use deterministic 2-agent/8-view layout (pair_agents=True).",
    )
    parser.add_argument(
        "--pair_agent_policy",
        type=str,
        default="nearest",
        choices=("nearest", "random"),
        help="(pinhole) Policy for selecting the paired agent when pair_agents is enabled.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="images_only",
        help="Model task config under `configs/model/task/` (e.g. images_only, posed_sfm, mvs).",
    )
    parser.add_argument(
        "--det_head_cfg",
        type=str,
        default=None,
        help="Optional: det head config name under configs/model/det_head (e.g. bev_centernet_wide_v7_gate_noconf).",
    )
    parser.add_argument("--det_score_thresh", type=float, default=0.05)
    parser.add_argument("--det_nms_iou", type=float, default=0.1)
    parser.add_argument("--det_max_dets", type=int, default=100)
    parser.add_argument("--det_min_density", type=float, default=None)
    parser.add_argument("--det_min_high_ratio", type=float, default=None)
    parser.add_argument("--det_min_var_z", type=float, default=None)
    parser.add_argument("--max_points_viz", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score_thresh", type=float, default=0.3, help="(unused; reserved)")
    parser.add_argument(
        "--mask_thresh",
        type=float,
        default=0.5,
        help="Filter predicted points by non_ambiguous_mask > thresh (set <0 to disable filtering).",
    )
    parser.add_argument(
        "--max_agent_distance",
        type=float,
        default=None,
        help="Optional: only sample cooperative agents whose GT ego distance to main is <= this XY radius (meters).",
    )
    parser.add_argument(
        "--agent_selection_policy",
        type=str,
        default="random",
        choices=("random", "nearest"),
        help="How to select agents when more than num_views are available.",
    )
    parser.add_argument("--out_html", type=Path, default=Path("eval_runs/pred_viz/validate_idx0.html"))
    parser.add_argument("--out_png", type=Path, default=None, help="Optional: save static PNG (requires kaleido).")
    parser.add_argument("--png_width", type=int, default=1600)
    parser.add_argument("--png_height", type=int, default=900)
    parser.add_argument("--radius_max", type=float, default=60.0, help="BEV crop radius (m)")
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
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
            "model=mapanything_det",
            "model.encoder.uses_torch_hub=false",
            f"model/task={args.task}",
            f"model/det_head={args.det_head_cfg}" if args.det_head_cfg else None,
        ],
        "strict": False,
    }
    local_cfg["config_overrides"] = [v for v in local_cfg["config_overrides"] if v]
    model = initialize_mapanything_local(local_cfg, device)
    model.eval()

    if args.dataset == "cyl":
        dataset = OPV2VCoopCylindricalDataset(
            split=args.split,
            ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
            depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
            camera_ids=(0, 1, 2, 3),
            num_views=args.num_views,
            variable_num_views=False,
            resolution=(1008, 252),
            transform="imgnorm",
            data_norm_type="dinov2",
            seed=777,
            include_vehicle_boxes=True,
            max_num_boxes=128,
            bbox_range=120.0,
            min_agents=2,
            min_num_views=args.num_views,
            max_num_views=args.num_views,
            panorama_resolution=(1008, 252),
            panorama_vertical_fov_deg=90.0,
            panorama_elevation_center_deg=0.0,
            view_selection_margin=0.05,
            max_num_retries=0,
            main_agent=args.main_agent,
            ensure_main_agent_first=True,
            max_agent_distance=float(args.max_agent_distance) if args.max_agent_distance is not None else None,
            agent_selection_policy=str(args.agent_selection_policy),
        )
    else:
        dataset = OPV2VCoopDataset(
            split=args.split,
            ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
            depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
            camera_ids=(0, 1, 2, 3),
            num_views=args.num_views,
            variable_num_views=False,
            resolution=tuple(args.resolution),
            transform="imgnorm",
            data_norm_type="dinov2",
            seed=777,
            include_vehicle_boxes=True,
            max_num_boxes=128,
            bbox_range=120.0,
            min_agents=2,
            min_num_views=args.num_views,
            max_num_views=args.num_views,
            pair_agents=bool(args.pair_agents),
            pair_agent_policy=str(args.pair_agent_policy),
            main_agent=args.main_agent,
            main_agent_policy="first",
            max_num_retries=0,
        )

    if args.sequence is not None or args.frame is not None:
        if not (args.sequence and args.frame):
            raise ValueError("When using --sequence/--frame, both must be provided.")
        desired_seq = str(args.sequence)
        desired_frame = str(args.frame)
        found = None
        for idx, scene in enumerate(dataset.scenes):
            if str(scene.get("sequence")) != desired_seq:
                continue
            if str(scene.get("frame")) != desired_frame:
                continue
            if args.main_agent is not None and str(args.main_agent) not in list(scene.get("agents") or []):
                continue
            found = idx
            break
        if found is None:
            raise ValueError(f"Scene not found: split={args.split} sequence={desired_seq} frame={desired_frame}")
        args.index = int(found)

    if args.index < 0 or args.index >= len(dataset.scenes):
        raise IndexError(f"Index {args.index} out of range for split={args.split} (len={len(dataset.scenes)}).")

    subset = torch.utils.data.Subset(dataset, [int(args.index)])
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    views = next(iter(loader))

    for view in views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)

    scene_info = dataset.scenes[int(args.index)]
    sequence = str(scene_info["sequence"])
    frame_id = str(scene_info["frame"])
    if args.dataset == "cyl":
        main_agent = _scalar_str(views[0].get("main_agent_id", "unknown"))
    else:
        if args.main_agent is not None:
            main_agent = str(args.main_agent)
        else:
            agents = scene_info.get("agents") or []
            if agents:
                main_agent = dataset._choose_main_agent(list(agents))
            else:
                main_agent = "unknown"
    gt_pcd_path = (REPO_ROOT / "data" / "opv2v" / args.split / sequence / main_agent / f"{frame_id}.pcd").resolve()
    gt_points = _load_ascii_pcd_xyz(gt_pcd_path)

    gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
    gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
    gt_boxes = gt_boxes[gt_mask]
    gt_corners = _boxes_to_corners(gt_boxes)

    with torch.no_grad():
        preds = model(views)

    det_boxes = np.zeros((0, 7), dtype=np.float32)
    det_corners = np.zeros((0, 8, 3), dtype=np.float32)
    det_summary = ""
    if preds and isinstance(preds[0], dict) and preds[0].get("bev_det") is not None:
        det_cfg_raw = _load_det_cfg(args.det_head_cfg)
        if det_cfg_raw:
            x_range = det_cfg_raw.get("x_range", (-50.0, 120.0))
            y_range = det_cfg_raw.get("y_range", (-50.0, 50.0))
            voxel_size = det_cfg_raw.get("voxel_size", 0.5)
        else:
            x_range = (-50.0, 120.0)
            y_range = (-50.0, 50.0)
            voxel_size = 0.5
        det_cfg = DetMetricConfig(
            enabled=True,
            score_thresh=float(args.det_score_thresh),
            nms_iou=float(args.det_nms_iou),
            max_dets=int(args.det_max_dets),
            iou_thresh=0.5,
            min_density=float(args.det_min_density) if args.det_min_density is not None else None,
            min_high_ratio=float(args.det_min_high_ratio) if args.det_min_high_ratio is not None else None,
            min_var_z=float(args.det_min_var_z) if args.det_min_var_z is not None else None,
            bbox_range=120.0,
            max_num_boxes=128,
            x_range=(float(x_range[0]), float(x_range[1])),
            y_range=(float(y_range[0]), float(y_range[1])),
            voxel_size=float(voxel_size),
        )
        decoded = decode_bev_centernet(preds[0]["bev_det"], det_cfg)
        if decoded:
            det_boxes = np.stack(
                [item["box"] for item in decoded if isinstance(item.get("box"), np.ndarray)],
                axis=0,
            )
            if det_boxes.ndim == 2 and det_boxes.shape[1] == 7:
                det_corners = _boxes_to_corners(det_boxes)
            det_summary = f"Pred det boxes={det_boxes.shape[0]}"

    # Aggregate points across views in OpenCV (RDF) coordinates.
    # - `pred["pts3d"]` uses the model's predicted pose.
    # - `pred["pts3d_cam"]` is camera-frame, so we can place it using GT `view["camera_pose"]`
    #   to decouple depth quality from pose quality (this matches the detection-head GT-pose path).
    pts_cv_pred_list = []
    pts_cv_gtpose_list = []
    weights_list = []
    agent_ids: List[str] = []
    gt_poses_cv: List[np.ndarray | None] = []
    pred_poses_cv: List[np.ndarray | None] = []

    for view, pred in zip(views, preds):
        pts3d = pred.get("pts3d")
        pts3d_cam = pred.get("pts3d_cam")
        if pts3d is None or pts3d_cam is None:
            continue
        if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
            continue
        if pts3d_cam.ndim != 4 or pts3d_cam.shape[-1] != 3:
            continue

        pts_cv_pred_list.append(pts3d.reshape(1, -1, 3)[0])

        if "non_ambiguous_mask" in pred:
            weights_list.append(pred["non_ambiguous_mask"].reshape(1, -1).float()[0])
        else:
            weights_list.append(torch.ones((pts3d.numel() // 3,), device=device, dtype=torch.float32))

        agent_id = view.get("agent_id")
        if isinstance(agent_id, (list, tuple)) and agent_id:
            agent_ids.append(str(agent_id[0]))
        else:
            agent_ids.append(f"view{len(agent_ids)}")

        pose_gt = view.get("camera_pose")
        if pose_gt is not None and torch.is_tensor(pose_gt) and pose_gt.ndim == 3 and pose_gt.shape[-2:] == (4, 4):
            rot = pose_gt[:, :3, :3]
            trans = pose_gt[:, :3, 3]
            pts_gtpose = torch.einsum("bij,bhwj->bhwi", rot, pts3d_cam) + trans[:, None, None, :]
            pts_cv_gtpose_list.append(pts_gtpose.reshape(1, -1, 3)[0])
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

    if not pts_cv_pred_list:
        raise RuntimeError("No valid predicted points found in model outputs.")

    pts_cv_pred = torch.cat(pts_cv_pred_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    pts_cv_gtpose = torch.cat(pts_cv_gtpose_list, dim=0) if pts_cv_gtpose_list else None

    if args.mask_thresh is not None and float(args.mask_thresh) >= 0:
        keep = weights > float(args.mask_thresh)
    else:
        keep = torch.ones_like(weights, dtype=torch.bool)

    pts_cv_pred_np = pts_cv_pred[keep].detach().cpu().numpy().astype(np.float32)
    pts_carla_pred = pts_cv_pred_np @ _CARLA_TO_CV

    pts_carla_gtpose = None
    if pts_cv_gtpose is not None and pts_cv_gtpose.shape[0] == keep.shape[0]:
        pts_cv_gtpose_np = pts_cv_gtpose[keep].detach().cpu().numpy().astype(np.float32)
        pts_carla_gtpose = pts_cv_gtpose_np @ _CARLA_TO_CV

    if args.z_min is not None:
        pts_carla_pred = pts_carla_pred[pts_carla_pred[:, 2] >= float(args.z_min)]
        if pts_carla_gtpose is not None:
            pts_carla_gtpose = pts_carla_gtpose[pts_carla_gtpose[:, 2] >= float(args.z_min)]
        gt_points = gt_points[gt_points[:, 2] >= float(args.z_min)]
    if args.z_max is not None:
        pts_carla_pred = pts_carla_pred[pts_carla_pred[:, 2] <= float(args.z_max)]
        if pts_carla_gtpose is not None:
            pts_carla_gtpose = pts_carla_gtpose[pts_carla_gtpose[:, 2] <= float(args.z_max)]
        gt_points = gt_points[gt_points[:, 2] <= float(args.z_max)]

    if args.radius_max is not None and args.radius_max > 0:
        r_pred = np.sqrt(pts_carla_pred[:, 0] ** 2 + pts_carla_pred[:, 1] ** 2)
        r_gt = np.sqrt(gt_points[:, 0] ** 2 + gt_points[:, 1] ** 2)
        pts_carla_pred = pts_carla_pred[r_pred <= float(args.radius_max)]
        if pts_carla_gtpose is not None:
            r_gtpose = np.sqrt(pts_carla_gtpose[:, 0] ** 2 + pts_carla_gtpose[:, 1] ** 2)
            pts_carla_gtpose = pts_carla_gtpose[r_gtpose <= float(args.radius_max)]
        gt_points = gt_points[r_gt <= float(args.radius_max)]

    pts_carla_pred = _downsample(pts_carla_pred, int(args.max_points_viz), rng)
    if pts_carla_gtpose is not None:
        pts_carla_gtpose = _downsample(pts_carla_gtpose, int(args.max_points_viz), rng)
    gt_points = _downsample(gt_points, int(args.max_points_viz), rng)

    gt_counts = [_count_points_in_box(gt_points, box) for box in gt_boxes]
    pred_counts = [_count_points_in_box(pts_carla_pred, box) for box in gt_boxes]
    gtpose_counts = (
        [_count_points_in_box(pts_carla_gtpose, box) for box in gt_boxes] if pts_carla_gtpose is not None else []
    )
    if gt_counts:
        gt_summary = f"GT pts-in-box min/med/max={min(gt_counts)}/{int(np.median(gt_counts))}/{max(gt_counts)}"
    else:
        gt_summary = "GT pts-in-box n/a"
    if pred_counts:
        pred_summary = (
            f"PredPose pts-in-box min/med/max={min(pred_counts)}/{int(np.median(pred_counts))}/{max(pred_counts)}"
        )
    else:
        pred_summary = "PredPose pts-in-box n/a"
    if gtpose_counts:
        gtpose_summary = (
            f"GTPose pts-in-box min/med/max={min(gtpose_counts)}/{int(np.median(gtpose_counts))}/{max(gtpose_counts)}"
        )
    else:
        gtpose_summary = "GTPose pts-in-box n/a"

    pose_summary = ""
    if gt_poses_cv and pred_poses_cv and gt_poses_cv[0] is not None and pred_poses_cv[0] is not None:
        gt_ref_inv = np.linalg.inv(gt_poses_cv[0])
        pred_ref_inv = np.linalg.inv(pred_poses_cv[0])
        per_view = []
        for name, gt_pose, pred_pose in zip(agent_ids, gt_poses_cv, pred_poses_cv):
            if gt_pose is None or pred_pose is None:
                continue
            gt_rel = gt_ref_inv @ gt_pose
            pred_rel = pred_ref_inv @ pred_pose
            abs_trans, abs_rot = _pose_error(pred_rel, gt_rel)
            per_view.append(f"{name}:t={abs_trans:.1f}m,r={abs_rot:.1f}°")
        if per_view:
            pose_summary = " | pose_err_rel " + ", ".join(per_view)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("BEV (pred pose)", "BEV (GT pose)", "3D (pred pose)", "3D (GT pose)"),
    )

    def _add_gt(*, row: int, col: int, is_3d: bool) -> None:
        if is_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=gt_points[:, 0],
                    y=gt_points[:, 1],
                    z=gt_points[:, 2],
                    mode="markers",
                    name="GT LiDAR",
                    legendgroup="gt",
                    showlegend=False,
                    marker=dict(size=1, color="rgba(140,140,140,0.20)"),
                ),
                row=row,
                col=col,
            )
            _add_boxes_3d(
                fig,
                gt_corners,
                name="GT boxes",
                color="rgba(0,0,0,0.9)",
                row=row,
                col=col,
                showlegend=False,
                legendgroup="gt_boxes",
            )
        else:
            showlegend = row == 1 and col == 1
            fig.add_trace(
                go.Scatter(
                    x=gt_points[:, 0],
                    y=gt_points[:, 1],
                    mode="markers",
                    name="GT LiDAR",
                    legendgroup="gt",
                    showlegend=showlegend,
                    marker=dict(size=2, color=gt_points[:, 2], colorscale="Turbo", opacity=0.25, showscale=False),
                ),
                row=row,
                col=col,
            )
            _add_boxes_bev(
                fig,
                gt_boxes,
                name="GT boxes",
                color="rgba(0,0,0,0.9)",
                row=row,
                col=col,
                showlegend=showlegend,
                legendgroup="gt_boxes",
            )

    _add_gt(row=1, col=1, is_3d=False)
    _add_gt(row=1, col=2, is_3d=False)
    _add_gt(row=2, col=1, is_3d=True)
    _add_gt(row=2, col=2, is_3d=True)

    if det_boxes.size > 0:
        _add_boxes_bev(
            fig,
            det_boxes,
            name="Pred boxes",
            color="lime",
            row=1,
            col=1,
            showlegend=True,
            legendgroup="pred_boxes",
        )
        _add_boxes_bev(
            fig,
            det_boxes,
            name="Pred boxes",
            color="lime",
            row=1,
            col=2,
            showlegend=False,
            legendgroup="pred_boxes",
        )
        _add_boxes_3d(
            fig,
            det_corners,
            name="Pred boxes",
            color="lime",
            row=2,
            col=1,
            showlegend=False,
            legendgroup="pred_boxes",
        )
        _add_boxes_3d(
            fig,
            det_corners,
            name="Pred boxes",
            color="lime",
            row=2,
            col=2,
            showlegend=False,
            legendgroup="pred_boxes",
        )

    # Pred-pose points.
    fig.add_trace(
        go.Scatter(
            x=pts_carla_pred[:, 0],
            y=pts_carla_pred[:, 1],
            mode="markers",
            name="Pred pts (pred pose)",
            legendgroup="predpose",
            showlegend=True,
            marker=dict(size=2, color="rgba(0,114,178,0.55)"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_carla_pred[:, 0],
            y=pts_carla_pred[:, 1],
            z=pts_carla_pred[:, 2],
            mode="markers",
            name="Pred pts (pred pose)",
            legendgroup="predpose",
            showlegend=False,
            marker=dict(size=1, color="rgba(0,114,178,0.55)"),
        ),
        row=2,
        col=1,
    )

    # GT-pose-aligned points (depth sanity check).
    if pts_carla_gtpose is not None:
        fig.add_trace(
            go.Scatter(
                x=pts_carla_gtpose[:, 0],
                y=pts_carla_gtpose[:, 1],
                mode="markers",
                name="Pred pts (GT pose)",
                legendgroup="gtpose",
                showlegend=True,
                marker=dict(size=2, color="rgba(0,158,115,0.55)"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=pts_carla_gtpose[:, 0],
                y=pts_carla_gtpose[:, 1],
                z=pts_carla_gtpose[:, 2],
                mode="markers",
                name="Pred pts (GT pose)",
                legendgroup="gtpose",
                showlegend=False,
                marker=dict(size=1, color="rgba(0,158,115,0.55)"),
            ),
            row=2,
            col=2,
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        title=(
            f"Pred vs GT: split={args.split} idx={args.index} seq={sequence} agent={main_agent} frame={frame_id} "
            f"(pred_pts={pts_carla_pred.shape[0]}, gt_pts={gt_points.shape[0]}, boxes={gt_boxes.shape[0]}, "
            f"mask_thresh={args.mask_thresh})<br>"
            f"{gt_summary} | {pred_summary} | {gtpose_summary}"
            f"{' | ' + det_summary if det_summary else ''}{pose_summary}"
        ),
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_path = args.out_html
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="directory", full_html=True)
    print(f"[OK] wrote {out_path}")
    if args.out_png is not None:
        out_png = args.out_png
        if not out_png.is_absolute():
            out_png = (REPO_ROOT / out_png).resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(out_png), width=int(args.png_width), height=int(args.png_height), scale=1)
            print(f"[OK] wrote {out_png}")
        except Exception as exc:  # pragma: no cover - depends on kaleido
            print(f"[WARN] failed to write PNG (install kaleido?): {exc}")


if __name__ == "__main__":
    main()
