#!/usr/bin/env python3
"""
Visualize VGGT (fine-tuned) point cloud vs OPV2V GT LiDAR + GT vehicle boxes (offline HTML).

This script is meant for sanity-checking scale + coordinate conventions:
  - OPV2V LiDAR/boxes are in CARLA/UE coords (X forward, Y right, Z up).
  - MapAnything/VGGT geometry tensors are in OpenCV coords (X right, Y down, Z forward).

We run VGGT, take its predicted *camera-frame* pointmaps (`pts3d_cam`), place them into the
main-agent ego frame using *GT camera poses*, then convert points back to CARLA and overlay
with the GT LiDAR + boxes.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_vggt_pred_vs_gt_boxes_html.py \
    --checkpoint /abs/path/to/checkpoint-best.pth \
    --split validate --index 0 --num_views 8 \
    --out_html eval_runs/vggt_pred_viz/validate_idx0.html
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from mapanything.datasets.opv2v import OPV2VCoopDataset
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local

REPO_ROOT = Path(__file__).resolve().parents[1]

# Carla/UE (X forward, Y right, Z up) -> OpenCV (X right, Y down, Z forward)
# For row-vector points: p_carla = p_cv @ CARLA_TO_CV
_CARLA_TO_CV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _take_first(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


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


def _extract_scene_ids(views) -> tuple[str, str, str]:
    label = str(_take_first(views[0].get("label", "")) or "")
    instance = str(_take_first(views[0].get("instance", "")) or "")
    label_parts = Path(label).parts
    if len(label_parts) < 2:
        raise ValueError(f"Unexpected label format: {label!r}")
    sequence, main_agent = label_parts[-2], label_parts[-1]
    inst_parts = Path(instance).parts
    if not inst_parts:
        raise ValueError(f"Unexpected instance format: {instance!r}")
    frame_id = inst_parts[0]
    return sequence, main_agent, frame_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Finetuned checkpoint (expects state_dict under key 'model', as produced by MapAnything training).",
    )
    parser.add_argument(
        "--enable_metric_scale_head",
        action="store_true",
        help="Enable VGGT's metric scale head when loading the checkpoint (needed for scale-head finetunes).",
    )
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None)
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--max_points_viz", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius_max", type=float, default=60.0, help="BEV crop radius (m)")
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--out_html", type=Path, default=Path("eval_runs/vggt_pred_viz/validate_idx0.html"))
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
            "model=vggt",
            "model.model_config.load_pretrained_weights=false",
            f"model.model_config.enable_metric_scale_head={'true' if args.enable_metric_scale_head else 'false'}",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)
    model.eval()

    dataset = OPV2VCoopDataset(
        split=args.split,
        ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
        depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
        camera_ids=(0, 1, 2, 3),
        num_views=int(args.num_views),
        variable_num_views=False,
        resolution=(448, 252),
        principal_point_centered=True,
        transform="imgnorm",
        data_norm_type="identity",
        seed=777,
        include_vehicle_boxes=True,
        max_num_boxes=128,
        bbox_range=120.0,
        min_agents=2,
        min_num_views=int(args.num_views),
        max_num_views=int(args.num_views),
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

    sequence, main_agent, frame_id = _extract_scene_ids(views)
    gt_pcd_path = (REPO_ROOT / "data" / "opv2v" / args.split / sequence / main_agent / f"{frame_id}.pcd").resolve()
    gt_points = _load_ascii_pcd_xyz(gt_pcd_path)

    gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
    gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
    gt_boxes = gt_boxes[gt_mask]
    gt_corners = _boxes_to_corners(gt_boxes)

    with torch.no_grad():
        preds = model(views)

    metric_scale = None
    if preds and "metric_scaling_factor" in preds[0]:
        metric_scale = float(preds[0]["metric_scaling_factor"][0, 0].detach().cpu().item())

    pred_points_cv = []
    for view, pred in zip(views, preds):
        pts_cam = pred["pts3d_cam"][0].reshape(-1, 3)
        valid = view["valid_mask"][0].reshape(-1).bool()
        pts_cam = pts_cam[valid].detach().cpu().numpy().astype(np.float32)

        pose = view["camera_pose"][0].detach().cpu().numpy().astype(np.float32)  # cam2world (OpenCV)
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        pts_world = pts_cam @ rot.T + trans[None]
        pred_points_cv.append(pts_world)

    pts_cv = np.concatenate(pred_points_cv, axis=0) if pred_points_cv else np.zeros((0, 3), dtype=np.float32)
    pts_carla = pts_cv @ _CARLA_TO_CV

    if args.z_min is not None:
        pts_carla = pts_carla[pts_carla[:, 2] >= float(args.z_min)]
        gt_points = gt_points[gt_points[:, 2] >= float(args.z_min)]
    if args.z_max is not None:
        pts_carla = pts_carla[pts_carla[:, 2] <= float(args.z_max)]
        gt_points = gt_points[gt_points[:, 2] <= float(args.z_max)]

    if args.radius_max is not None and args.radius_max > 0:
        r_pred = np.sqrt(pts_carla[:, 0] ** 2 + pts_carla[:, 1] ** 2)
        pts_carla = pts_carla[r_pred <= float(args.radius_max)]
        r_gt = np.sqrt(gt_points[:, 0] ** 2 + gt_points[:, 1] ** 2)
        gt_points = gt_points[r_gt <= float(args.radius_max)]

    pts_carla = _downsample(pts_carla, int(args.max_points_viz), rng)
    gt_points = _downsample(gt_points, int(args.max_points_viz), rng)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("BEV (+X forward, +Y right)", "3D (CARLA ego)"),
    )

    fig.add_trace(
        go.Scatter(
            x=gt_points[:, 0],
            y=gt_points[:, 1],
            mode="markers",
            name="GT LiDAR",
            legendgroup="gt",
            marker=dict(size=2, color=gt_points[:, 2], colorscale="Turbo", opacity=0.25, showscale=False),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pts_carla[:, 0],
            y=pts_carla[:, 1],
            mode="markers",
            name="VGGT pts",
            legendgroup="pred",
            marker=dict(size=2, color="rgba(0,114,178,0.55)"),
        ),
        row=1,
        col=1,
    )
    _add_boxes_bev(
        fig,
        gt_boxes,
        name="GT boxes",
        color="rgba(0,0,0,0.9)",
        row=1,
        col=1,
        showlegend=True,
        legendgroup="gt_boxes",
    )

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
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_carla[:, 0],
            y=pts_carla[:, 1],
            z=pts_carla[:, 2],
            mode="markers",
            name="VGGT pts",
            legendgroup="pred",
            showlegend=False,
            marker=dict(size=1, color="rgba(0,114,178,0.55)"),
        ),
        row=1,
        col=2,
    )
    _add_boxes_3d(
        fig,
        gt_corners,
        name="GT boxes",
        color="rgba(0,0,0,0.9)",
        row=1,
        col=2,
        showlegend=False,
        legendgroup="gt_boxes",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    title = (
        f"VGGT pred vs GT (split={args.split} idx={args.index} seq={sequence} main={main_agent} frame={frame_id})"
        f"<br>pred_pts={pts_carla.shape[0]} gt_pts={gt_points.shape[0]} boxes={gt_boxes.shape[0]}"
    )
    if metric_scale is not None:
        title += f" metric_scale={metric_scale:.4f}"
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_path = args.out_html
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="directory", full_html=True)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
