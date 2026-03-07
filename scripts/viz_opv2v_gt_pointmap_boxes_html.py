#!/usr/bin/env python3
"""
Visualize OPV2V GT pointmap (from depth) vs GT vehicle boxes (offline HTML).

This is a coordinate-system sanity check for training: our VehicleBoxPointAlignmentLoss
labels "vehicle pixels" by checking whether the *GT pointmap* lies inside GT boxes,
so we must ensure that GT pointmap and GT boxes are aligned in ego CARLA/UE coords.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_gt_pointmap_boxes_html.py \
    --split validate --sequence 2021_09_11_00_33_16 --main_agent 1007 --frame 000203 \
    --num_views 4 --max_points_viz 120000 \
    --out_html eval_runs/gt_pointmap_baseline/2021_09_11_00_33_16_1007_000203.html
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset


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


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None)
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--max_points_viz", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius_max", type=float, default=60.0)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--out_html", type=Path, default=Path("eval_runs/gt_pointmap_baseline/validate_idx0.html"))
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))

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

    scene_info = dataset.scenes[int(args.index)]
    sequence = str(scene_info["sequence"])
    frame_id = str(scene_info["frame"])
    main_agent = _scalar_str(views[0].get("main_agent_id", "unknown"))

    gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
    gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
    gt_boxes = gt_boxes[gt_mask]
    gt_corners = _boxes_to_corners(gt_boxes)

    pts_list = []
    for view_idx, view in enumerate(views):
        if "pts3d" not in view:
            continue
        pts_cv = view["pts3d"][0].detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        if "valid_mask" in view:
            valid = view["valid_mask"][0].detach().cpu().numpy().astype(bool).reshape(-1)
            pts_cv = pts_cv[valid]
        pts_list.append(pts_cv)
    if pts_list:
        pts_cv_all = np.concatenate(pts_list, axis=0)
    else:
        pts_cv_all = np.zeros((0, 3), dtype=np.float32)

    pts_carla = pts_cv_all @ _CARLA_TO_CV

    if args.z_min is not None:
        pts_carla = pts_carla[pts_carla[:, 2] >= float(args.z_min)]
    if args.z_max is not None:
        pts_carla = pts_carla[pts_carla[:, 2] <= float(args.z_max)]

    if args.radius_max is not None and args.radius_max > 0:
        r = np.sqrt(pts_carla[:, 0] ** 2 + pts_carla[:, 1] ** 2)
        pts_carla = pts_carla[r <= float(args.radius_max)]

    pts_carla = _downsample(pts_carla, int(args.max_points_viz), rng)

    gt_counts = [_count_points_in_box(pts_carla, box) for box in gt_boxes]
    if gt_counts:
        summary = f"GT pointmap pts-in-box min/med/max={min(gt_counts)}/{int(np.median(gt_counts))}/{max(gt_counts)}"
    else:
        summary = "GT pointmap pts-in-box n/a"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("BEV (+X forward, +Y right)", "3D (CARLA ego)"),
    )

    fig.add_trace(
        go.Scatter(
            x=pts_carla[:, 0],
            y=pts_carla[:, 1],
            mode="markers",
            name="GT pointmap",
            legendgroup="gt_pm",
            marker=dict(size=2, color=pts_carla[:, 2], colorscale="Turbo", opacity=0.25, showscale=False),
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
            x=pts_carla[:, 0],
            y=pts_carla[:, 1],
            z=pts_carla[:, 2],
            mode="markers",
            name="GT pointmap",
            legendgroup="gt_pm",
            showlegend=False,
            marker=dict(size=1, color="rgba(140,140,140,0.20)"),
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
    fig.update_layout(
        title=(
            f"GT pointmap vs GT boxes: split={args.split} idx={args.index} seq={sequence} agent={main_agent} frame={frame_id} "
            f"(pts={pts_carla.shape[0]}, boxes={gt_boxes.shape[0]})<br>{summary}"
        ),
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

