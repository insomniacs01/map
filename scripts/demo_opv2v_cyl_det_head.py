#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.train.losses import BEVCenternetDetLoss
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local


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


def _scalar_str(value) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return str(value[0])
    return str(value)


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
    edges = [
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
    return edges


def _add_boxes(
    fig: go.Figure,
    corners: np.ndarray,
    *,
    name: str,
    color: str,
    legendgroup: str | None = None,
    showlegend: bool = True,
    row: int | None = None,
    col: int | None = None,
) -> None:
    if corners.size == 0:
        return

    xs = []
    ys = []
    zs = []
    for corner in corners:
        for a, b in _iter_box_edges():
            xs.extend([float(corner[a, 0]), float(corner[b, 0]), None])
            ys.extend([float(corner[a, 1]), float(corner[b, 1]), None])
            zs.extend([float(corner[a, 2]), float(corner[b, 2]), None])

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        line=dict(color=color, width=4),
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _decode_boxes(
    heatmap: torch.Tensor,
    reg: torch.Tensor,
    *,
    x_min: float,
    y_min: float,
    voxel_size: float,
    score_thresh: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if heatmap.ndim != 4 or reg.ndim != 4:
        raise ValueError("heatmap/reg must be batched tensors.")
    if heatmap.shape[0] != 1:
        raise ValueError("demo only supports batch_size=1.")

    _, _, height, width = heatmap.shape
    scores, inds = torch.topk(heatmap.view(1, -1), k=min(int(topk), height * width), dim=1)
    scores = scores[0]
    inds = inds[0]
    xs = (inds % width).to(torch.long)
    ys = (inds // width).to(torch.long)

    reg_sel = reg[0, :, ys, xs].transpose(0, 1)  # (K, 8)
    scores_np = scores.detach().cpu().numpy()

    keep = scores_np >= float(score_thresh)
    if not np.any(keep):
        return np.zeros((0, 7), dtype=np.float32), scores_np[:0].astype(np.float32)

    reg_sel = reg_sel[keep]
    xs = xs[keep]
    ys = ys[keep]
    scores_np = scores_np[keep].astype(np.float32)

    offset = reg_sel[:, 0:2]
    log_lw = reg_sel[:, 2:4]
    z = reg_sel[:, 4:5]
    log_h = reg_sel[:, 5:6]
    sin_yaw = reg_sel[:, 6:7]
    cos_yaw = reg_sel[:, 7:8]

    center_x = (xs.float().unsqueeze(-1) + offset[:, 0:1]) * voxel_size + x_min
    center_y = (ys.float().unsqueeze(-1) + offset[:, 1:2]) * voxel_size + y_min
    length = torch.exp(log_lw[:, 0:1])
    width_ = torch.exp(log_lw[:, 1:2])
    height_ = torch.exp(log_h)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    boxes = torch.cat([center_x, center_y, z, length, width_, height_, yaw], dim=1)
    return boxes.detach().cpu().numpy().astype(np.float32), scores_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/facebook_map-anything-v1.pth"),
        help="MapAnything checkpoint path (state_dict under key 'model').",
    )
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--train_iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max_points_viz", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_html", type=Path, default=Path("eval_runs/demo_det/val0_det.html"))
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = (repo_root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    local_cfg = {
        "path": str((repo_root / "configs" / "train.yaml").resolve()),
        "checkpoint_path": str(checkpoint_path),
        "config_overrides": [
            "machine=local_a800",
            "model=mapanything_det",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)
    if not hasattr(model, "det_head") or model.det_head is None:
        raise RuntimeError("Model was initialized without det_head enabled.")

    dataset = OPV2VCoopCylindricalDataset(
        split=args.split,
        ROOT=str((repo_root / "data" / "opv2v_images").resolve()),
        depth_root=str((repo_root / "data" / "opv2v_depth").resolve()),
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
        ensure_main_agent_first=True,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    views = None
    for i, batch in enumerate(loader):
        if i == args.index:
            views = batch
            break
    if views is None:
        raise IndexError(f"Index {args.index} out of range for split={args.split}.")

    for view in views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(views)

    pts_cv = torch.cat([p["pts3d"].reshape(1, -1, 3) for p in preds], dim=1)
    weights = torch.cat(
        [
            p["non_ambiguous_mask"].reshape(1, -1).float()
            if "non_ambiguous_mask" in p
            else torch.ones((1, p["pts3d"].numel() // 3), device=device)
            for p in preds
        ],
        dim=1,
    )
    pts_carla = pts_cv @ model._CARLA_TO_CV.to(pts_cv)

    gt_boxes = views[0]["vehicle_boxes"].float()
    gt_mask = views[0]["vehicle_boxes_mask"]
    det_loss = BEVCenternetDetLoss(x_range=(0.0, 120.0), y_range=(-50.0, 50.0), voxel_size=0.5).to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.det_head.parameters():
        param.requires_grad = True
    model.det_head.train()

    optim = torch.optim.AdamW(model.det_head.parameters(), lr=args.lr)
    fake_batch = [{"vehicle_boxes": gt_boxes, "vehicle_boxes_mask": gt_mask}]

    for step in range(int(args.train_iters)):
        optim.zero_grad(set_to_none=True)
        det_out = model.det_head(pts_carla, point_weights=weights)
        fake_preds = [{"bev_det": det_out, "pts3d": pts_cv}]
        loss, _ = det_loss(fake_batch, fake_preds)
        loss.backward()
        optim.step()
        if step % 50 == 0 or step == int(args.train_iters) - 1:
            print(f"[train] step={step} loss={float(loss.detach()):.4f}")

    model.det_head.eval()
    with torch.no_grad():
        det_out = model.det_head(pts_carla, point_weights=weights)

    pred_boxes, pred_scores = _decode_boxes(
        det_out["heatmap"],
        det_out["reg"],
        x_min=det_loss.x_min,
        y_min=det_loss.y_min,
        voxel_size=det_loss.voxel_size,
        score_thresh=args.score_thresh,
        topk=args.topk,
    )

    gt_mask_np = gt_mask[0].detach().cpu().numpy().astype(bool)
    gt_boxes_np = gt_boxes[0].detach().cpu().numpy()[gt_mask_np]
    gt_corners = _boxes_to_corners(gt_boxes_np)
    pred_corners = _boxes_to_corners(pred_boxes)

    pts_np = pts_carla[0].detach().cpu().numpy()
    weights_np = weights[0].detach().cpu().numpy()
    pts_np = pts_np[weights_np > 0.5]
    pts_np = _downsample(pts_np, args.max_points_viz, rng)

    scene_info = dataset.scenes[int(args.index)]
    sequence = str(scene_info["sequence"])
    frame_id = str(scene_info["frame"])
    main_agent = _scalar_str(views[0].get("main_agent_id", "unknown"))
    gt_pcd_path = (repo_root / "data" / "opv2v" / args.split / sequence / main_agent / f"{frame_id}.pcd").resolve()
    gt_points = _load_ascii_pcd_xyz(gt_pcd_path)
    gt_points = _downsample(gt_points, args.max_points_viz, rng)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("GT LiDAR + GT boxes", "MapAnything pts + GT/Pred boxes"),
    )

    fig.add_trace(
        go.Scatter3d(
            x=gt_points[:, 0],
            y=gt_points[:, 1],
            z=gt_points[:, 2],
            mode="markers",
            name="GT LiDAR (CARLA)",
            legendgroup="gt_lidar",
            marker=dict(size=1, color="rgba(140,140,140,0.35)"),
        ),
        row=1,
        col=1,
    )
    _add_boxes(
        fig,
        gt_corners,
        name="GT boxes",
        color="rgba(0,0,0,0.9)",
        legendgroup="gt_boxes",
        showlegend=True,
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=gt_points[:, 0],
            y=gt_points[:, 1],
            z=gt_points[:, 2],
            mode="markers",
            name="GT LiDAR (CARLA)",
            legendgroup="gt_lidar",
            showlegend=False,
            marker=dict(size=1, color="rgba(140,140,140,0.20)"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_np[:, 0],
            y=pts_np[:, 1],
            z=pts_np[:, 2],
            mode="markers",
            name="MapAnything pts (CARLA)",
            legendgroup="pred_pts",
            marker=dict(size=1, color="rgba(0,114,178,0.55)"),
        ),
        row=1,
        col=2,
    )
    _add_boxes(
        fig,
        gt_corners,
        name="GT boxes",
        color="rgba(0,0,0,0.9)",
        legendgroup="gt_boxes",
        showlegend=False,
        row=1,
        col=2,
    )
    _add_boxes(
        fig,
        pred_corners,
        name="Pred boxes",
        color="rgba(213,94,0,0.9)",
        legendgroup="pred_boxes",
        showlegend=True,
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"OPV2V Cyl Coop Detection Demo (split={args.split}, idx={args.index}, pred={len(pred_boxes)}, gt={len(gt_boxes_np)})",
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_path = args.out_html
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="directory", full_html=True)
    print(f"[done] wrote {out_path}")
    if pred_scores.size:
        print(f"[pred] top scores: {np.sort(pred_scores)[-5:][::-1]}")


if __name__ == "__main__":
    main()
