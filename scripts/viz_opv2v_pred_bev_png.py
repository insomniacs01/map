#!/usr/bin/env python3
"""
Create a quick BEV PNG overlay for MapAnything predictions on OPV2V coop cylindrical scenes.

It plots GT LiDAR + GT boxes together with the predicted point cloud:
  - left: points placed using the model's predicted pose (pred['pts3d'])
  - right: points placed using GT ego pose (GT pose applied to pred['pts3d_cam'])

The GT-pose panel helps decouple pose errors from depth/ray-direction errors.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_pred_bev_png.py \
    --checkpoint experiments/opv2v_cyl_geom_quick/checkpoint-last.pth \
    --split validate --sequence 2021_08_22_22_01_17 --frame 000318 --main_agent 1715 \
    --num_views 2 --mask_thresh 0.5 \
    --out_png eval_runs/demo_alignment/pred_bev.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_processing.opv2v_pose_utils import cords_to_pose, load_frame_metadata
from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
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
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if points.ndim == 1:
        points = points.reshape(1, 3)
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

    all_points = []
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


def _box_to_bev_poly(box: np.ndarray) -> np.ndarray:
    x, y, _z, l, w, _h, yaw = [float(v) for v in box]
    dx = l / 2.0
    dy = w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _plot_boxes(ax: plt.Axes, boxes: Iterable[np.ndarray], *, color: str, lw: float) -> None:
    for box in boxes:
        poly = _box_to_bev_poly(box)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=lw)


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


def _aggregate_pred_points(
    views: List[dict],
    preds: List[dict],
    *,
    mask_thresh: float | None,
) -> Tuple[np.ndarray, np.ndarray | None]:
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
    return pts_carla_pred, pts_carla_gtpose


def _find_scene_index(
    dataset: OPV2VCoopCylindricalDataset,
    *,
    sequence: str,
    frame: str,
    main_agent: str | None,
) -> int:
    for idx, scene in enumerate(dataset.scenes):
        if str(scene.get("sequence")) != str(sequence):
            continue
        if str(scene.get("frame")) != str(frame):
            continue
        if main_agent is not None and str(main_agent) not in list(scene.get("agents") or []):
            continue
        return int(idx)
    raise ValueError(f"Scene not found: split={dataset.split} sequence={sequence} frame={frame} main_agent={main_agent}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument(
        "--task",
        type=str,
        default="images_only",
        help="Model task config under `configs/model/task/` (e.g. images_only, posed_sfm, mvs).",
    )
    parser.add_argument("--mask_thresh", type=float, default=0.5)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=80_000)
    parser.add_argument("--radius_max", type=float, default=60.0)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument(
        "--gt_lidar_mode",
        type=str,
        default="main",
        choices=("main", "fused"),
        help="Which GT LiDAR to overlay: main-agent only or fused across selected agents (transformed into main frame).",
    )
    parser.add_argument("--out_png", type=Path, required=True)
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
            "model.encoder.uses_torch_hub=false",
            f"model/task={args.task}",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)
    model.eval()

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
        main_agent=args.main_agent,
        ensure_main_agent_first=True,
        max_agent_distance=float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        agent_selection_policy=str(args.agent_selection_policy),
    )

    scene_idx = _find_scene_index(
        dataset,
        sequence=str(args.sequence),
        frame=str(args.frame),
        main_agent=str(args.main_agent) if args.main_agent is not None else None,
    )

    subset = torch.utils.data.Subset(dataset, [int(scene_idx)])
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    batch_views = next(iter(loader))
    for view in batch_views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)

    main_agent = _scalar_str(batch_views[0].get("main_agent_id", args.main_agent or "unknown"))
    agent_ids = [_scalar_str(v.get("agent_id")) for v in batch_views if _scalar_str(v.get("agent_id"))]
    gt_points = _load_gt_lidar_points(
        lidar_root=REPO_ROOT / "data" / "opv2v",
        split=str(args.split),
        sequence=str(args.sequence),
        frame=str(args.frame),
        main_agent=str(main_agent),
        agent_ids=agent_ids,
        mode=str(args.gt_lidar_mode),
    )

    gt_boxes = batch_views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
    gt_mask = batch_views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
    gt_boxes = gt_boxes[gt_mask]

    with torch.no_grad():
        preds = model(batch_views)

    pred_points, gtpose_points = _aggregate_pred_points(
        batch_views,
        preds,
        mask_thresh=float(args.mask_thresh) if args.mask_thresh is not None else None,
    )

    gt_points = _filter_points(gt_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
    pred_points = _filter_points(pred_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
    if gtpose_points is not None:
        gtpose_points = _filter_points(gtpose_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)

    gt_points = _downsample(gt_points, int(args.max_points), rng)
    pred_points = _downsample(pred_points, int(args.max_points), rng)
    if gtpose_points is not None:
        gtpose_points = _downsample(gtpose_points, int(args.max_points), rng)

    has_gtpose = gtpose_points is not None and gtpose_points.size > 0
    cols = 2 if has_gtpose else 1
    fig, axes = plt.subplots(1, cols, figsize=(8 * cols, 8), dpi=150)
    if cols == 1:
        axes = [axes]

    def _plot(ax: plt.Axes, *, pred: np.ndarray, title: str) -> None:
        gt_label = "GT LiDAR (main)" if str(args.gt_lidar_mode) == "main" else "GT LiDAR (fused)"
        ax.scatter(gt_points[:, 0], gt_points[:, 1], s=0.12, c="0.5", alpha=0.18, rasterized=True, label=gt_label)
        ax.scatter(pred[:, 0], pred[:, 1], s=0.15, c="tab:blue", alpha=0.30, rasterized=True, label="Pred pts")
        _plot_boxes(ax, gt_boxes, color="black", lw=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("+X forward (m)")
        ax.set_ylabel("+Y right (m)")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    ckpt_name = ckpt_path.name
    _plot(
        axes[0],
        pred=pred_points,
        title=f"Pred pose | {ckpt_name} | seq={args.sequence} agent={main_agent} frame={args.frame} mask>{args.mask_thresh}",
    )
    if has_gtpose:
        _plot(
            axes[1],
            pred=gtpose_points,
            title=f"GT pose | {ckpt_name} | seq={args.sequence} agent={main_agent} frame={args.frame} mask>{args.mask_thresh}",
        )

    out_png = args.out_png.expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
