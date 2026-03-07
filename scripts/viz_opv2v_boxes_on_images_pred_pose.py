#!/usr/bin/env python3
"""
Project OPV2V GT 3D vehicle boxes onto camera images using:
  - GT camera poses (green), and/or
  - VGGT-predicted camera poses (magenta) aligned to GT view0.

Why alignment to GT view0?
VGGT predicts poses in an arbitrary "world" frame for the N-view set. To compare
against OPV2V ego coordinates and to visualize box projections, we anchor the
predicted poses by replacing view0 pose with the GT view0 pose:
  T_ego_cam_i(pred) = T_ego_cam0(GT) @ T_cam0_cam_i(pred)

This yields a predicted cam2ego for each view in the OPV2V ego frame, allowing
metric sanity checks and visual inspection of pose correctness.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_boxes_on_images_pred_pose.py \
    --checkpoint experiments/vggt/training/opv2v_coop_vggt_pose_metric/XXXX/checkpoint-best.pth \
    --split validate --index 0 --num_views 8 \
    --out_dir eval_runs/opv2v_boxproj_predpose/val_idx0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mapanything.datasets.opv2v import OPV2VCoopDataset, OPV2VDataset
from mapanything.utils.geometry import (
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local

REPO_ROOT = Path(__file__).resolve().parents[1]

# Carla/UE (X forward, Y right, Z up) -> OpenCV (X right, Y down, Z forward)
_CARLA_TO_CAMERA_CV = np.array(
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


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points


def _boxes_to_corners_carla(boxes: np.ndarray) -> np.ndarray:
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


def _carla_points_to_cv(points_carla: np.ndarray) -> np.ndarray:
    # Row-vector conversion: p_cv = p_carla @ S^T
    return points_carla @ _CARLA_TO_CAMERA_CV.T


def _ego_cv_to_cam_cv(points_ego_cv: np.ndarray, cam2ego_cv: np.ndarray) -> np.ndarray:
    # cam2ego: p_ego = p_cam @ R^T + t  ->  p_cam = (p_ego - t) @ R
    rot = cam2ego_cv[:3, :3]
    trans = cam2ego_cv[:3, 3]
    return (points_ego_cv - trans[None]) @ rot


def _project_pinhole(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    eps = 1e-8
    z_safe = np.where(z > eps, z, eps)
    u = (K[0, 0] * points_cam[:, 0]) / z_safe + K[0, 2]
    v = (K[1, 1] * points_cam[:, 1]) / z_safe + K[1, 2]
    return np.stack([u, v], axis=-1), z


def _tensor_img_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    if img_tensor.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img_tensor.shape)}")
    img = img_tensor.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img_u8)


def _overlay_points(
    img: Image.Image,
    uv: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    if uv.size == 0:
        return img
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    arr = np.asarray(img).copy()
    H, W = arr.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not mask.any():
        return img
    uu = u[mask].astype(np.int32)
    vv = v[mask].astype(np.int32)
    col = np.array(color, dtype=np.float32)[None]
    base = arr[vv, uu].astype(np.float32)
    arr[vv, uu] = (base * (1.0 - alpha) + col * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_box_wireframes(
    img: Image.Image,
    corners_uv: np.ndarray,
    corners_z: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw = ImageDraw.Draw(img)
    if corners_uv.size == 0:
        return
    for box_uv, box_z in zip(corners_uv, corners_z):
        for a, b in _iter_box_edges():
            if box_z[a] <= 0.1 or box_z[b] <= 0.1:
                continue
            xa, ya = float(box_uv[a, 0]), float(box_uv[a, 1])
            xb, yb = float(box_uv[b, 0]), float(box_uv[b, 1])
            draw.line([(xa, ya), (xb, yb)], fill=color, width=width)


def _draw_label(img: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([(0, 0), (img.size[0], 18)], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)


def _make_grid(images: List[Image.Image], *, cols: int, pad: int = 6, bg=(20, 20, 20)) -> Image.Image:
    if not images:
        raise ValueError("No images to grid")
    cols = max(1, int(cols))
    W, H = images[0].size
    for im in images:
        if im.size != (W, H):
            raise ValueError("All images must share the same size")
    rows = int(math.ceil(len(images) / cols))
    grid_w = cols * W + (cols + 1) * pad
    grid_h = rows * H + (rows + 1) * pad
    out = Image.new("RGB", (grid_w, grid_h), color=bg)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (W + pad)
        y0 = pad + r * (H + pad)
        out.paste(im, (x0, y0))
    return out


def _rotation_angle_deg_from_quats_xyzw(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dot)
    return ang * (180.0 / math.pi)


def _se3_from_quat_trans_xyzw(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"Expected (B,4) quats, got {tuple(q.shape)}")
    if t.ndim != 2 or t.shape[-1] != 3:
        raise ValueError(f"Expected (B,3) trans, got {tuple(t.shape)}")
    R = quaternion_to_rotation_matrix(q)  # (B,3,3)
    B = int(q.shape[0])
    T = torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enable_metric_scale_head", action="store_true")
    parser.add_argument("--mode", type=str, choices=("coop", "single"), default="coop")
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("eval_runs/opv2v_boxproj_predpose/val_idx0"))
    parser.add_argument("--draw_gt", action="store_true", default=True)
    parser.add_argument("--no_draw_gt", dest="draw_gt", action="store_false")
    parser.add_argument("--draw_pred", action="store_true", default=True)
    parser.add_argument("--no_draw_pred", dest="draw_pred", action="store_false")
    parser.add_argument("--overlay_lidar", action="store_true")
    parser.add_argument("--lidar_pose", type=str, choices=("gt", "pred"), default="gt")
    parser.add_argument("--max_lidar_points", type=int, default=15000)
    parser.add_argument("--box_line_width", type=int, default=2)
    args = parser.parse_args()

    ckpt_path = (REPO_ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if args.mode == "coop":
        num_views = int(args.num_views or 8)
        resolution = (448, 252)
        dataset = OPV2VCoopDataset(
            split=args.split,
            ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
            depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
            camera_ids=(0, 1, 2, 3),
            pair_agents=True,
            pair_agent_policy="nearest",
            num_views=num_views,
            variable_num_views=False,
            resolution=resolution,
            principal_point_centered=True,
            transform="imgnorm",
            data_norm_type="identity",
            seed=777,
            include_vehicle_boxes=True,
            max_num_boxes=128,
            bbox_range=120.0,
            min_agents=2,
            min_num_views=num_views,
            max_num_views=num_views,
            main_agent=None,
            main_agent_policy="first",
            max_num_retries=0,
        )
    else:
        num_views = int(args.num_views or 4)
        resolution = (518, 392)
        dataset = OPV2VDataset(
            split=args.split,
            ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
            depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
            camera_ids=(0, 1, 2, 3),
            deterministic_camera_order=True,
            num_views=num_views,
            variable_num_views=False,
            resolution=resolution,
            principal_point_centered=True,
            transform="imgnorm",
            data_norm_type="identity",
            seed=777,
            include_vehicle_boxes=True,
            max_num_boxes=128,
            bbox_range=120.0,
            max_num_retries=0,
        )

    if args.index < 0 or args.index >= len(dataset.scenes):
        raise IndexError(f"Index {args.index} out of range for split={args.split} (len={len(dataset.scenes)}).")

    subset = torch.utils.data.Subset(dataset, [int(args.index)])
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    views = next(iter(loader))

    # Move tensors to GPU.
    for view in views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device, non_blocking=True)

    preds = model(views)

    sequence, main_agent, frame_id = _extract_scene_ids(views)

    gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
    gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
    gt_boxes = gt_boxes[gt_mask]

    corners_carla = _boxes_to_corners_carla(gt_boxes)  # (M, 8, 3) in ego CARLA
    corners_ego_cv = _carla_points_to_cv(corners_carla.reshape(-1, 3)).reshape(corners_carla.shape)

    # Anchor predicted poses with GT view0 cam2ego.
    gt_cam0_pose = views[0]["camera_pose"][0]  # (4,4) cam2ego (OpenCV)
    pred_cam0_q = preds[0]["cam_quats"]
    pred_cam0_t = preds[0]["cam_trans"]

    gt_q0 = views[0]["camera_pose_quats"]
    gt_t0 = views[0]["camera_pose_trans"]

    pred_cam2ego: list[np.ndarray] = []
    per_view_errors: list[tuple[float, float]] = []
    for view_idx in range(len(views)):
        if view_idx == 0:
            pred_pose = gt_cam0_pose
            trans_err = 0.0
            rot_err = 0.0
        else:
            q_rel_pred, t_rel_pred = transform_pose_using_quats_and_trans_2_to_1(
                pred_cam0_q, pred_cam0_t, preds[view_idx]["cam_quats"], preds[view_idx]["cam_trans"]
            )
            q_rel_gt, t_rel_gt = transform_pose_using_quats_and_trans_2_to_1(
                gt_q0, gt_t0, views[view_idx]["camera_pose_quats"], views[view_idx]["camera_pose_trans"]
            )
            trans_err = float(torch.linalg.norm((t_rel_pred - t_rel_gt).float(), dim=-1)[0].detach().cpu())
            rot_err = float(_rotation_angle_deg_from_quats_xyzw(q_rel_pred.float(), q_rel_gt.float())[0].detach().cpu())

            T_c0_ci = _se3_from_quat_trans_xyzw(q_rel_pred, t_rel_pred)[0]  # (4,4)
            pred_pose = gt_cam0_pose @ T_c0_ci

        pred_cam2ego.append(pred_pose.detach().cpu().numpy().astype(np.float32))
        per_view_errors.append((trans_err, rot_err))

    lidar_points_carla = None
    if args.overlay_lidar:
        pcd_path = (REPO_ROOT / "data" / "opv2v" / args.split / sequence / main_agent / f"{frame_id}.pcd").resolve()
        lidar_points_carla = _load_ascii_pcd_xyz(pcd_path)
        if args.max_lidar_points > 0 and lidar_points_carla.shape[0] > args.max_lidar_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(lidar_points_carla.shape[0], size=int(args.max_lidar_points), replace=False)
            lidar_points_carla = lidar_points_carla[idx]
        lidar_points_carla = lidar_points_carla.astype(np.float32)

    rendered: List[Image.Image] = []
    for view_idx, view in enumerate(views):
        img_pil = _tensor_img_to_pil(view["img"][0])
        K = view["camera_intrinsics"][0].detach().cpu().numpy().astype(np.float32)

        # GT pose projection (green).
        if args.draw_gt:
            gt_pose = view["camera_pose"][0].detach().cpu().numpy().astype(np.float32)
            corners_cam = _ego_cv_to_cam_cv(corners_ego_cv.reshape(-1, 3), gt_pose).reshape(corners_ego_cv.shape)
            corners_uv, corners_z = _project_pinhole(corners_cam.reshape(-1, 3), K)
            corners_uv = corners_uv.reshape(corners_cam.shape[0], 8, 2)
            corners_z = corners_z.reshape(corners_cam.shape[0], 8)
            _draw_box_wireframes(img_pil, corners_uv, corners_z, color=(0, 255, 0), width=int(args.box_line_width))

        # Pred pose projection (magenta).
        if args.draw_pred:
            pred_pose = pred_cam2ego[view_idx]
            corners_cam = _ego_cv_to_cam_cv(corners_ego_cv.reshape(-1, 3), pred_pose).reshape(corners_ego_cv.shape)
            corners_uv, corners_z = _project_pinhole(corners_cam.reshape(-1, 3), K)
            corners_uv = corners_uv.reshape(corners_cam.shape[0], 8, 2)
            corners_z = corners_z.reshape(corners_cam.shape[0], 8)
            _draw_box_wireframes(img_pil, corners_uv, corners_z, color=(255, 0, 255), width=int(args.box_line_width))

        if lidar_points_carla is not None and lidar_points_carla.size > 0:
            lidar_ego_cv = _carla_points_to_cv(lidar_points_carla)
            if args.lidar_pose == "pred":
                pose = pred_cam2ego[view_idx]
            else:
                pose = view["camera_pose"][0].detach().cpu().numpy().astype(np.float32)
            lidar_cam = _ego_cv_to_cam_cv(lidar_ego_cv, pose)
            uv, z = _project_pinhole(lidar_cam, K)
            uv = uv[z > 0.1]
            img_pil = _overlay_points(img_pil, uv, color=(255, 0, 0), alpha=0.55)

        inst = str(_take_first(view.get("instance", "")) or f"view{view_idx}")
        t_err, r_err = per_view_errors[view_idx]
        _draw_label(img_pil, f"{sequence}/{main_agent} {frame_id} | {inst} | t={t_err:.2f}m r={r_err:.2f}deg")
        rendered.append(img_pil)

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_view_paths = []
    for idx, im in enumerate(rendered):
        path = out_dir / f"view{idx:02d}.png"
        im.save(path)
        per_view_paths.append(path)

    grid = _make_grid(rendered, cols=min(4, len(rendered)))
    grid_path = out_dir / "grid.png"
    grid.save(grid_path)

    html_path = out_dir / "index.html"
    rel_grid = grid_path.name
    rows = "\n".join(f"<div><img src='{p.name}' style='max-width:100%'></div>" for p in per_view_paths)
    html_path.write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>OPV2V box projection (pred pose)</title></head>
<body style="background:#111;color:#eee;font-family:monospace">
<h3>OPV2V box projection (pred pose anchored to GT view0)</h3>
<p>split={args.split} idx={args.index} seq={sequence} main={main_agent} frame={frame_id}</p>
<ul>
  <li>Green: GT cam poses</li>
  <li>Magenta: VGGT predicted poses (aligned to GT view0)</li>
  <li>Red: LiDAR points (optional, projected with pose={args.lidar_pose})</li>
</ul>
<div><img src="{rel_grid}" style="max-width:100%"></div>
<hr/>
{rows}
</body>
</html>
""",
        encoding="utf-8",
    )

    print(f"[OK] wrote {grid_path}")
    print(f"[OK] wrote {html_path}")


if __name__ == "__main__":
    main()
