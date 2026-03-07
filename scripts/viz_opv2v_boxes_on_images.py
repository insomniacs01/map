#!/usr/bin/env python3
"""
Project OPV2V GT 3D vehicle boxes (CARLA/UE ego) onto each camera view image.

This is a quick coordinate-convention sanity check:
  - OPV2V boxes / LiDAR are in CARLA/UE (X forward, Y right, Z up).
  - MapAnything camera poses are converted to OpenCV (X right, Y down, Z forward).

We:
  1) load one cooperative OPV2V sample (8 views by default),
  2) convert box corners CARLA->OpenCV,
  3) transform ego->camera using GT camera_pose,
  4) project with intrinsics and draw wireframes.

Optionally overlays GT LiDAR points projected into each camera to further validate axes.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_boxes_on_images.py \
    --split validate --index 0 --num_views 8 \
    --overlay_lidar --max_lidar_points 15000 \
    --out_dir eval_runs/opv2v_boxproj/val_idx0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mapanything.datasets.opv2v import OPV2VCoopDataset

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
    color: tuple[int, int, int] = (0, 255, 0),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None)
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, default=Path("eval_runs/opv2v_boxproj/val_idx0"))
    parser.add_argument("--overlay_lidar", action="store_true")
    parser.add_argument("--max_lidar_points", type=int, default=15000)
    parser.add_argument("--box_line_width", type=int, default=2)
    args = parser.parse_args()

    dataset = OPV2VCoopDataset(
        split=args.split,
        ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
        depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
        camera_ids=(0, 1, 2, 3),
        pair_agents=True,
        pair_agent_policy="nearest",
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

    sequence, main_agent, frame_id = _extract_scene_ids(views)

    gt_boxes = views[0]["vehicle_boxes"][0].detach().cpu().numpy().astype(np.float32)
    gt_mask = views[0]["vehicle_boxes_mask"][0].detach().cpu().numpy().astype(bool)
    gt_boxes = gt_boxes[gt_mask]

    corners_carla = _boxes_to_corners_carla(gt_boxes)  # (M, 8, 3) in ego CARLA
    corners_ego_cv = _carla_points_to_cv(corners_carla.reshape(-1, 3)).reshape(corners_carla.shape)

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
        cam_pose = view["camera_pose"][0].detach().cpu().numpy().astype(np.float32)  # cam2ego (OpenCV)

        # Project boxes.
        corners_cam = _ego_cv_to_cam_cv(corners_ego_cv.reshape(-1, 3), cam_pose).reshape(corners_ego_cv.shape)
        corners_uv, corners_z = _project_pinhole(corners_cam.reshape(-1, 3), K)
        corners_uv = corners_uv.reshape(corners_cam.shape[0], 8, 2)
        corners_z = corners_z.reshape(corners_cam.shape[0], 8)
        _draw_box_wireframes(
            img_pil,
            corners_uv,
            corners_z,
            color=(0, 255, 0),
            width=int(args.box_line_width),
        )

        # Optionally overlay LiDAR points.
        if lidar_points_carla is not None and lidar_points_carla.size > 0:
            lidar_ego_cv = _carla_points_to_cv(lidar_points_carla)
            lidar_cam = _ego_cv_to_cam_cv(lidar_ego_cv, cam_pose)
            uv, z = _project_pinhole(lidar_cam, K)
            uv = uv[z > 0.1]
            img_pil = _overlay_points(img_pil, uv, color=(255, 0, 0), alpha=0.55)

        inst = str(_take_first(view.get("instance", "")) or f"view{view_idx}")
        _draw_label(img_pil, f"{sequence}/{main_agent} {frame_id} | {inst}")
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
<head><meta charset="utf-8"><title>OPV2V box projection</title></head>
<body style="background:#111;color:#eee;font-family:monospace">
<h3>OPV2V box projection (split={args.split} idx={args.index} seq={sequence} main={main_agent} frame={frame_id})</h3>
<p>Green: GT 3D boxes projected. Red: projected LiDAR (optional).</p>
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
