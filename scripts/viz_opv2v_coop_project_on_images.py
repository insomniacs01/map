#!/usr/bin/env python3
"""Project OPV2V cooperative colored point clouds back onto camera images.

This script is meant for side-by-side diagnosis of cooperative geometry quality.

- PredPose: use `pred["pts3d"]` and anchor the predicted world to GT ref-view0
- GTPose: use GT `pose_C2E_cv` to place `pred["pts3d_cam"]` into ego frame

By default it uses only finite checks plus the model `mask`, with no extra
radius/downsample/confidence filtering.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import batch_eval as be  # type: ignore
from mapanything.third_party.projection import project_3D_points_np
from mapanything.utils.image import rgb


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
            f"model={args.model_cfg}",
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
) -> Tuple[List[dict], List[dict]]:
    norm_type = _resolved_norm_type(args.model_arch, args.data_norm_type)
    processed_for_projection = be.preprocess_inputs(raw_views, norm_type=norm_type)
    processed_for_infer = be.preprocess_inputs(raw_views, norm_type=norm_type)
    for view in processed_for_infer:
        view.pop("depth_z", None)
    if not args.keep_camera_poses:
        be.strip_external_calibration_inputs(processed_for_infer)
    with torch.no_grad():
        if args.model_arch == "vggt":
            be._move_views_to_device(processed_for_infer, device)
            raw_preds = model(processed_for_infer)
            preds = be._vggt_preds_to_batch_eval_format(raw_preds, processed_for_infer)
        else:
            preds = model.infer(processed_for_infer, memory_efficient_inference=True)
    return preds, processed_for_projection


def _as_pose_np(value) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float64, copy=False)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4,4), got {arr.shape}")
    return arr


def _as_intrinsics_np(value) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float32, copy=False)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3,3), got {arr.shape}")
    return arr


def _transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32, copy=False)
    pts_h = np.concatenate(
        [points.astype(np.float64, copy=False), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (T_dst_src @ pts_h.T).T[:, :3]
    return out.astype(np.float32, copy=False)


def _tensor_image_to_u8(view: dict, pred: dict) -> np.ndarray:
    img_no_norm = pred.get("img_no_norm")
    if img_no_norm is not None and torch.is_tensor(img_no_norm):
        arr = img_no_norm[0].detach().cpu().numpy()
    else:
        norm_type = str(view["data_norm_type"][0])
        arr = rgb(view["img"], norm_type=norm_type)[0]
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        out = arr
    else:
        if arr.size and float(arr.max()) > 1.0:
            arr = arr / 255.0
        out = np.clip(arr, 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
    return out


def _colors_to_float(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _collect_clouds(
    args: argparse.Namespace,
    preds: Sequence[dict],
    processed_views: Sequence[dict],
    cam_infos: Sequence[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], np.ndarray]:
    pred_points: List[np.ndarray] = []
    pred_colors: List[np.ndarray] = []
    gtpose_points: List[np.ndarray] = []
    gtpose_colors: List[np.ndarray] = []
    per_view_stats: List[dict] = []

    gt_ref_pose = _as_pose_np(cam_infos[0]["pose_C2E_cv"])
    pred_ref_pose_tensor = preds[0].get("camera_poses") if preds else None
    if pred_ref_pose_tensor is not None:
        pred_ref_pose = _as_pose_np(pred_ref_pose_tensor)
        T_ego_pred = gt_ref_pose @ np.linalg.inv(pred_ref_pose)
    else:
        T_ego_pred = np.eye(4, dtype=np.float64)

    for idx, (pred, view, cam_info) in enumerate(zip(preds, processed_views, cam_infos)):
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
        valid0 = valid[0].detach().cpu().numpy().astype(bool, copy=False)
        if not valid0.any():
            continue

        pts3d_np = pts3d[0].detach().cpu().numpy().astype(np.float32, copy=False)
        pts3d_cam_np = pts3d_cam[0].detach().cpu().numpy().astype(np.float32, copy=False)
        colors_np = _tensor_image_to_u8(view, pred).reshape(-1, 3)

        pts3d_sel = pts3d_np.reshape(-1, 3)[valid0.reshape(-1)]
        pts3d_cam_sel = pts3d_cam_np.reshape(-1, 3)[valid0.reshape(-1)]
        colors_sel = _colors_to_float(colors_np[valid0.reshape(-1)])

        if args.align_pred_to_gt_ref_view:
            pred_sel_ego = _transform_points(pts3d_sel, T_ego_pred)
        else:
            pred_sel_ego = pts3d_sel.astype(np.float32, copy=False)

        pose_gt = _as_pose_np(cam_info["pose_C2E_cv"])
        gtpose_sel_ego = _transform_points(pts3d_cam_sel, pose_gt)

        pred_points.append(pred_sel_ego)
        pred_colors.append(colors_sel)
        gtpose_points.append(gtpose_sel_ego)
        gtpose_colors.append(colors_sel)

        pred_pose = pred.get("camera_poses")
        pose_stats = {
            "name": str(cam_info.get("name", f"view{idx}")),
            "valid_points": int(pts3d_sel.shape[0]),
        }
        if pred_pose is not None:
            pred_pose_np = _as_pose_np(pred_pose)
            pred_aligned = T_ego_pred @ pred_pose_np
            gt_rot = pose_gt[:3, :3]
            pr_rot = pred_aligned[:3, :3]
            trace_val = np.clip((np.trace(gt_rot.T @ pr_rot) - 1.0) / 2.0, -1.0, 1.0)
            pose_stats["pose_abs_trans_m"] = float(np.linalg.norm(pred_aligned[:3, 3] - pose_gt[:3, 3]))
            pose_stats["pose_abs_rot_deg"] = float(math.degrees(math.acos(trace_val)))
        per_view_stats.append(pose_stats)

    if not pred_points:
        raise RuntimeError("No valid predicted points found in model outputs.")

    return (
        np.concatenate(pred_points, axis=0),
        np.concatenate(pred_colors, axis=0),
        np.concatenate(gtpose_points, axis=0),
        np.concatenate(gtpose_colors, axis=0),
        per_view_stats,
        T_ego_pred.astype(np.float32, copy=False),
    )


def _dilate_pixels(colors_u8: np.ndarray, alpha: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    if radius <= 0:
        return colors_u8, alpha
    out_colors = colors_u8.copy()
    out_alpha = alpha.copy()
    base_colors = colors_u8.copy()
    base_mask = alpha > 0
    H, W = alpha.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(H, H - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(W, W - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(H, H + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(W, W + dx)
            if src_y0 >= src_y1 or src_x0 >= src_x1:
                continue
            src_mask = base_mask[src_y0:src_y1, src_x0:src_x1]
            if not src_mask.any():
                continue
            dst_colors = out_colors[dst_y0:dst_y1, dst_x0:dst_x1]
            dst_alpha = out_alpha[dst_y0:dst_y1, dst_x0:dst_x1]
            src_colors_view = base_colors[src_y0:src_y1, src_x0:src_x1]
            dst_colors[src_mask] = src_colors_view[src_mask]
            dst_alpha[src_mask] = 255
    return out_colors, out_alpha


def _project_cloud_to_image(
    points_ego: np.ndarray,
    colors_rgb: np.ndarray,
    pose_c2e_cv: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: Tuple[int, int],
    *,
    point_radius: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    H, W = image_hw
    if points_ego.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8), {"visible_points": 0, "filled_pixels": 0}

    e2c = np.linalg.inv(pose_c2e_cv)[:3, :4].astype(np.float32, copy=False)
    points2d, points_cam = project_3D_points_np(
        points_ego.astype(np.float32, copy=False),
        e2c[None],
        intrinsics.astype(np.float32, copy=False)[None],
    )
    uv = points2d[0]
    z = points_cam[0, 2]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 1e-4)
    if not valid.any():
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8), {"visible_points": 0, "filled_pixels": 0}

    uv = uv[valid]
    z = z[valid]
    cols = _colors_to_float(colors_rgb[valid])
    uu = np.rint(uv[:, 0]).astype(np.int32)
    vv = np.rint(uv[:, 1]).astype(np.int32)
    in_bounds = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    if not in_bounds.any():
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8), {"visible_points": 0, "filled_pixels": 0}

    uu = uu[in_bounds]
    vv = vv[in_bounds]
    z = z[in_bounds]
    cols = cols[in_bounds]

    order = np.argsort(z)[::-1]
    uu = uu[order]
    vv = vv[order]
    cols_u8 = (cols[order] * 255.0 + 0.5).astype(np.uint8)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)
    canvas[vv, uu] = cols_u8
    alpha[vv, uu] = 255
    canvas, alpha = _dilate_pixels(canvas, alpha, int(point_radius))
    stats = {
        "visible_points": int(len(order)),
        "filled_pixels": int(np.count_nonzero(alpha)),
    }
    return canvas, alpha, stats


def _overlay_projection(base_u8: np.ndarray, proj_u8: np.ndarray, alpha: np.ndarray, overlay_alpha: float) -> np.ndarray:
    out = base_u8.astype(np.float32).copy()
    mask = alpha > 0
    if mask.any():
        out[mask] = out[mask] * (1.0 - overlay_alpha) + proj_u8[mask].astype(np.float32) * overlay_alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _draw_titled_cell(image: np.ndarray, title: str, subtitle: Optional[str] = None) -> Image.Image:
    img = Image.fromarray(image)
    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(img)
    bbox = draw_probe.textbbox((0, 0), title, font=font)
    title_h = (bbox[3] - bbox[1]) + 6
    subtitle_h = 0
    if subtitle:
        bbox2 = draw_probe.textbbox((0, 0), subtitle, font=font)
        subtitle_h = (bbox2[3] - bbox2[1]) + 4
    pad = 4
    canvas = Image.new("RGB", (img.width, img.height + title_h + subtitle_h + pad * 2), (20, 20, 20))
    canvas.paste(img, (0, title_h + subtitle_h + pad * 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, pad), title, fill=(240, 240, 240), font=font)
    if subtitle:
        draw.text((6, pad + title_h), subtitle, fill=(170, 170, 170), font=font)
    return canvas


def _compose_row(images: Sequence[Image.Image], gap: int = 8) -> Image.Image:
    width = sum(im.width for im in images) + gap * max(0, len(images) - 1)
    height = max(im.height for im in images)
    canvas = Image.new("RGB", (width, height), (10, 10, 10))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def _stack_rows(rows: Sequence[Image.Image], top_text: str, gap: int = 12) -> Image.Image:
    if not rows:
        raise ValueError("rows cannot be empty")
    font = ImageFont.load_default()
    tmp = Image.new("RGB", (10, 10), (0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.multiline_textbbox((0, 0), top_text, font=font, spacing=4)
    header_h = (bbox[3] - bbox[1]) + 16
    width = max(im.width for im in rows)
    height = header_h + sum(im.height for im in rows) + gap * max(0, len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, 8), top_text, fill=(240, 240, 240), font=font, spacing=4)
    y = header_h
    for im in rows:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas


def _save_panel_set(
    frame_dir: Path,
    row_name: str,
    rgb_u8: np.ndarray,
    pred_proj_u8: np.ndarray,
    pred_overlay_u8: np.ndarray,
    gtpose_proj_u8: np.ndarray,
    gtpose_overlay_u8: np.ndarray,
    pred_stats: Dict[str, int],
    gtpose_stats: Dict[str, int],
) -> Image.Image:
    Image.fromarray(rgb_u8).save(frame_dir / f"{row_name}_rgb.png")
    Image.fromarray(pred_proj_u8).save(frame_dir / f"{row_name}_predpose_points.png")
    Image.fromarray(pred_overlay_u8).save(frame_dir / f"{row_name}_predpose_overlay.png")
    Image.fromarray(gtpose_proj_u8).save(frame_dir / f"{row_name}_gtpose_points.png")
    Image.fromarray(gtpose_overlay_u8).save(frame_dir / f"{row_name}_gtpose_overlay.png")

    row = _compose_row(
        [
            _draw_titled_cell(rgb_u8, "RGB"),
            _draw_titled_cell(
                pred_overlay_u8,
                "PredPose Overlay",
                subtitle=f"pts={pred_stats['visible_points']} pix={pred_stats['filled_pixels']}",
            ),
            _draw_titled_cell(
                pred_proj_u8,
                "PredPose Points",
                subtitle=f"pts={pred_stats['visible_points']} pix={pred_stats['filled_pixels']}",
            ),
            _draw_titled_cell(
                gtpose_overlay_u8,
                "GTPose Overlay",
                subtitle=f"pts={gtpose_stats['visible_points']} pix={gtpose_stats['filled_pixels']}",
            ),
            _draw_titled_cell(
                gtpose_proj_u8,
                "GTPose Points",
                subtitle=f"pts={gtpose_stats['visible_points']} pix={gtpose_stats['filled_pixels']}",
            ),
        ]
    )
    row.save(frame_dir / f"{row_name}_panel.png")
    return row


def _write_summary(
    out_path: Path,
    *,
    args: argparse.Namespace,
    frame_tag: str,
    pred_points: np.ndarray,
    gtpose_points: np.ndarray,
    align_matrix: np.ndarray,
    per_view_stats: Sequence[dict],
    projection_stats: Sequence[dict],
) -> None:
    lines = [
        f"frame={frame_tag}",
        f"split={args.split}",
        f"checkpoint={args.checkpoint}",
        f"pred_points_total={pred_points.shape[0]}",
        f"gtpose_points_total={gtpose_points.shape[0]}",
        f"align_pred_to_gt_ref_view={args.align_pred_to_gt_ref_view}",
        "T_ego_pred=",
        np.array2string(align_matrix, precision=4, suppress_small=False),
        "",
        "per_view_fused_counts:",
    ]
    for item in per_view_stats:
        name = item.get("name", "unknown")
        valid_points = item.get("valid_points", 0)
        pose_abs_trans = item.get("pose_abs_trans_m")
        pose_abs_rot = item.get("pose_abs_rot_deg")
        if pose_abs_trans is None or pose_abs_rot is None:
            lines.append(f"- {name}: valid_points={valid_points}")
        else:
            lines.append(
                f"- {name}: valid_points={valid_points} pose_abs_trans_m={pose_abs_trans:.3f} pose_abs_rot_deg={pose_abs_rot:.3f}"
            )
    lines.append("")
    lines.append("per_view_projection:")
    for item in projection_stats:
        lines.append(
            "- {name}: pred_visible_points={pred_visible_points} pred_filled_pixels={pred_filled_pixels} "
            "gtpose_visible_points={gtpose_visible_points} gtpose_filled_pixels={gtpose_filled_pixels}".format(**item)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="mapanything",
        help="Hydra model config name for mapanything checkpoints, e.g. mapanything_opv2v_coop_lowmem.",
    )
    parser.add_argument("--model_task", type=str, default="calibrated_sfm")
    parser.add_argument("--data_norm_type", type=str, default=None)
    parser.add_argument("--vggt_enable_metric_scale_head", action="store_true")
    parser.add_argument("--vggt_autocast_dtype", type=str, default=None)
    parser.add_argument("--keep_camera_poses", action="store_true")
    parser.add_argument("--align_pred_to_gt_ref_view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--overlay_alpha", type=float, default=0.82)
    parser.add_argument("--dinov2_hub_repo_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    _patch_torch_hub_for_local_dinov2(args.dinov2_hub_repo_dir)

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
    preds, processed_views = _infer_predictions(args, model, device, raw_views)
    pred_points, pred_colors, gtpose_points, gtpose_colors, per_view_stats, align_matrix = _collect_clouds(
        args,
        preds,
        processed_views,
        cam_infos,
    )

    frame_tag = f"{args.sequence}_{args.frame}"
    frame_dir = args.out_dir
    frame_dir.mkdir(parents=True, exist_ok=True)

    panel_rows: List[Image.Image] = []
    projection_stats: List[dict] = []
    for idx, (view, pred, cam_info) in enumerate(zip(processed_views, preds, cam_infos)):
        rgb_u8 = _tensor_image_to_u8(view, pred)
        H, W = rgb_u8.shape[:2]
        intrinsics = _as_intrinsics_np(view["intrinsics"])
        pose_c2e = _as_pose_np(cam_info["pose_C2E_cv"])
        row_name = str(cam_info.get("name", f"view{idx}"))

        pred_proj_u8, pred_alpha, pred_stats = _project_cloud_to_image(
            pred_points,
            pred_colors,
            pose_c2e,
            intrinsics,
            (H, W),
            point_radius=int(args.point_radius),
        )
        gtpose_proj_u8, gtpose_alpha, gtpose_stats = _project_cloud_to_image(
            gtpose_points,
            gtpose_colors,
            pose_c2e,
            intrinsics,
            (H, W),
            point_radius=int(args.point_radius),
        )
        pred_overlay_u8 = _overlay_projection(rgb_u8, pred_proj_u8, pred_alpha, float(args.overlay_alpha))
        gtpose_overlay_u8 = _overlay_projection(rgb_u8, gtpose_proj_u8, gtpose_alpha, float(args.overlay_alpha))
        row_panel = _save_panel_set(
            frame_dir,
            row_name,
            rgb_u8,
            pred_proj_u8,
            pred_overlay_u8,
            gtpose_proj_u8,
            gtpose_overlay_u8,
            pred_stats,
            gtpose_stats,
        )
        panel_rows.append(row_panel)
        projection_stats.append(
            {
                "name": row_name,
                "pred_visible_points": int(pred_stats["visible_points"]),
                "pred_filled_pixels": int(pred_stats["filled_pixels"]),
                "gtpose_visible_points": int(gtpose_stats["visible_points"]),
                "gtpose_filled_pixels": int(gtpose_stats["filled_pixels"]),
            }
        )

    top_text = "\n".join(
        [
            f"frame={frame_tag} split={args.split}",
            f"checkpoint={args.checkpoint.name}",
            f"pred_points_total={pred_points.shape[0]} gtpose_points_total={gtpose_points.shape[0]}",
            f"pred_anchor={'gt-ref0' if args.align_pred_to_gt_ref_view else 'raw-pred-world'} point_radius={args.point_radius}",
        ]
    )
    contact_sheet = _stack_rows(panel_rows, top_text)
    contact_sheet.save(frame_dir / "contact_sheet.png")
    _write_summary(
        frame_dir / "summary.txt",
        args=args,
        frame_tag=frame_tag,
        pred_points=pred_points,
        gtpose_points=gtpose_points,
        align_matrix=align_matrix,
        per_view_stats=per_view_stats,
        projection_stats=projection_stats,
    )


if __name__ == "__main__":
    main()
