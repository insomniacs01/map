#!/usr/bin/env python3
"""
Run MapAnything inference on a dataset sample and export an interactive HTML report that
shows both the input panoramas and an RGB point cloud rendered with Plotly.

Example (cylindrical OPV2V checkpoint):

PYTHONPATH=$(pwd) python scripts/render_pointcloud_html.py \
    --checkpoint /home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_coop/20251201_021942/checkpoint-best.pth \
    --split val --index 0 \
    --output-html eval_runs/opv2v_cyl_precheck/val_idx0_pointcloud.html
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys

if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))

import mapanything.datasets as dataset_registry
from mapanything.models import init_model
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

CONFIG_DIR = REPO_ROOT / "configs"


def _parse_overrides(s: str) -> List[str]:
    return [ov.strip() for ov in s.split(",") if ov.strip()]


def _parse_vec3(text: str | None):
    if not text:
        return None
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got '{text}'")
    return dict(x=parts[0], y=parts[1], z=parts[2])


def _compose_cfg(overrides: List[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=overrides)


def _extract_dataset_exprs(dataset_spec: str) -> List[str]:
    text = dataset_spec.replace("\n", " ").strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    exprs = []
    idx = 0
    while idx < len(text):
        at_pos = text.find("@", idx)
        if at_pos == -1:
            break
        cursor = at_pos + 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        start = cursor
        depth = 0
        seen_paren = False
        while cursor < len(text):
            ch = text[cursor]
            if ch == "(":
                depth += 1
                seen_paren = True
            elif ch == ")":
                depth -= 1
                if seen_paren and depth == 0:
                    cursor += 1
                    break
            cursor += 1
        exprs.append(text[start:cursor].strip())
        idx = cursor
    if not exprs and text:
        exprs.append(text.strip())
    return exprs


def _instantiate_dataset(dataset_expr: str):
    context = vars(dataset_registry).copy()
    context["np"] = np
    context["__builtins__"] = __builtins__
    if "OPV2VCoopCylindricalDataset" not in context:
        print("DEBUG: OPV2VCoopCylindricalDataset missing from eval context")
    return eval(dataset_expr, context)


def _fetch_views(dataset, sample_idx: int):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    if isinstance(dataset.num_views, int):
        return dataset[sample_idx]
    return dataset[(sample_idx, 0, 0)]


def _torchify(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(array):
        return array
    return torch.from_numpy(array)


def _denorm_image(tensor: torch.Tensor, norm_type: str) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    norm_cfg = IMAGE_NORMALIZATION_DICT.get(norm_type)
    if norm_cfg is not None:
        mean = torch.tensor(norm_cfg.mean, dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(norm_cfg.std, dtype=torch.float32)[None, :, None, None]
        tensor = tensor * std + mean
    tensor = tensor.squeeze(0).clamp(0.0, 1.0)
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    return (tensor * 255.0).astype(np.uint8)


def _prepare_views_for_inference(
    raw_views: Sequence[Dict[str, Any]],
    pose_input_mode: str,
    pose_noise_trans: float,
    pose_noise_rot: float,
    depth_input_mode: str,
    use_calibration_inputs: bool,
    metric_scale_input: bool,
    sparse_keep_ratio: float,
    rng: np.random.Generator,
):
    prepared: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []

    for view in raw_views:
        img_tensor = view["img"].detach().clone()
        data_norm_type = view.get("data_norm_type", "identity")
        camera_model = view.get("camera_model", "pinhole")
        camera_pose = view["camera_pose"].astype(np.float32)
        pose_tensor = torch.from_numpy(camera_pose).unsqueeze(0)

        entry: Dict[str, Any] = {
            "img": img_tensor.unsqueeze(0),
            "data_norm_type": [data_norm_type],
            "camera_model": camera_model,
            "instance": [view.get("instance", "")],
        }

        if use_calibration_inputs:
            if camera_model.lower() in ("cyl", "cylindrical"):
                entry["ray_directions"] = (
                    torch.from_numpy(view["ray_directions_cam"].astype(np.float32))
                    .unsqueeze(0)
                )
            else:
                entry["intrinsics"] = (
                    torch.from_numpy(view["camera_intrinsics"].astype(np.float32))
                    .unsqueeze(0)
                )

        if depth_input_mode != "none":
            depth_cam = view["pts3d_cam"][..., 2].astype(np.float32)
            if depth_input_mode == "sparse":
                valid = depth_cam > 0
                sample_mask = np.zeros_like(valid, dtype=bool)
                valid_idx = np.argwhere(valid)
                if valid_idx.size > 0:
                    keep = max(1, int(math.ceil(len(valid_idx) * sparse_keep_ratio)))
                    keep_idx = rng.choice(len(valid_idx), size=keep, replace=False)
                    coords = valid_idx[keep_idx]
                    sample_mask[coords[:, 0], coords[:, 1]] = True
                    depth_cam = np.where(sample_mask, depth_cam, 0.0)
            entry["depth_z"] = torch.from_numpy(depth_cam).unsqueeze(0)

        if pose_input_mode != "none":
            pose_np = camera_pose.copy()
            if pose_input_mode == "noisy":
                if pose_noise_trans > 0:
                    pose_np[:3, 3] += rng.normal(scale=pose_noise_trans, size=3)
                if pose_noise_rot > 0:
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis) + 1e-8
                    angle = math.radians(rng.normal(scale=pose_noise_rot))
                    rot_delta = Rotation.from_rotvec(axis * angle).as_matrix()
                    pose_np[:3, :3] = rot_delta @ pose_np[:3, :3]
            pose_tensor = torch.from_numpy(pose_np).unsqueeze(0)
            entry["camera_poses"] = pose_tensor
        else:
            pose_tensor = torch.from_numpy(camera_pose).unsqueeze(0)

        entry["is_metric_scale"] = torch.tensor(
            [metric_scale_input], dtype=torch.bool
        )

        prepared.append(entry)

        infos.append(
            {
                "camera_model": camera_model,
                "virtual_camera": view.get("virtual_camera"),
                "agent_id": view.get("agent_id", ""),
                "label": view.get("label", ""),
                "instance": view.get("instance", ""),
                "rgb_tensor": img_tensor,
                "data_norm_type": data_norm_type,
                "gt_pts3d_world": view["pts3d"],
                "gt_pts3d_cam": view["pts3d_cam"],
                "gt_mask": view["non_ambiguous_mask"].astype(bool),
                "valid_mask": view["valid_mask"].astype(bool),
                "gt_pose": camera_pose,
                "input_pose": pose_tensor[0].cpu().numpy()
                if "camera_poses" in entry
                else None,
            }
        )

    return prepared, infos


def _determine_pose_matrix(
    view_info: Dict[str, Any], pred: Dict[str, Any], mode: str
) -> np.ndarray:
    if mode == "pred":
        pose = pred.get("camera_pose_pred")
        if pose is not None:
            return pose
    if mode == "input" and view_info.get("input_pose") is not None:
        return view_info["input_pose"]
    return view_info["gt_pose"]


def _sample_mask(mask: np.ndarray, keep: float, rng: np.random.Generator) -> np.ndarray:
    flat = mask.reshape(-1)
    idx = np.flatnonzero(flat)
    if len(idx) == 0 or keep >= 1.0:
        return mask
    num_keep = max(1, int(len(idx) * keep))
    chosen = rng.choice(idx, size=num_keep, replace=False)
    new_mask = np.zeros_like(flat, dtype=bool)
    new_mask[chosen] = True
    return new_mask.reshape(mask.shape)


def _transform_points(points_cam: np.ndarray, pose: np.ndarray) -> np.ndarray:
    R = pose[:3, :3]
    t = pose[:3, 3]
    pts = points_cam.reshape(-1, 3)
    transformed = pts @ R.T + t
    return transformed


def _build_point_cloud(
    view_infos: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
    point_depth_source: str,
    point_pose_source: str,
    point_scale_source: str,
    sparse_ratio: float,
    max_points: int,
    rng: np.random.Generator,
    mask_points: bool,
) -> Dict[str, Any]:
    all_points = []
    all_colors = []
    per_view_stats = []

    for view_idx, (info, pred) in enumerate(zip(view_infos, preds)):
        stats_entry = {
            "view_idx": view_idx,
            "agent": info.get("agent_id", ""),
            "camera_model": info["camera_model"],
            "points_kept": 0,
            "pose_used": point_pose_source,
            "depth_source": point_depth_source,
        }
        pose = _determine_pose_matrix(info, pred, point_pose_source)
        if point_depth_source == "pred":
            pts_cam = pred["pts3d_cam"]
            mask = pred.get("mask_pred")
            if mask is None:
                mask = np.ones(pts_cam.shape[:2], dtype=bool)
        elif point_depth_source == "gt_dense":
            pts_cam = info["gt_pts3d_cam"]
            mask = info["gt_mask"] & info["valid_mask"]
        else:  # gt_sparse
            pts_cam = info["gt_pts3d_cam"]
            mask = info["gt_mask"] & info["valid_mask"]
            mask = mask & _sample_mask(mask, sparse_ratio, rng)

        rgb = pred["rgb_image"]
        mask = mask & (np.isfinite(pts_cam).all(axis=-1))
        valid_points = pts_cam[mask]
        valid_colors = rgb[mask]

        if valid_points.size == 0:
            per_view_stats.append(stats_entry)
            continue

        points_world = _transform_points(valid_points, pose)

        if point_scale_source == "pred":
            scale = pred.get("metric_scale")
            if scale is not None:
                points_world *= scale

        all_points.append(points_world)
        all_colors.append(valid_colors)
        stats_entry["points_kept"] = points_world.shape[0]
        per_view_stats.append(stats_entry)

    if not all_points:
        return {
            "points": np.zeros((1, 3)),
            "colors": np.array([[255, 255, 255]], dtype=np.uint8),
            "per_view": per_view_stats,
        }

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if mask_points and points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return {
        "points": points,
        "colors": colors,
        "per_view": per_view_stats,
    }


def _camera_traces(pose: np.ndarray, name: str, scale: float = 0.5):
    origin = pose[:3, 3]
    axes = pose[:3, :3]
    colors = {"x": "red", "y": "green", "z": "blue"}
    traces = []
    for axis_idx, axis_key in enumerate(["x", "y", "z"]):
        end = origin + axes[:, axis_idx] * scale
        traces.append(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=colors[axis_key], width=3),
                name=f"{name}_{axis_key}",
                showlegend=False,
            )
        )
    return traces


def _build_html_figure(
    pointcloud: Dict[str, Any],
    view_infos: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[go.Figure, int]:
    rows = 1 + len(view_infos)
    specs = [[{"type": "scene", "colspan": 3}, None, None]]
    for _ in view_infos:
        specs.append([{"type": "xy"}, {"type": "xy"}, {"type": "xy"}])

    total_height = args.pointcloud_height + len(view_infos) * args.view_panel_height
    row_heights = (
        [args.pointcloud_height / total_height]
        + [args.view_panel_height / total_height] * len(view_infos)
    )

    fig = make_subplots(
        rows=rows,
        cols=3,
        specs=specs,
        column_widths=[0.33, 0.33, 0.34],
        row_heights=row_heights,
        subplot_titles=(
            ["3D point cloud"]
            + sum(
                [
                    [
                        f"View {idx} RGB",
                        "Depth",
                        "Mask",
                    ]
                    for idx in range(len(view_infos))
                ],
                [],
            )
        ),
        vertical_spacing=0.04,
        horizontal_spacing=0.03,
    )

    pts = pointcloud["points"]
    cols = pointcloud["colors"]
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=args.point_size,
                color=[f"rgb({r},{g},{b})" for r, g, b in cols],
            ),
            name="points",
        ),
        row=1,
        col=1,
    )

    for vidx, (info, pred) in enumerate(zip(view_infos, preds)):
        pose = _determine_pose_matrix(info, pred, args.point_pose_source)
        for trace in _camera_traces(pose, f"cam_{vidx}", scale=args.cam_axis_scale):
            fig.add_trace(trace, row=1, col=1)
        fig.add_trace(
            go.Scatter3d(
                x=[pose[0, 3]],
                y=[pose[1, 3]],
                z=[pose[2, 3]],
                mode="text",
                text=[f"View {vidx}"],
                textposition="top center",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        row = vidx + 2
        rgb = pred["rgb_image"]
        depth = pred["depth_vis"]
        mask = pred["mask_vis"]

        fig.add_trace(go.Image(z=rgb), row=row, col=1)
        fig.add_trace(
            go.Heatmap(
                z=depth,
                colorscale="Turbo",
                showscale=False,
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=mask.astype(float),
                colorscale=[[0, "black"], [1, "lime"]],
                zmin=0,
                zmax=1,
                showscale=False,
            ),
            row=row,
            col=3,
        )

        meta = textwrap.dedent(
            f"""
            view {vidx} · agent={info.get('agent_id','?')} · model={info['camera_model']}
            pose used: {args.point_pose_source} · depth source: {args.point_depth_source}
            points kept: {pointcloud['per_view'][vidx]['points_kept'] if vidx < len(pointcloud['per_view']) else 0}
            """
        ).strip()
        fig.add_annotation(
            text=meta,
            xref=f"x{row} domain",
            yref=f"y{row} domain",
            x=0,
            y=1.1,
            showarrow=False,
            row=row,
            col=1,
            font=dict(size=10),
        )

    calib_flag = "off" if args.no_calibration_input else "on"
    summary = textwrap.dedent(
        f"""
        checkpoint: {Path(args.checkpoint).name} · split={args.split} idx={args.index}
        pose_input={args.pose_input} depth_input={args.depth_input} calibration={calib_flag}
        point_pose_source={args.point_pose_source} point_depth_source={args.point_depth_source} point_scale_source={args.point_scale_source}
        """
    ).strip()
    fig.update_layout(
        title=summary,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
            dragmode="orbit",
        ),
        height=total_height,
        width=args.figure_width,
        autosize=False,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    return fig, total_height


def _convert_preds_for_vis(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for pred in preds:
        rgb = (
            pred["img_no_norm"][0]
            .detach()
            .cpu()
            .numpy()
        )
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        pts3d_cam = pred.get("pts3d_cam")
        if pts3d_cam is None:
            raise ValueError("Predictions must contain pts3d_cam for visualization.")
        pts3d_cam_np = pts3d_cam[0].detach().cpu().numpy()
        mask_pred = pred.get("mask")
        if mask_pred is not None:
            mask_np = mask_pred[0, ..., 0].detach().cpu().numpy() > 0.5
        else:
            mask_np = np.ones(pts3d_cam_np.shape[:2], dtype=bool)
        metric_scale = (
            float(pred["metric_scaling_factor"][0].item())
            if pred.get("metric_scaling_factor") is not None
            else None
        )
        pose_pred = (
            pred["camera_poses"][0].detach().cpu().numpy()
            if pred.get("camera_poses") is not None
            else None
        )
        depth_tensor = pred.get("depth_along_ray")
        if depth_tensor is None:
            depth_tensor = pred.get("depth_z")
        if depth_tensor is not None:
            depth_np = depth_tensor[0, ..., 0].detach().cpu().numpy()
        else:
            depth_np = np.zeros(pts3d_cam_np.shape[:2], dtype=np.float32)
        mask_vis = pred.get("non_ambiguous_mask")
        if mask_vis is not None:
            mask_vis_np = mask_vis[0].detach().cpu().numpy()
        else:
            mask_vis_np = np.ones(pts3d_cam_np.shape[:2], dtype=bool)
        view = {
            "rgb_image": rgb,
            "pts3d_cam": pts3d_cam_np,
            "mask_pred": mask_np,
            "metric_scale": metric_scale,
            "camera_pose_pred": pose_pred,
            "depth_vis": depth_np,
            "mask_vis": mask_vis_np,
        }
        converted.append(view)
    return converted


def _save_view_images(output_dir: Path, view_infos, pred_views):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (info, pred) in enumerate(zip(view_infos, pred_views)):
        agent = info.get("agent_id", "unknown")
        base = f"view{idx}_agent{agent}"
        Image.fromarray(pred["rgb_image"]).save(output_dir / f"{base}_rgb.png")

        depth = pred["depth_vis"]
        finite = np.isfinite(depth)
        if finite.any():
            vmin = float(depth[finite].min())
            vmax = float(depth[finite].max())
            norm = (depth - vmin) / (vmax - vmin + 1e-6)
        else:
            norm = np.zeros_like(depth)
        depth_img = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(output_dir / f"{base}_depth.png")

        mask = (pred["mask_vis"].astype(np.uint8)) * 255
        Image.fromarray(mask).save(output_dir / f"{base}_mask.png")


def _pose_errors(gt: np.ndarray, other: np.ndarray | None):
    if other is None:
        return None
    rot_delta = gt[:3, :3].T @ other[:3, :3]
    angle = Rotation.from_matrix(rot_delta).magnitude()
    trans = np.linalg.norm(gt[:3, 3] - other[:3, 3])
    return {
        "rot_deg": math.degrees(angle),
        "trans": float(trans),
    }


def _collect_debug_stats(view_infos, pred_views, pointcloud):
    stats = []
    per_view = pointcloud.get("per_view", [])
    for idx, (info, pred) in enumerate(zip(view_infos, pred_views)):
        mask_vis = pred["mask_vis"]
        mask_ratio = float(mask_vis.mean()) if mask_vis.size else 0.0
        mask_pred = pred.get("mask_pred")
        pred_ratio = float(mask_pred.mean()) if mask_pred is not None else None

        depth = pred["depth_vis"]
        finite = np.isfinite(depth) & (depth > 0)
        if finite.any():
            depth_min = float(depth[finite].min())
            depth_max = float(depth[finite].max())
        else:
            depth_min = depth_max = None

        stats.append(
            {
                "view_idx": idx,
                "agent_id": info.get("agent_id"),
                "camera_model": info["camera_model"],
                "mask_ratio": mask_ratio,
                "pred_mask_ratio": pred_ratio,
                "depth_min": depth_min,
                "depth_max": depth_max,
                "metric_scale": pred.get("metric_scale"),
                "pose_pred_error": _pose_errors(info["gt_pose"], pred.get("camera_pose_pred")),
                "pose_input_error": _pose_errors(info["gt_pose"], info.get("input_pose")),
                "points_kept": per_view[idx]["points_kept"] if idx < len(per_view) else 0,
            }
        )
    return stats


def load_model_from_checkpoint(cfg, checkpoint_path: str, device: torch.device):
    model = init_model(
        cfg.model.model_str,
        cfg.model.model_config,
        torch_hub_force_reload=cfg.model.torch_hub_force_reload,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--expr-index", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-html", required=True, type=str)
    parser.add_argument("--output-image", type=str, help="Optional path to save a static image (requires kaleido).")
    parser.add_argument(
        "--hydra-overrides",
        type=str,
        default="machine=local3090,dataset=opv2v_cyl_coop_ft,model=mapanything,loss=overall_loss,train_params=opv2v_cyl_coop",
    )
    parser.add_argument("--pose-input", choices=["gt", "noisy", "none"], default="gt")
    parser.add_argument("--pose-noise-trans", type=float, default=0.3)
    parser.add_argument("--pose-noise-rot", type=float, default=3.0)
    parser.add_argument(
        "--depth-input", choices=["dense", "sparse", "none"], default="dense"
    )
    parser.add_argument(
        "--point-depth-source",
        choices=["pred", "gt_dense", "gt_sparse"],
        default="pred",
    )
    parser.add_argument(
        "--point-pose-source", choices=["pred", "gt", "input"], default="pred"
    )
    parser.add_argument(
        "--point-scale-source", choices=["pred", "gt"], default="pred"
    )
    parser.add_argument("--sparse-ratio", type=float, default=0.1)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--point-size", type=float, default=1.6)
    parser.add_argument("--cam-axis-scale", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mask-points", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-calibration-input", action="store_true")
    parser.add_argument("--disable-metric-scale-input", action="store_true")
    parser.add_argument("--sparse-input-ratio", type=float, default=0.05)
    parser.add_argument("--memory-efficient", action="store_true")
    parser.add_argument("--figure-width", type=int, default=1600, help="Width of the final HTML figure in pixels.")
    parser.add_argument(
        "--pointcloud-height",
        type=int,
        default=600,
        help="Pixel height allocated to the 3D point cloud panel.",
    )
    parser.add_argument(
        "--view-panel-height",
        type=int,
        default=360,
        help="Pixel height allocated to each RGB/depth/mask row.",
    )
    parser.add_argument("--view-output-dir", type=str, help="Optional directory for per-view RGB/depth/mask exports.")
    parser.add_argument("--debug-summary-json", type=str, help="Optional JSON path for per-view metrics.")
    parser.add_argument("--scene-camera-eye", type=str, help="Comma separated camera eye for Plotly scene, e.g. '2,2,1'.")
    parser.add_argument("--scene-camera-center", type=str, help="Comma separated camera center.")
    parser.add_argument("--scene-camera-up", type=str, help="Comma separated camera up vector.")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = _parse_overrides(args.hydra_overrides)
    cfg = _compose_cfg(overrides)

    if args.split == "train":
        dataset_spec = cfg.dataset.train_dataset
    else:
        dataset_spec = cfg.dataset.test_dataset
    exprs = _extract_dataset_exprs(dataset_spec)
    if not exprs:
        raise ValueError(f"No dataset expressions for split {args.split}")
    dataset_expr = exprs[min(max(args.expr_index, 0), len(exprs) - 1)]
    print(f"[render_pointcloud_html] Using dataset expression: {dataset_expr}")
    dataset = _instantiate_dataset(dataset_expr)
    raw_views = _fetch_views(dataset, args.index)

    rng = np.random.default_rng(args.seed)
    prepared_views, view_infos = _prepare_views_for_inference(
        raw_views,
        pose_input_mode=args.pose_input,
        pose_noise_trans=args.pose_noise_trans,
        pose_noise_rot=args.pose_noise_rot,
        depth_input_mode=args.depth_input,
        use_calibration_inputs=not args.no_calibration_input,
        metric_scale_input=not args.disable_metric_scale_input,
        sparse_keep_ratio=args.sparse_input_ratio,
        rng=rng,
    )

    if args.no_calibration_input and args.depth_input != "none":
        raise ValueError(
            "Depth inputs require calibration. Disable depth inputs or enable calibration inputs."
        )

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_checkpoint(cfg, args.checkpoint, device)

    preds = model.infer(
        prepared_views,
        memory_efficient_inference=args.memory_efficient,
        use_amp=not args.disable_amp and device.type == "cuda",
        ignore_calibration_inputs=args.no_calibration_input,
        ignore_depth_inputs=args.depth_input == "none",
        ignore_pose_inputs=args.pose_input == "none",
        ignore_depth_scale_inputs=args.disable_metric_scale_input,
        ignore_pose_scale_inputs=False,
    )

    pred_views = _convert_preds_for_vis(preds)

    pointcloud = _build_point_cloud(
        view_infos=view_infos,
        preds=pred_views,
        point_depth_source=args.point_depth_source,
        point_pose_source=args.point_pose_source,
        point_scale_source=args.point_scale_source,
        sparse_ratio=args.sparse_ratio,
        max_points=args.max_points,
        rng=rng,
        mask_points=not args.no_mask_points,
    )

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = _build_html_figure(pointcloud, view_infos, pred_views, args)
    scene_camera = {}
    eye = _parse_vec3(args.scene_camera_eye)
    center = _parse_vec3(args.scene_camera_center)
    up = _parse_vec3(args.scene_camera_up)
    if eye:
        scene_camera["eye"] = eye
    if center:
        scene_camera["center"] = center
    if up:
        scene_camera["up"] = up
    if scene_camera:
        fig.update_layout(scene_camera=scene_camera)

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    print(f"Saved HTML visualization to {output_path}")

    if args.output_image:
        image_path = Path(args.output_image)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(image_path))
            print(f"Saved static image to {image_path}")
        except ValueError as exc:
            print(
                f"Warning: could not export {image_path} (missing kaleido?). "
                f"Install 'kaleido' and retry. Original error: {exc}"
            )

    if args.view_output_dir:
        _save_view_images(Path(args.view_output_dir), view_infos, pred_views)

    stats = _collect_debug_stats(view_infos, pred_views, pointcloud)
    for entry in stats:
        pose_err = entry["pose_pred_error"]
        pose_info = (
            f"rot={pose_err['rot_deg']:.2f}deg trans={pose_err['trans']:.2f}m"
            if pose_err
            else "n/a"
        )
        print(
            f"view {entry['view_idx']} agent={entry['agent_id']} mask={entry['mask_ratio']:.3f} "
            f"depth=({entry['depth_min']},{entry['depth_max']}) scale={entry['metric_scale']} "
            f"pose_pred_err={pose_info} points={entry['points_kept']}"
        )

    if args.debug_summary_json:
        summary = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "index": args.index,
            "pose_input": args.pose_input,
            "depth_input": args.depth_input,
            "point_pose_source": args.point_pose_source,
            "point_depth_source": args.point_depth_source,
            "point_scale_source": args.point_scale_source,
            "stats": stats,
        }
        json_path = Path(args.debug_summary_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved debug summary to {json_path}")


if __name__ == "__main__":
    main()
