#!/usr/bin/env python3
"""
Readable 2D BEV PNG gallery for OPV2V eval runs (side-by-side across models).

Pipeline
--------
1) Read per-frame metrics from `<eval_run_dir>/<model_key>/<mode>_metrics.csv`
2) Select representative frames (best/median/worst) for a set of metric keys
   using the same derived-metric definitions as `metric_viz_from_eval_csv.py`.
3) For each selected frame and each model:
   - export (infer) or reuse a predicted ego-frame point cloud (.pcd)
   - load GT LiDAR points (main or fused main+coop) + GT boxes from OPV2V YAML
   - render a BEV overlay PNG with the key numbers printed on-tile
4) Write a simple offline HTML gallery with rows=frames and cols=models.
5) Optionally write distribution plots (hist/scatter) to interpret metrics.

Example
-------
  cd map-anything
  python scripts/metric_viz_readable_bev_gallery.py \\
    --models nm4_base=eval_runs/geom_nm4_test500_... \\
            promoted_v2=eval_runs/geom_promoted_test500_... \\
            vggt_plan2=eval_runs/geom_vggt_test500_20260226_plan2_coop_metric_e30_best \\
    --mode coop \\
    --metrics scale_to_gt_mult_err cross_agent_pose_trans_m cross_agent_pose_rot_deg depth_rel \\
    --out_dir eval_runs/metric_viz_test500_20260226_readable \\
    --write_plots
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing.opv2v_pose_utils import (  # noqa: E402
    cords_to_pose,
    load_ascii_pcd_xyz,
    load_frame_metadata,
)
from mapanything.datasets.opv2v import _extract_vehicle_boxes_in_ego  # noqa: E402


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_nx3(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError(f"expected Nx3 points; got {points.shape}")
    return points.astype(np.float32, copy=False)


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _filter_points(points: np.ndarray, *, z_min: float | None, z_max: float | None, radius_max: float | None) -> np.ndarray:
    if points.size == 0:
        return points
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= points[:, 2] <= float(z_max)
    if radius_max is not None and float(radius_max) > 0:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= float(radius_max)
    return points[mask]


def _transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    pts_h = np.concatenate(
        [points.astype(np.float64, copy=False), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = (T_dst_src @ pts_h.T).T[:, :3]
    return out.astype(np.float32)


def _bev_cell_indices(
    points_xy: np.ndarray,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return BEV grid indices (ix, iy) and valid mask for XY points.

    Grid convention:
      - ix indexes X (columns), iy indexes Y (rows)
      - (x_min, y_min) maps to (0, 0)
    """
    if points_xy.size == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=bool),
            0,
            0,
        )
    if float(resolution) <= 0:
        raise ValueError(f"resolution must be > 0; got {resolution}")
    x_min, x_max = float(xlim[0]), float(xlim[1])
    y_min, y_max = float(ylim[0]), float(ylim[1])
    w = int(math.ceil((x_max - x_min) / float(resolution)))
    h = int(math.ceil((y_max - y_min) / float(resolution)))
    ix = np.floor((points_xy[:, 0] - x_min) / float(resolution)).astype(np.int32)
    iy = np.floor((points_xy[:, 1] - y_min) / float(resolution)).astype(np.int32)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    return ix, iy, valid, w, h


def _bev_match_mask(
    *,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: float,
) -> np.ndarray:
    """Binary match mask for pred points based on GT occupancy in a BEV grid."""
    if pred_points.size == 0:
        return np.zeros((0,), dtype=bool)
    ix_gt, iy_gt, v_gt, w, h = _bev_cell_indices(gt_points[:, :2], xlim=xlim, ylim=ylim, resolution=resolution)
    if w <= 0 or h <= 0:
        return np.zeros((pred_points.shape[0],), dtype=bool)
    gt_occ = np.zeros((h, w), dtype=bool)
    if ix_gt.size and bool(v_gt.any()):
        gt_occ[iy_gt[v_gt], ix_gt[v_gt]] = True

    ix_p, iy_p, v_p, _w2, _h2 = _bev_cell_indices(pred_points[:, :2], xlim=xlim, ylim=ylim, resolution=resolution)
    match = np.zeros((pred_points.shape[0],), dtype=bool)
    if ix_p.size and bool(v_p.any()):
        match[v_p] = gt_occ[iy_p[v_p], ix_p[v_p]]
    return match


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
    """Load GT LiDAR points in the main-agent ego frame (CARLA coordinates)."""

    if mode not in ("main", "fused"):
        raise ValueError(f"unknown gt lidar mode: {mode} (expected: main|fused)")

    lidar_root = lidar_root.expanduser().resolve()
    main_yaml = lidar_root / split / sequence / main_agent / f"{frame}.yaml"
    if not main_yaml.is_file():
        raise FileNotFoundError(main_yaml)

    meta_main = load_frame_metadata(main_yaml)
    T_world_main = cords_to_pose(meta_main["lidar_pose"])
    T_main_world = np.linalg.inv(T_world_main)

    selected_agents = [main_agent] if mode == "main" else sorted({str(main_agent), *[str(a) for a in coop_agents]})

    clouds: List[np.ndarray] = []
    for agent in selected_agents:
        pcd_path = lidar_root / split / sequence / agent / f"{frame}.pcd"
        yaml_path = lidar_root / split / sequence / agent / f"{frame}.yaml"
        if not pcd_path.is_file() or not yaml_path.is_file():
            continue
        pts = _ensure_nx3(load_ascii_pcd_xyz(pcd_path))
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


def _load_gt_boxes_ego(
    *,
    lidar_root: Path,
    split: str,
    sequence: str,
    frame: str,
    main_agent: str,
    bbox_range: float,
    max_boxes: int,
) -> np.ndarray:
    yaml_path = lidar_root / split / sequence / main_agent / f"{frame}.yaml"
    if not yaml_path.is_file():
        return np.zeros((0, 7), dtype=np.float32)
    meta = load_frame_metadata(yaml_path)
    padded, mask = _extract_vehicle_boxes_in_ego(meta, max_range=float(bbox_range), max_num_boxes=int(max_boxes))
    boxes = padded[mask]
    return boxes.astype(np.float32, copy=False)


def _box_to_bev_poly(box: np.ndarray) -> np.ndarray:
    # box: [x, y, z, l, w, h, yaw]
    x, y, _z, l, w, _h, yaw = [float(v) for v in box]
    dx = l / 2.0
    dy = w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _plot_boxes(ax: plt.Axes, boxes: np.ndarray, *, color: str = "black", lw: float = 1.2) -> None:
    if boxes.size == 0:
        return
    for box in boxes:
        poly = _box_to_bev_poly(box)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=lw)


def _infer_metric_value(row: Mapping[str, str], metric_key: str) -> float | None:
    """Metric lookup + derived keys aligned with `metric_viz_from_eval_csv.py`."""

    if metric_key in row:
        return _safe_float(row.get(metric_key))

    if metric_key == "scale_to_gt_mult_err":
        eq_rel = _safe_float(row.get("scale_to_gt_eq_rel_err"))
        if eq_rel is not None:
            return float(eq_rel) + 1.0
        log_err = _safe_float(row.get("scale_to_gt_log_err"))
        if log_err is not None:
            return float(math.exp(float(log_err)))
        return None

    if metric_key == "scale_to_gt_abs_ratio_err":
        ratio = _safe_float(row.get("scale_to_gt_ratio_mean"))
        if ratio is None:
            return None
        return abs(float(ratio) - 1.0)

    return None


@dataclass(frozen=True)
class FrameMeta:
    sequence: str
    frame: str
    main_agent: str
    coop_agents: Tuple[str, ...]


def _load_frames_from_summary(summary_json: Path) -> Dict[Tuple[str, str], FrameMeta]:
    doc = json.loads(summary_json.read_text(encoding="utf-8"))
    frames = doc.get("frames") or []
    out: Dict[Tuple[str, str], FrameMeta] = {}
    if not isinstance(frames, list):
        return out
    for item in frames:
        if not isinstance(item, dict):
            continue
        seq = item.get("sequence")
        fr = item.get("frame")
        main = item.get("main_agent")
        coop = item.get("coop_agents") or ()
        if not (isinstance(seq, str) and isinstance(fr, str) and isinstance(main, str)):
            continue
        if isinstance(coop, (list, tuple)):
            coop_t = tuple(str(x) for x in coop)
        else:
            coop_t = (str(main),)
        out[(seq, fr)] = FrameMeta(sequence=seq, frame=fr, main_agent=str(main), coop_agents=coop_t)
    return out


def _load_metrics_csv(csv_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            seq = row.get("sequence")
            fr = row.get("frame")
            if not seq or not fr:
                continue
            out[(str(seq), str(fr))] = dict(row)
    return out


@dataclass(frozen=True)
class SelectedFrame:
    sequence: str
    frame: str
    metric_key: str
    tag: str  # best|median|worst
    value: float


def _select_best_median_worst(
    rows: Iterable[Tuple[Tuple[str, str], Mapping[str, str]]],
    *,
    metric_key: str,
    higher_is_better: bool,
) -> List[SelectedFrame]:
    scored: List[SelectedFrame] = []
    for (seq, fr), row in rows:
        v = _infer_metric_value(row, metric_key)
        if v is None:
            continue
        scored.append(SelectedFrame(sequence=str(seq), frame=str(fr), metric_key=str(metric_key), tag="", value=float(v)))
    if not scored:
        raise ValueError(f"no valid rows for metric_key={metric_key}")

    scored.sort(key=lambda x: x.value, reverse=bool(higher_is_better))
    best = scored[0]
    worst = scored[-1]
    median = scored[len(scored) // 2]
    return [
        SelectedFrame(sequence=best.sequence, frame=best.frame, metric_key=metric_key, tag="best", value=best.value),
        SelectedFrame(sequence=median.sequence, frame=median.frame, metric_key=metric_key, tag="median", value=median.value),
        SelectedFrame(sequence=worst.sequence, frame=worst.frame, metric_key=metric_key, tag="worst", value=worst.value),
    ]


def _baseline_distance_m(*, lidar_root: Path, split: str, meta: FrameMeta) -> float | None:
    """Baseline distance between main and the *other* coop agent (world XYZ)."""

    main_yaml = lidar_root / split / meta.sequence / meta.main_agent / f"{meta.frame}.yaml"
    if not main_yaml.is_file():
        return None
    meta_main = load_frame_metadata(main_yaml)
    x0, y0, z0 = [float(v) for v in meta_main["lidar_pose"][:3]]

    if not meta.coop_agents:
        return None
    other: Optional[str]
    if meta.coop_agents[0] != meta.main_agent:
        other = str(meta.coop_agents[0])
    elif len(meta.coop_agents) > 1:
        other = str(meta.coop_agents[1])
    else:
        other = None
    if other is None:
        return None
    other_yaml = lidar_root / split / meta.sequence / other / f"{meta.frame}.yaml"
    if not other_yaml.is_file():
        return None
    meta_other = load_frame_metadata(other_yaml)
    x1, y1, z1 = [float(v) for v in meta_other["lidar_pose"][:3]]
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _load_batch_eval_module() -> Any:
    batch_eval_path = Path(__file__).resolve().parent / "batch_eval.py"
    spec = importlib.util.spec_from_file_location("_batch_eval_mod_metric_viz_readable", batch_eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load batch_eval module from {batch_eval_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_batch_eval_mod_metric_viz_readable"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _resolve_ckpt(eval_run_dir: Path, *, model_key: str) -> Path:
    summary_json = eval_run_dir / "summary_test.json"
    if not summary_json.is_file():
        candidates = sorted(eval_run_dir.glob("summary_*.json"))
        if not candidates:
            raise FileNotFoundError(f"no summary json under {eval_run_dir}")
        summary_json = candidates[0]
    doc = json.loads(summary_json.read_text(encoding="utf-8"))
    mp = doc.get("model_paths") or {}
    if not isinstance(mp, dict) or model_key not in mp:
        raise KeyError(f"summary json missing model_paths[{model_key}]: {summary_json}")
    ckpt = Path(str(mp[model_key]))
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt} (from {summary_json})")
    return ckpt


def _infer_pred_points(
    *,
    be: Any,
    model: Any,
    model_arch: str,
    norm_type: str,
    images_root: Path,
    depth_root: Path,
    split: str,
    meta: FrameMeta,
    mode: str,
    device: str,
    keep_camera_poses: bool,
    return_groups: bool,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Infer predicted points (ego CARLA) and optionally return per-agent groups.

    Grouping uses the OPV2V batch-eval view naming convention:
      - coop:  cameraX_<agent_id>
      - single: cameraX
    """

    def _parse_agent_from_view_name(name: Any) -> str:
        if not isinstance(name, str) or not name:
            return "__unknown__"
        parts = name.split("_")
        if len(parts) >= 2 and parts[-1]:
            return parts[-1]
        return "__single__"

    info = be.FrameInfo(sequence=meta.sequence, frame=meta.frame, main_agent=meta.main_agent, coop_agents=tuple(meta.coop_agents))
    if mode == "single":
        raw_views, camera_infos = be.build_single_raw(images_root, depth_root, split, info)
    else:
        raw_views, camera_infos = be.build_coop_raw(images_root, depth_root, split, info)

    gt_ref_pose_cv = None
    if raw_views and "camera_poses" in raw_views[0]:
        gt_ref_pose_cv = np.asarray(raw_views[0]["camera_poses"], dtype=np.float32)

    processed = be.preprocess_inputs(raw_views, norm_type=norm_type)
    for view in processed:
        view.pop("depth_z", None)
    if not keep_camera_poses:
        be.strip_external_calibration_inputs(processed)

    with be.torch.no_grad():
        if model_arch == "vggt":
            be._move_views_to_device(processed, be.torch.device(device))
            raw_preds = model(processed)
            preds = be._vggt_preds_to_batch_eval_format(raw_preds, processed)
        else:
            preds = model.infer(processed, memory_efficient_inference=True)

    # Most OPV2V evals provide GT ref pose in raw_views (camera0 in ego); use that to anchor ego-frame visualization.
    if gt_ref_pose_cv is None:
        pts, _ = be.predictions_to_pointcloud(preds, colorize=False)
        pts = be._points_opencv_to_carla(pts)
        return pts.astype(np.float32, copy=False), None

    if not return_groups:
        pts, _ = be.predictions_to_ego_pointcloud(preds, gt_ref_pose_cv, colorize=False)
        pts = be._points_opencv_to_carla(pts)
        return pts.astype(np.float32, copy=False), None

    # Re-implement `predictions_to_ego_pointcloud` but keep per-view groups.
    # NOTE: This can surface coop pose drift more clearly because each agent gets its own color.
    pred_ref_pose = preds[0].get("camera_poses")
    if pred_ref_pose is None:
        pts, _ = be.predictions_to_ego_pointcloud(preds, gt_ref_pose_cv, colorize=False)
        pts = be._points_opencv_to_carla(pts)
        return pts.astype(np.float32, copy=False), None
    pred_ref_pose = pred_ref_pose[0]
    pred_ref_inv = be.torch.linalg.inv(pred_ref_pose)

    gt_ref_pose_cv = np.asarray(gt_ref_pose_cv, dtype=np.float32)
    if gt_ref_pose_cv.shape != (4, 4):
        raise ValueError(f"expected gt_ref_pose_cv 4x4; got {gt_ref_pose_cv.shape}")

    view_names: List[str] = []
    for cam in camera_infos:
        name = cam.get("name") if isinstance(cam, dict) else None
        view_names.append(str(name) if name is not None else "")

    groups: Dict[str, List[np.ndarray]] = {}
    per_view_pts: List[np.ndarray] = []
    for view_idx, pred in enumerate(preds):
        try:
            depth = pred["depth_z"][0].squeeze(-1)
            intr = pred["intrinsics"][0]
            pose = pred["camera_poses"][0]
        except Exception:  # noqa: BLE001
            continue

        rel_pose = pred_ref_inv @ pose
        pts_ref, valid_mask = be.depthmap_to_world_frame(depth, intr, rel_pose)
        pred_mask = pred.get("mask")
        if pred_mask is None:
            final_mask = valid_mask
        else:
            final_mask = valid_mask & pred_mask[0].squeeze(-1).to(dtype=be.torch.bool)
        # Index on-device to avoid moving the dense (H,W,3) tensor to CPU.
        pts_sel = pts_ref[final_mask]
        pts_np = pts_sel.detach().cpu().numpy().astype(np.float32, copy=False)
        if pts_np.size == 0:
            continue

        pts_h = np.concatenate([pts_np, np.ones((pts_np.shape[0], 1), dtype=np.float32)], axis=1)
        pts_ego_cv = (gt_ref_pose_cv @ pts_h.T).T[:, :3].astype(np.float32, copy=False)
        pts_ego = be._points_opencv_to_carla(pts_ego_cv).astype(np.float32, copy=False)

        per_view_pts.append(pts_ego)
        agent = _parse_agent_from_view_name(view_names[view_idx] if view_idx < len(view_names) else "")
        groups.setdefault(agent, []).append(pts_ego)

    pts_all = np.concatenate(per_view_pts, axis=0).astype(np.float32, copy=False) if per_view_pts else np.zeros((0, 3), dtype=np.float32)
    groups_np: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in groups.items() if v}
    return pts_all, (groups_np or None)


def _render_bev_png(
    *,
    out_png: Path,
    title: str,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    pred_groups: Optional[Dict[str, np.ndarray]],
    gt_boxes: np.ndarray,
    note_lines: List[str],
    seed: int,
    max_points_gt: int,
    max_points_pred: int,
    z_min: float,
    z_max: float,
    radius_max: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    pred_color_by: str,
    match_res: float,
) -> None:
    rng = np.random.default_rng(int(seed))

    gt_points = _filter_points(_ensure_nx3(gt_points), z_min=z_min, z_max=z_max, radius_max=radius_max)
    pred_points = _filter_points(_ensure_nx3(pred_points), z_min=z_min, z_max=z_max, radius_max=radius_max)

    gt_points = _downsample(gt_points, int(max_points_gt), rng)
    pred_points = _downsample(pred_points, int(max_points_pred), rng)

    # Some color modes need per-agent groups; when unavailable (e.g., PCD reuse) fall back gracefully.
    color_mode = str(pred_color_by)
    if color_mode in ("agent", "agent_match") and not pred_groups:
        color_mode = "match"

    # If we do have groups, keep them filtered/downsampled too (so that match/agent plots are consistent).
    pred_groups_f: Optional[Dict[str, np.ndarray]] = None
    if pred_groups and color_mode in ("agent", "agent_match"):
        labels = sorted(pred_groups.keys())
        per = max(1, int(max_points_pred) // max(1, len(labels)))
        pred_groups_f = {}
        for lab in labels:
            pts = _filter_points(_ensure_nx3(pred_groups[lab]), z_min=z_min, z_max=z_max, radius_max=radius_max)
            pts = _downsample(pts, int(per), rng)
            if pts.size:
                pred_groups_f[lab] = pts

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    # NOTE: keep GT very light to reduce visual masking; focus is on pred-vs-gt differences.
    if gt_points.size:
        ax.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            s=0.08,
            c="0.35",
            alpha=0.12,
            rasterized=True,
            label="GT LiDAR",
        )

    # Pred points
    if color_mode == "z":
        if pred_points.size:
            zc = np.clip(pred_points[:, 2], -2.5, 2.5)
            ax.scatter(
                pred_points[:, 0],
                pred_points[:, 1],
                s=0.10,
                c=zc,
                cmap="turbo",
                alpha=0.55,
                rasterized=True,
                label="Pred (z)",
            )
    elif color_mode == "match":
        if pred_points.size and gt_points.size:
            match = _bev_match_mask(gt_points=gt_points, pred_points=pred_points, xlim=xlim, ylim=ylim, resolution=float(match_res))
            n = int(pred_points.shape[0])
            n_match = int(match.sum())
            if n > 0:
                note_lines.append(f"bev_match={100.0 * n_match / n:.0f}% (res={float(match_res):.2g}m)")
            if bool(match.any()):
                ax.scatter(
                    pred_points[match, 0],
                    pred_points[match, 1],
                    s=0.11,
                    c="tab:green",
                    alpha=0.32,
                    rasterized=True,
                    label=f"Pred match ({n_match}/{n})",
                )
            if bool((~match).any()):
                ax.scatter(
                    pred_points[~match, 0],
                    pred_points[~match, 1],
                    s=0.11,
                    c="tab:red",
                    alpha=0.36,
                    rasterized=True,
                    label=f"Pred mismatch ({n - n_match}/{n})",
                )
        else:
            # Fallback: no GT points in range (or no pred).
            if pred_points.size:
                ax.scatter(pred_points[:, 0], pred_points[:, 1], s=0.11, c="tab:blue", alpha=0.38, rasterized=True, label="Pred")
    elif color_mode in ("agent", "agent_match"):
        if pred_groups_f:
            cmap = plt.get_cmap("tab10")
            all_pred = np.concatenate(list(pred_groups_f.values()), axis=0) if pred_groups_f else np.zeros((0, 3), dtype=np.float32)
            match_all = None
            if color_mode == "agent_match" and gt_points.size and all_pred.size:
                match_all = _bev_match_mask(gt_points=gt_points, pred_points=all_pred, xlim=xlim, ylim=ylim, resolution=float(match_res))
                n = int(all_pred.shape[0])
                n_match = int(match_all.sum())
                if n > 0:
                    note_lines.append(f"bev_match={100.0 * n_match / n:.0f}% (res={float(match_res):.2g}m)")

            # Plot agent-matched points first (fainter), then mismatches in red (on top).
            offset = 0
            for i, lab in enumerate(sorted(pred_groups_f.keys())):
                pts = pred_groups_f[lab]
                if pts.size == 0:
                    continue
                col = cmap(i % 10)
                if match_all is None:
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        s=0.11,
                        c=[col],
                        alpha=0.40,
                        rasterized=True,
                        label=f"Pred agent {lab}",
                    )
                else:
                    m = match_all[offset : offset + pts.shape[0]]
                    offset += int(pts.shape[0])
                    if bool(m.any()):
                        ax.scatter(
                            pts[m, 0],
                            pts[m, 1],
                            s=0.11,
                            c=[col],
                            alpha=0.28,
                            rasterized=True,
                            label=f"Pred agent {lab}",
                        )
                    if bool((~m).any()):
                        ax.scatter(
                            pts[~m, 0],
                            pts[~m, 1],
                            s=0.12,
                            c="tab:red",
                            alpha=0.40,
                            rasterized=True,
                            label="Pred mismatch" if i == 0 else None,
                        )
            if match_all is not None and offset != int(all_pred.shape[0]):
                # Extremely defensive: if ordering ever changes, don't silently mis-label points.
                note_lines.append("WARN: agent_match offset mismatch")
        else:
            # Fallback: no groups available (e.g. pred_pcd_strategy=reuse).
            if pred_points.size:
                ax.scatter(pred_points[:, 0], pred_points[:, 1], s=0.11, c="tab:blue", alpha=0.38, rasterized=True, label="Pred")
    else:
        raise ValueError(f"unknown pred_color_by mode: {color_mode}")

    _plot_boxes(ax, gt_boxes, color="black", lw=1.3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("+X forward (m)")
    ax.set_ylabel("+Y right (m)")
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, linewidth=0.4, alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    if note_lines:
        ax.text(
            0.01,
            0.01,
            "\n".join(note_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            family="monospace",
            bbox=dict(facecolor="white", edgecolor="0.2", alpha=0.92, boxstyle="round,pad=0.35"),
        )

    out_png = out_png.expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _write_gallery_html(
    out_dir: Path,
    *,
    models: List[str],
    rows: List[Dict[str, Any]],
    plots: Dict[str, str],
    legend_lines: Sequence[str],
) -> Path:
    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def esc_attr(s: str) -> str:
        # HTML attribute escape + keep newlines readable in tooltips.
        return esc(s).replace("\n", "&#10;")

    css = """
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
    code { background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 4px; }
    .plots img { max-width: 1100px; width: 100%; border: 1px solid #ddd; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
    th { position: sticky; top: 0; background: #fafafa; }
    .framecell { min-width: 320px; }
    .tile img { width: 420px; max-width: 420px; border: 1px solid #ccc; }
    .tilecap { margin-top: 4px; font-size: 12px; line-height: 1.15; white-space: pre-line; }
    .badges { margin-top: 4px; margin-bottom: 4px; }
    .badge { display: inline-block; padding: 0.12rem 0.45rem; border-radius: 999px; font-size: 11px; font-weight: 700; border: 1px solid transparent; margin-right: 0.35rem; }
    .badge.best { background: #dcfce7; color: #166534; border-color: #86efac; }
    .badge.median { background: #e5e7eb; color: #374151; border-color: #d1d5db; }
    .badge.worst { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
    .muted { color: #555; font-size: 12px; }
    """

    plot_html = []
    if plots:
        plot_html.append("<h2>Distributions</h2>")
        plot_html.append("<div class='plots'>")
        for title, rel in plots.items():
            plot_html.append(f"<h3>{esc(title)}</h3>")
            plot_html.append(f"<img src='{esc(rel)}' loading='lazy' />")
        plot_html.append("</div>")

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>OPV2V Readable Metric Viz</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>OPV2V Readable Metric Viz</h1>",
        "<p class='muted'>Rows are selected frames; columns are models. Tiles include on-image annotations.</p>",
        "<details open><summary><b>How to read tiles</b></summary>",
        "<div class='muted'><ul>" + "".join([f"<li>{esc(x)}</li>" for x in legend_lines]) + "</ul></div>",
        "</details>",
        *plot_html,
        "<h2>BEV Gallery</h2>",
        "<table>",
        "<tr><th>Frame</th>" + "".join([f"<th>{esc(m)}</th>" for m in models]) + "</tr>",
    ]

    for r in rows:
        seq = r["sequence"]
        fr = r["frame"]
        reasons = r.get("reasons") or []
        reason_str = ", ".join(reasons)
        # Parse selection tag badges from "metric:tag (value)" reasons.
        tag_order = {"best": 0, "median": 1, "worst": 2}
        tags = []
        for reason in reasons:
            if not isinstance(reason, str) or ":" not in reason:
                continue
            _metric, rest = reason.split(":", 1)
            tag = rest.split(" ", 1)[0].strip() if rest else ""
            if tag in tag_order:
                tags.append(tag)
        tags_sorted = sorted(set(tags), key=lambda t: tag_order.get(t, 9))
        badges_html = ""
        if tags_sorted:
            badges_html = "<div class='badges'>" + "".join([f"<span class='badge {t}'>{esc(t)}</span>" for t in tags_sorted]) + "</div>"
        row_html = [
            "<tr>",
            "<td class='framecell'>",
            f"<div><code>{esc(seq)}/{esc(fr)}</code></div>",
            badges_html,
            f"<div class='muted'>{esc(reason_str)}</div>" if reason_str else "",
            "</td>",
        ]
        for m in models:
            rel = (r.get("tiles") or {}).get(m)
            if rel:
                notes = ((r.get("tile_notes") or {}).get(m) or [])
                note_text = "\n".join([str(x) for x in notes]) if isinstance(notes, list) else str(notes)
                title_attr = f" title='{esc_attr(note_text)}'" if note_text else ""
                row_html.append("<td class='tile'>")
                row_html.append(f"<img src='{esc(rel)}' loading='lazy'{title_attr} />")
                if note_text:
                    row_html.append(f"<div class='tilecap muted'>{esc(note_text)}</div>")
                row_html.append("</td>")
            else:
                row_html.append("<td class='tile'><div class='muted'>missing</div></td>")
        row_html.append("</tr>")
        lines.append("".join(row_html))

    lines.extend(["</table>", "</body></html>"])

    out_path = out_dir / "index.html"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _write_distribution_plots(
    out_dir: Path,
    *,
    models: List[str],
    metrics: Dict[str, Dict[Tuple[str, str], Dict[str, str]]],
    frames: Dict[Tuple[str, str], FrameMeta],
    lidar_root: Path,
    split: str,
    seed: int,
) -> Dict[str, str]:
    _ = seed

    # Prepare per-model vectors.
    per_model_ratio: Dict[str, List[float]] = {}
    per_model_cross_trans: Dict[str, List[float]] = {}
    per_model_pairs: Dict[str, List[Tuple[float, float, float]]] = {}  # (baseline, ratio, cross_trans)

    for name in models:
        rows = metrics[name]
        ratios: List[float] = []
        cross_ts: List[float] = []
        pairs: List[Tuple[float, float, float]] = []
        for key, row in rows.items():
            ratio = _safe_float(row.get("scale_to_gt_ratio_mean"))
            cross_t = _safe_float(row.get("cross_agent_pose_trans_m"))
            if ratio is not None:
                ratios.append(float(ratio))
            if cross_t is not None:
                cross_ts.append(float(cross_t))
            meta = frames.get(key)
            if meta is None:
                continue
            baseline = _baseline_distance_m(lidar_root=lidar_root, split=split, meta=meta)
            if baseline is None or ratio is None or cross_t is None:
                continue
            pairs.append((float(baseline), float(ratio), float(cross_t)))
        per_model_ratio[name] = ratios
        per_model_cross_trans[name] = cross_ts
        per_model_pairs[name] = pairs

    plots: Dict[str, str] = {}
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    for name in models:
        xs = [x for x in per_model_ratio[name] if math.isfinite(x)]
        if xs:
            ax.hist(xs, bins=60, histtype="step", linewidth=2.0, alpha=0.9, label=name)
    ax.set_title("Per-frame scale_to_gt_ratio_mean (s/g) histogram", fontsize=11)
    ax.set_xlabel("scale_to_gt_ratio_mean (ideal=1)")
    ax.set_ylabel("frames")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(fontsize=8)
    p = plots_dir / "hist_scale_to_gt_ratio_mean.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    plots["Histogram: scale_to_gt_ratio_mean"] = str(p.relative_to(out_dir).as_posix())

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    for name in models:
        xs = [x for x in per_model_cross_trans[name] if math.isfinite(x)]
        if xs:
            ax.hist(xs, bins=60, histtype="step", linewidth=2.0, alpha=0.9, label=name)
    ax.set_title("Per-frame cross_agent_pose_trans_m histogram", fontsize=11)
    ax.set_xlabel("cross_agent_pose_trans_m (m) (lower is better)")
    ax.set_ylabel("frames")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(fontsize=8)
    p = plots_dir / "hist_cross_agent_pose_trans_m.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    plots["Histogram: cross_agent_pose_trans_m"] = str(p.relative_to(out_dir).as_posix())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=160)
    for name in models:
        pairs = per_model_pairs[name]
        if not pairs:
            continue
        b = np.asarray([x[0] for x in pairs], dtype=np.float32)
        r = np.asarray([x[1] for x in pairs], dtype=np.float32)
        ct = np.asarray([x[2] for x in pairs], dtype=np.float32)
        axes[0].scatter(b, r, s=10, alpha=0.35, label=name)
        axes[1].scatter(b, ct, s=10, alpha=0.35, label=name)
    axes[0].set_title("Baseline distance vs scale_to_gt_ratio_mean", fontsize=10)
    axes[0].set_xlabel("baseline distance (m)")
    axes[0].set_ylabel("scale_to_gt_ratio_mean (ideal=1)")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[1].set_title("Baseline distance vs cross_agent_pose_trans_m", fontsize=10)
    axes[1].set_xlabel("baseline distance (m)")
    axes[1].set_ylabel("cross_agent_pose_trans_m (m)")
    axes[1].grid(True, linewidth=0.4, alpha=0.35)
    axes[0].legend(fontsize=8)
    p = plots_dir / "scatter_baseline_vs_scale_ratio_and_cross_trans.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    plots["Scatter: baseline vs (scale ratio, cross trans)"] = str(p.relative_to(out_dir).as_posix())

    return plots


@dataclass(frozen=True)
class ModelRun:
    name: str
    eval_run_dir: Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Repeated entries like name=/path/to/eval_run_dir")
    ap.add_argument("--model_key", type=str, default="geom_model")
    ap.add_argument("--mode", type=str, default="coop", choices=("single", "coop"))
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["scale_to_gt_mult_err", "cross_agent_pose_trans_m", "cross_agent_pose_rot_deg", "depth_rel"],
    )
    ap.add_argument("--metric_higher_is_better", nargs="*", default=[])
    ap.add_argument("--selector_model", type=str, default=None, help="Model name used for selecting frames (default: first)")
    ap.add_argument("--out_dir", type=Path, required=True)

    # GT assets
    ap.add_argument("--lidar_root", type=Path, default=REPO_ROOT / "data" / "opv2v")
    ap.add_argument("--split", type=str, default=None, help="Override split (default: read from selector summary JSON)")
    ap.add_argument("--gt_lidar_mode", type=str, default="fused", choices=("main", "fused"))
    ap.add_argument("--gt_bbox_range", type=float, default=120.0)
    ap.add_argument("--gt_max_boxes", type=int, default=128)

    # Rendering
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_points_gt", type=int, default=120_000)
    ap.add_argument("--max_points_pred", type=int, default=100_000)
    ap.add_argument("--z_min", type=float, default=-3.0)
    ap.add_argument("--z_max", type=float, default=3.0)
    ap.add_argument("--radius_max", type=float, default=60.0)
    ap.add_argument("--xlim", type=float, nargs=2, default=(-20.0, 80.0))
    ap.add_argument("--ylim", type=float, nargs=2, default=(-60.0, 60.0))
    ap.add_argument(
        "--pred_color_by",
        type=str,
        default="auto",
        choices=("auto", "match", "agent_match", "agent", "z"),
        help=(
            "How to color predicted points. "
            "'match' colors pred points by BEV cell overlap with GT (green/red). "
            "'agent' colors pred points by agent id (coop only). "
            "'agent_match' combines agent colors (matches) + red mismatches. "
            "'z' colors by height (turbo). "
            "'auto' defaults to agent_match for coop and match for single."
        ),
    )
    ap.add_argument("--match_res", type=float, default=0.5, help="BEV resolution (m) used for match-based coloring.")

    # Pred PCD source
    ap.add_argument("--pred_pcd_strategy", type=str, default="infer", choices=("infer", "reuse"))
    ap.add_argument("--reuse_pcd_root", type=Path, default=None)
    ap.add_argument("--skip_existing", action="store_true", help="Skip rendering tiles if PNG already exists.")

    # Inference (only for --pred_pcd_strategy=infer)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--keep_camera_poses", action="store_true")
    ap.add_argument("--images_root", type=Path, default=None)
    ap.add_argument("--depth_root", type=Path, default=None)
    ap.add_argument("--override_model_arch", type=str, default=None, choices=(None, "mapanything", "vggt"))
    ap.add_argument("--override_model_task", type=str, default=None)
    ap.add_argument("--override_norm_type", type=str, default=None, help="dinov2|identity")

    ap.add_argument("--write_plots", action="store_true")
    args = ap.parse_args()

    # Parse models.
    runs: List[ModelRun] = []
    for item in args.models:
        if "=" not in item:
            raise SystemExit(f"bad --models entry (missing '='): {item}")
        name, p = item.split("=", 1)
        runs.append(ModelRun(name=name.strip(), eval_run_dir=Path(p.strip()).expanduser().resolve()))
    if not runs:
        raise SystemExit("no models provided")

    selector = args.selector_model or runs[0].name
    if selector not in {r.name for r in runs}:
        raise SystemExit(f"--selector_model {selector} not in models {[r.name for r in runs]}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_key = str(args.model_key)
    mode = str(args.mode)
    pred_color_by = str(args.pred_color_by)
    if pred_color_by == "auto":
        pred_color_by = "agent_match" if mode == "coop" else "match"
    need_pred_groups = pred_color_by in ("agent", "agent_match")

    # Load frame metadata from selector run.
    selector_run = next(r for r in runs if r.name == selector)
    selector_summary = selector_run.eval_run_dir / "summary_test.json"
    if not selector_summary.is_file():
        candidates = sorted(selector_run.eval_run_dir.glob("summary_*.json"))
        if not candidates:
            raise FileNotFoundError(f"no summary json under {selector_run.eval_run_dir}")
        selector_summary = candidates[0]
    selector_doc = json.loads(selector_summary.read_text(encoding="utf-8"))
    split = args.split or str(selector_doc.get("split") or "")
    if not split:
        raise SystemExit(f"could not determine split from {selector_summary}; pass --split")
    frames = _load_frames_from_summary(selector_summary)
    if not frames:
        raise SystemExit(f"no frames found in {selector_summary}")

    # Load per-model metrics.
    per_model_rows: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = {}
    for r in runs:
        csv_path = r.eval_run_dir / model_key / f"{mode}_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing metrics csv: {csv_path}")
        per_model_rows[r.name] = _load_metrics_csv(csv_path)

    # Select frames using selector model metrics.
    selector_rows = per_model_rows[selector]
    all_selected: List[SelectedFrame] = []
    higher_set = {str(x) for x in (args.metric_higher_is_better or [])}
    for mk in args.metrics:
        all_selected.extend(
            _select_best_median_worst(
                selector_rows.items(),
                metric_key=str(mk),
                higher_is_better=(str(mk) in higher_set),
            )
        )

    # Group reasons by frame.
    frame_reasons: Dict[Tuple[str, str], List[str]] = {}
    for sel in all_selected:
        k = (sel.sequence, sel.frame)
        frame_reasons.setdefault(k, []).append(f"{sel.metric_key}:{sel.tag} ({sel.value:.4g})")

    # Stable ordering: by metric list order, then best/median/worst.
    tag_order = {"best": 0, "median": 1, "worst": 2}
    metric_order = {m: i for i, m in enumerate([str(x) for x in args.metrics])}

    def _frame_sort_key(k: Tuple[str, str]) -> Tuple[int, int, str, str]:
        reasons = frame_reasons.get(k, [])
        if not reasons:
            return (999, 999, k[0], k[1])
        m0, rest = reasons[0].split(":", 1)
        t0 = rest.split(" ", 1)[0] if rest else "median"
        return (metric_order.get(m0, 999), tag_order.get(t0, 9), k[0], k[1])

    selected_keys = sorted(frame_reasons.keys(), key=_frame_sort_key)

    # Optional inference setup (init models once).
    be = None
    models_infer: Dict[str, Any] = {}
    model_arch_by_name: Dict[str, str] = {}
    norm_type_by_name: Dict[str, str] = {}
    images_root_by_name: Dict[str, Path] = {}
    depth_root_by_name: Dict[str, Path] = {}

    if str(args.pred_pcd_strategy) == "infer":
        be = _load_batch_eval_module()
        for r in runs:
            summ = r.eval_run_dir / "summary_test.json"
            if not summ.is_file():
                candidates = sorted(r.eval_run_dir.glob("summary_*.json"))
                if not candidates:
                    raise FileNotFoundError(f"no summary json under {r.eval_run_dir}")
                summ = candidates[0]
            doc = json.loads(summ.read_text(encoding="utf-8"))
            meta = doc.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}

            model_arch = str(args.override_model_arch or meta.get("model_arch") or "mapanything")
            model_task = str(args.override_model_task or meta.get("model_task") or "calibrated_sfm")
            data_norm_type = str(args.override_norm_type or meta.get("data_norm_type") or "").strip()
            if not data_norm_type:
                data_norm_type = "identity" if model_arch == "vggt" else "dinov2"

            images_root = Path(args.images_root or meta.get("images_root") or (REPO_ROOT / "data" / "opv2v")).expanduser().resolve()
            depth_root = Path(args.depth_root or meta.get("depth_root") or (REPO_ROOT / "data" / "opv2v_depth")).expanduser().resolve()

            ckpt = _resolve_ckpt(r.eval_run_dir, model_key=model_key)

            if model_arch == "vggt":
                cfg_overrides = [
                    "machine=local_a800",
                    "dataset=opv2v_ft_a800",
                    "model=vggt",
                    "model.model_config.load_pretrained_weights=false",
                    f"model.model_config.enable_metric_scale_head={'true' if bool(meta.get('vggt_enable_metric_scale_head')) else 'false'}",
                    f"model.model_config.autocast_dtype={meta.get('vggt_autocast_dtype')}" if meta.get("vggt_autocast_dtype") else None,
                    "loss=overall_loss",
                ]
            else:
                cfg_overrides = [
                    "machine=local_a800",
                    "dataset=opv2v_ft_a800",
                    "model=mapanything",
                    f"model/task={model_task}",
                    "model.encoder.uses_torch_hub=false",
                    "loss=overall_loss",
                ]

            cfg = {
                "path": str((be.REPO_ROOT / "configs/train.yaml").resolve()),
                "checkpoint_path": str(ckpt),
                "config_overrides": [x for x in cfg_overrides if x],
            }
            model = be.initialize_mapanything_local(cfg, be.torch.device(str(args.device)))
            model.eval()

            models_infer[r.name] = model
            model_arch_by_name[r.name] = model_arch
            norm_type_by_name[r.name] = str(data_norm_type)
            images_root_by_name[r.name] = images_root
            depth_root_by_name[r.name] = depth_root

    # Render tiles.
    tiles_by_frame: List[Dict[str, Any]] = []
    png_dir = out_dir / "bev_png"
    pcd_dir = out_dir / "pred_pcd"

    for (seq, fr) in selected_keys:
        meta = frames.get((seq, fr))
        if meta is None:
            continue

        row_out: Dict[str, Any] = {
            "sequence": seq,
            "frame": fr,
            "reasons": frame_reasons.get((seq, fr), []),
            "tiles": {},
            "tile_notes": {},
        }

        gt_points = _load_gt_lidar_points(
            lidar_root=args.lidar_root,
            split=split,
            sequence=seq,
            frame=fr,
            main_agent=meta.main_agent,
            coop_agents=meta.coop_agents,
            mode=str(args.gt_lidar_mode),
        )
        gt_boxes = _load_gt_boxes_ego(
            lidar_root=args.lidar_root,
            split=split,
            sequence=seq,
            frame=fr,
            main_agent=meta.main_agent,
            bbox_range=float(args.gt_bbox_range),
            max_boxes=int(args.gt_max_boxes),
        )

        for r in runs:
            out_png = png_dir / f"{seq}_{fr}" / f"{r.name}.png"

            # Tile annotation from metrics CSV (also used by HTML captions/tooltips).
            row = per_model_rows[r.name].get((seq, fr), {})
            pose_abs = _safe_float(row.get("pose_abs_m"))
            pose_rot = _safe_float(row.get("pose_rot_deg"))
            scale_to_gt_err = _safe_float(row.get("scale_to_gt_err"))
            mult_err = _infer_metric_value(row, "scale_to_gt_mult_err")
            ratio_mean = _safe_float(row.get("scale_to_gt_ratio_mean"))
            cross_t = _safe_float(row.get("cross_agent_pose_trans_m"))
            cross_r = _safe_float(row.get("cross_agent_pose_rot_deg"))
            depth_rel = _safe_float(row.get("depth_rel"))
            chamfer_f = _safe_float(row.get("chamfer_filtered_pred_to_gt"))
            bev_iou_f = _safe_float(row.get("bev_iou_filtered"))
            baseline = _baseline_distance_m(lidar_root=args.lidar_root, split=split, meta=meta)

            note: List[str] = []
            # Make the selection reason visible on-tile (frame-level, same across model columns).
            reasons = frame_reasons.get((seq, fr), [])
            if reasons:
                # Format "metric:tag (value)" -> "sel=metric/tag"
                try:
                    metric, rest = str(reasons[0]).split(":", 1)
                    tag = rest.split(" ", 1)[0].strip() if rest else ""
                    if metric and tag:
                        note.append(f"sel={metric}/{tag}")
                except Exception:
                    pass
            if pose_abs is not None:
                note.append(f"pose_abs={pose_abs:.3g}m")
            if pose_rot is not None:
                note.append(f"pose_rot={pose_rot:.2f}deg")
            if scale_to_gt_err is not None:
                note.append(f"scale_abs={scale_to_gt_err:.3g}")
            if mult_err is not None:
                note.append(f"scale_mult={mult_err:.3g}")
            if ratio_mean is not None:
                note.append(f"ratio_gt={ratio_mean:.3g}")
            if cross_t is not None:
                note.append(f"cross_t={cross_t:.3g}m")
            if cross_r is not None:
                note.append(f"cross_r={cross_r:.2f}deg")
            if depth_rel is not None:
                note.append(f"depth_rel={depth_rel:.3g}")
            if chamfer_f is not None:
                note.append(f"chamfer_f={chamfer_f:.3g}")
            if bev_iou_f is not None:
                note.append(f"bev_iou_f={bev_iou_f:.3g}")
            if baseline is not None:
                note.append(f"baseline={baseline:.1f}m")
            row_out["tile_notes"][r.name] = note

            if args.skip_existing and out_png.is_file():
                row_out["tiles"][r.name] = str(out_png.relative_to(out_dir).as_posix())
                continue

            pred_points: Optional[np.ndarray] = None
            pred_groups: Optional[Dict[str, np.ndarray]] = None
            if str(args.pred_pcd_strategy) == "reuse":
                if args.reuse_pcd_root is None:
                    raise SystemExit("--reuse_pcd_root is required when --pred_pcd_strategy=reuse")
                model_root = (args.reuse_pcd_root.expanduser().resolve() / r.name)
                if model_root.is_dir():
                    direct = model_root / f"{seq}_{fr}.pcd"
                    if direct.is_file():
                        pred_points = _ensure_nx3(load_ascii_pcd_xyz(direct))
                    else:
                        # Back-compat: some older exporters used suffixes like *_best.pcd.
                        matches = sorted(model_root.rglob(f"{seq}_{fr}_*.pcd"))
                        if matches:
                            pred_points = _ensure_nx3(load_ascii_pcd_xyz(matches[0]))
            else:
                assert be is not None
                pred_points, pred_groups = _infer_pred_points(
                    be=be,
                    model=models_infer[r.name],
                    model_arch=model_arch_by_name[r.name],
                    norm_type=norm_type_by_name[r.name],
                    images_root=images_root_by_name[r.name],
                    depth_root=depth_root_by_name[r.name],
                    split=split,
                    meta=meta,
                    mode=mode,
                    device=str(args.device),
                    keep_camera_poses=bool(args.keep_camera_poses),
                    return_groups=bool(need_pred_groups),
                )
                out_pcd = pcd_dir / r.name / f"{seq}_{fr}.pcd"
                out_pcd.parent.mkdir(parents=True, exist_ok=True)
                be.save_point_cloud(out_pcd, pred_points)

            if pred_points is None:
                continue

            _render_bev_png(
                out_png=out_png,
                title=f"{r.name} | {seq}/{fr}",
                gt_points=gt_points,
                pred_points=pred_points,
                pred_groups=pred_groups,
                gt_boxes=gt_boxes,
                note_lines=note,
                seed=int(args.seed),
                max_points_gt=int(args.max_points_gt),
                max_points_pred=int(args.max_points_pred),
                z_min=float(args.z_min),
                z_max=float(args.z_max),
                radius_max=float(args.radius_max),
                xlim=(float(args.xlim[0]), float(args.xlim[1])),
                ylim=(float(args.ylim[0]), float(args.ylim[1])),
                pred_color_by=str(pred_color_by),
                match_res=float(args.match_res),
            )
            row_out["tiles"][r.name] = str(out_png.relative_to(out_dir).as_posix())

        tiles_by_frame.append(row_out)

    # Selection metadata (for docs / reproducibility).
    _dump_json(
        out_dir / "selection.json",
        {
            "selector_model": selector,
            "model_key": model_key,
            "mode": mode,
            "metrics": [str(x) for x in args.metrics],
            "metric_higher_is_better": sorted(higher_set),
            "models": [{"name": r.name, "eval_run_dir": str(r.eval_run_dir)} for r in runs],
            "selected_frames": [
                {"sequence": seq, "frame": fr, "reasons": frame_reasons.get((seq, fr), [])} for (seq, fr) in selected_keys
            ],
            "gt": {
                "lidar_root": str(args.lidar_root),
                "split": split,
                "gt_lidar_mode": str(args.gt_lidar_mode),
            },
            "render": {
                "radius_max": float(args.radius_max),
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
                "xlim": [float(args.xlim[0]), float(args.xlim[1])],
                "ylim": [float(args.ylim[0]), float(args.ylim[1])],
                "pred_color_by": str(pred_color_by),
                "match_res": float(args.match_res),
            },
        },
    )

    plot_rels: Dict[str, str] = {}
    if args.write_plots:
        plot_rels = _write_distribution_plots(
            out_dir,
            models=[r.name for r in runs],
            metrics=per_model_rows,
            frames=frames,
            lidar_root=args.lidar_root,
            split=split,
            seed=int(args.seed),
        )

    legend_lines: List[str] = []
    legend_lines.append("GT LiDAR: light gray points (in main-agent ego frame).")
    legend_lines.append("GT boxes: black outlines.")
    if pred_color_by == "z":
        legend_lines.append("Pred points: colored by height z (turbo).")
    elif pred_color_by == "match":
        legend_lines.append("Pred points: green=BEV cell overlaps GT, red=does not.")
        legend_lines.append(f"bev_match: overlap ratio on a {float(args.match_res):.2g}m BEV grid (after filtering/downsampling).")
    elif pred_color_by == "agent":
        legend_lines.append("Pred points: colored by agent id (coop only).")
    elif pred_color_by == "agent_match":
        legend_lines.append("Pred points: colored by agent id (matches); red=mismatches.")
        legend_lines.append(f"bev_match: overlap ratio on a {float(args.match_res):.2g}m BEV grid (after filtering/downsampling).")
    else:
        legend_lines.append(f"Pred points: pred_color_by={pred_color_by}")

    idx = _write_gallery_html(
        out_dir,
        models=[r.name for r in runs],
        rows=tiles_by_frame,
        plots=plot_rels,
        legend_lines=legend_lines,
    )
    print(f"[OK] wrote gallery: {idx}")


if __name__ == "__main__":
    main()
