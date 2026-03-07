#!/usr/bin/env python3
"""Create a *readable* 2D BEV PNG gallery from existing representative PCDs.

This is intended as a companion to `scripts/metric_viz_from_eval_csv.py` +
`scripts/make_batch_eval_pcd_html.py`:

- Those scripts generate representative predicted point clouds (`*.pcd`) for
  best/median/worst frames by a chosen metric.
- This script turns those PCDs into simple BEV PNG overlays against OPV2V GT
  (fused LiDAR + GT boxes), and writes an offline `index.html` gallery.

Why
---
The 3D HTML point cloud view is powerful but can be visually noisy. A 2D BEV
plot (GT LiDAR in gray, predictions in color, GT boxes in black) is usually
faster to interpret for "cross-agent alignment" and "scale" failures.

Inputs
------
Point this at a metric-viz directory that contains:
  - `<model_name>/selected_*.json` (written by `metric_viz_from_eval_csv.py`)
  - `<model_name>/*_representatives/*.pcd`

Example
-------
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/make_metric_viz_bev_gallery_from_pcd.py \
    --metric_viz_root eval_runs/metric_viz_test500_20260226_all_lines \
    --frames_contract_json eval_runs/frames_test500_seed42.json \
    --images_root data/opv2v \
    --split test \
    --out_dir eval_runs/metric_viz_test500_20260226_all_lines/bev_png
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing.opv2v_pose_utils import cords_to_pose, load_frame_metadata  # noqa: E402
from mapanything.datasets.opv2v import _extract_vehicle_boxes_in_ego  # noqa: E402


def _slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        pts = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return pts


def _transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    pts_h = np.concatenate(
        [points.astype(np.float64, copy=False), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    pts_out = (T_dst_src @ pts_h.T).T[:, :3]
    return pts_out.astype(np.float32)


def _filter_points(points: np.ndarray, *, z_min: float | None, z_max: float | None, radius_max: float | None) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= points[:, 2] <= float(z_max)
    if radius_max is not None and float(radius_max) > 0:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= float(radius_max)
    return points[mask]


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _load_gt_lidar_points_fused(
    *,
    images_root: Path,
    split: str,
    sequence: str,
    frame: str,
    main_agent: str,
    coop_agents: Iterable[str],
) -> np.ndarray:
    images_root = images_root.expanduser().resolve()
    main_yaml = images_root / split / sequence / main_agent / f"{frame}.yaml"
    if not main_yaml.is_file():
        raise FileNotFoundError(main_yaml)

    meta_main = load_frame_metadata(main_yaml)
    T_world_main = cords_to_pose(meta_main["lidar_pose"])
    T_main_world = np.linalg.inv(T_world_main)

    selected_agents = sorted(set([str(main_agent), *[str(a) for a in coop_agents]]))

    all_points: List[np.ndarray] = []
    for agent in selected_agents:
        pcd_path = images_root / split / sequence / agent / f"{frame}.pcd"
        yaml_path = images_root / split / sequence / agent / f"{frame}.yaml"
        if not pcd_path.is_file() or not yaml_path.is_file():
            continue
        pts_agent = _load_ascii_pcd_xyz(pcd_path)
        if agent == str(main_agent):
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


def _load_gt_boxes_in_ego(yaml_path: Path, *, max_range: float, max_num_boxes: int) -> np.ndarray:
    meta = load_frame_metadata(yaml_path)
    gt_padded, gt_mask = _extract_vehicle_boxes_in_ego(meta, max_range=float(max_range), max_num_boxes=int(max_num_boxes))
    return gt_padded[gt_mask]


def _box_to_bev_poly(box: np.ndarray) -> np.ndarray:
    # box: (x, y, z, l, w, h, yaw) in ego (CARLA) coords, yaw in radians.
    x, y, _z, l, w, _h, yaw = [float(v) for v in box]
    dx = l / 2.0
    dy = w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _plot_boxes(ax, boxes: Iterable[np.ndarray], *, color: str, lw: float, alpha: float) -> None:
    for box in boxes:
        poly = _box_to_bev_poly(box)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=lw, alpha=float(alpha))


@dataclass(frozen=True)
class FrameContract:
    main_agent: str
    coop_agents: Tuple[str, ...]


def _load_frame_contract(frames_contract_json: Path) -> Dict[Tuple[str, str], FrameContract]:
    doc = json.loads(frames_contract_json.read_text(encoding="utf-8"))
    frames = doc.get("frames") or []
    if not isinstance(frames, list):
        raise ValueError(f"Bad frames_contract_json (missing list frames): {frames_contract_json}")
    out: Dict[Tuple[str, str], FrameContract] = {}
    for fr in frames:
        if not isinstance(fr, Mapping):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        if not isinstance(seq, str) or not isinstance(frame, str):
            continue
        main = str(fr.get("main_agent") or "641")
        coop = fr.get("coop_agents") or [main]
        if isinstance(coop, (list, tuple)):
            coop_ids = tuple(str(a) for a in coop)
        else:
            coop_ids = (main,)
        out[(seq, frame)] = FrameContract(main_agent=main, coop_agents=coop_ids)
    return out


def _load_csv_lookup(csv_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not csv_path.is_file():
        return {}
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            seq = row.get("sequence")
            frame = row.get("frame")
            if not seq or not frame:
                continue
            out[(str(seq), str(frame))] = dict(row)
    return out


def _resolve_existing_path(path_str: str, *, metric_viz_root: Path) -> Path | None:
    """Best-effort resolver for paths stored inside `selected_*.json`.

    Historically, some selection JSONs store:
      - absolute paths, OR
      - workspace-relative paths like `map-anything/eval_runs/...`, OR
      - repo-relative paths like `eval_runs/...`

    We try a small ordered set of bases and return the first existing file.
    """

    p = Path(path_str)
    if p.is_absolute():
        return p if p.is_file() else None

    bases = [
        # Most common: selection JSON stores a workspace-relative `map-anything/...` path.
        REPO_ROOT.parent,
        # Sometimes: repo-relative paths.
        REPO_ROOT,
        # Sometimes: metric_viz_root-relative (if author wrote relative paths).
        metric_viz_root,
        # Last resort: current working directory.
        Path.cwd(),
    ]
    for b in bases:
        cand = (b / p).resolve()
        if cand.is_file():
            return cand
    return None


def _render_bev_png(
    *,
    out_png: Path,
    pred_pcd: Path,
    yaml_main: Path,
    gt_points: np.ndarray,
    gt_boxes: np.ndarray,
    title: str,
    rng: np.random.Generator,
    max_pred_points: int,
    max_gt_points: int,
    z_min: float | None,
    z_max: float | None,
    radius_max: float | None,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    text_lines: List[str] | None,
) -> None:
    pred_pts = _load_ascii_pcd_xyz(pred_pcd)
    pred_pts = _filter_points(pred_pts, z_min=z_min, z_max=z_max, radius_max=radius_max)
    gt_pts = _filter_points(gt_points, z_min=z_min, z_max=z_max, radius_max=radius_max)
    pred_pts = _downsample(pred_pts, int(max_pred_points), rng)
    gt_pts = _downsample(gt_pts, int(max_gt_points), rng)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=170)
    if gt_pts.size:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=0.12, c="#9E9E9E", alpha=0.18, rasterized=True, label="GT LiDAR (fused)")
    if pred_pts.size:
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=0.18, c="tab:blue", alpha=0.30, rasterized=True, label="Pred (ego)")
    if gt_boxes.size:
        _plot_boxes(ax, gt_boxes, color="black", lw=1.0, alpha=0.85)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_xlabel("+X forward (m)")
    ax.set_ylabel("+Y right (m)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)

    if text_lines:
        ax.text(
            0.01,
            0.01,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0.0),
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric_viz_root", type=Path, required=True)
    ap.add_argument("--frames_contract_json", type=Path, required=True)
    ap.add_argument("--images_root", type=Path, default=REPO_ROOT / "data" / "opv2v")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")

    # Plot controls
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pred_points", type=int, default=120_000)
    ap.add_argument("--max_gt_points", type=int, default=140_000)
    ap.add_argument("--z_min", type=float, default=-3.0)
    ap.add_argument("--z_max", type=float, default=3.0)
    ap.add_argument("--radius_max", type=float, default=80.0)
    ap.add_argument("--gt_bbox_range", type=float, default=120.0)
    ap.add_argument("--gt_max_boxes", type=int, default=128)
    ap.add_argument("--x_min", type=float, default=-20.0)
    ap.add_argument("--x_max", type=float, default=120.0)
    ap.add_argument("--y_min", type=float, default=-50.0)
    ap.add_argument("--y_max", type=float, default=50.0)
    args = ap.parse_args()

    metric_viz_root = args.metric_viz_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    frames_contract = _load_frame_contract(args.frames_contract_json.expanduser().resolve())

    model_dirs = sorted([p for p in metric_viz_root.iterdir() if p.is_dir()])
    model_dirs = [p for p in model_dirs if list(p.glob("selected_*.json"))]
    if not model_dirs:
        raise SystemExit(f"No model dirs with selected_*.json found under {metric_viz_root}")

    rng = np.random.default_rng(int(args.seed))

    index_items: List[Dict[str, str]] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        selected_jsons = sorted(model_dir.glob("selected_*.json"))
        for sel_path in selected_jsons:
            sel = json.loads(sel_path.read_text(encoding="utf-8"))
            mode = str(sel.get("mode") or "coop")
            metric_key = str(sel.get("metric_key") or sel_path.stem.replace("selected_", ""))
            metric_slug = _slug(metric_key)

            eval_run_dir = sel.get("eval_run_dir")
            model_key = str(sel.get("model_key") or "geom_model")
            csv_lookup = {}
            if isinstance(eval_run_dir, str):
                csv_path = Path(eval_run_dir) / model_key / f"{mode}_metrics.csv"
                csv_lookup = _load_csv_lookup(csv_path)

            outputs = sel.get("outputs") or {}
            if not isinstance(outputs, Mapping):
                continue

            for tag, item in outputs.items():
                if not isinstance(item, Mapping):
                    continue
                seq = str(item.get("sequence") or "")
                frame = str(item.get("frame") or "")
                pcd = item.get("pcd")
                if not (seq and frame and isinstance(pcd, str)):
                    continue
                pred_pcd = _resolve_existing_path(pcd, metric_viz_root=metric_viz_root)
                if pred_pcd is None or not pred_pcd.is_file():
                    continue

                contract = frames_contract.get((seq, frame))
                if contract is None:
                    continue
                main_agent = contract.main_agent
                coop_agents = contract.coop_agents
                yaml_main = args.images_root.expanduser().resolve() / args.split / seq / main_agent / f"{frame}.yaml"
                if not yaml_main.is_file():
                    continue

                out_png = out_dir / model_name / mode / metric_slug / f"{seq}_{frame}_{tag}.png"
                if out_png.is_file() and not args.overwrite:
                    pass
                else:
                    gt_points = _load_gt_lidar_points_fused(
                        images_root=args.images_root,
                        split=args.split,
                        sequence=seq,
                        frame=frame,
                        main_agent=main_agent,
                        coop_agents=coop_agents,
                    )
                    gt_boxes = _load_gt_boxes_in_ego(yaml_main, max_range=float(args.gt_bbox_range), max_num_boxes=int(args.gt_max_boxes))

                    row = csv_lookup.get((seq, frame), {})
                    cross_t = _safe_float(row.get("cross_agent_pose_trans_m"))
                    cross_r = _safe_float(row.get("cross_agent_pose_rot_deg"))
                    depth_rel = _safe_float(row.get("depth_rel"))
                    ratio = _safe_float(row.get("scale_to_gt_ratio_mean"))
                    eq_rel = _safe_float(row.get("scale_to_gt_eq_rel_err"))
                    mult = (eq_rel + 1.0) if eq_rel is not None else None
                    metric_value = _safe_float(item.get("metric_value"))

                    text_lines = [
                        f"{seq}/{frame} main={main_agent}",
                        f"metric={metric_key}",
                        f"sel_{tag}={metric_value:.4g}" if metric_value is not None else f"sel_{tag}=nan",
                    ]
                    if mult is not None and ratio is not None:
                        text_lines.append(f"scale_mult={mult:.3f} ratio={ratio:.3f}")
                    if cross_t is not None and cross_r is not None:
                        text_lines.append(f"cross_t={cross_t:.2f}m cross_r={cross_r:.2f}deg")
                    if depth_rel is not None:
                        text_lines.append(f"depth_rel={depth_rel:.3f}")

                    title = f"{model_name} | {mode} | {metric_key} | {tag}"
                    _render_bev_png(
                        out_png=out_png,
                        pred_pcd=pred_pcd,
                        yaml_main=yaml_main,
                        gt_points=gt_points,
                        gt_boxes=gt_boxes,
                        title=title,
                        rng=rng,
                        max_pred_points=int(args.max_pred_points),
                        max_gt_points=int(args.max_gt_points),
                        z_min=float(args.z_min) if args.z_min is not None else None,
                        z_max=float(args.z_max) if args.z_max is not None else None,
                        radius_max=float(args.radius_max) if args.radius_max is not None else None,
                        x_min=float(args.x_min),
                        x_max=float(args.x_max),
                        y_min=float(args.y_min),
                        y_max=float(args.y_max),
                        text_lines=text_lines,
                    )

                index_items.append(
                    {
                        "model": model_name,
                        "mode": mode,
                        "metric": metric_key,
                        "tag": str(tag),
                        "png": str(out_png.relative_to(out_dir)),
                    }
                )

    # Write a minimal offline HTML gallery.
    index_html = out_dir / "index.html"
    index_html.parent.mkdir(parents=True, exist_ok=True)

    # Group by model->metric.
    by_model: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for it in index_items:
        by_model.setdefault(it["model"], {}).setdefault(f"{it['mode']}::{it['metric']}", []).append(it)
    for m in by_model:
        for k in by_model[m]:
            by_model[m][k].sort(key=lambda x: (x["tag"], x["png"]))

    parts: List[str] = []
    parts.append("<!doctype html><meta charset='utf-8' />")
    parts.append("<title>Metric Viz BEV Gallery</title>")
    parts.append(
        "<style>body{font-family:system-ui,Arial,sans-serif} .grid{display:flex;flex-wrap:wrap;gap:10px} "
        ".tile{border:1px solid #ddd;padding:6px} img{max-width:420px;height:auto;display:block}</style>"
    )
    parts.append("<h1>Metric Viz BEV Gallery (Pred vs GT LiDAR + GT boxes)</h1>")
    parts.append(f"<p>root: <code>{html.escape(str(metric_viz_root))}</code></p>")
    parts.append(f"<p>out: <code>{html.escape(str(out_dir))}</code></p>")

    for model_name, groups in sorted(by_model.items()):
        parts.append(f"<h2>{html.escape(model_name)}</h2>")
        for group_name, items in sorted(groups.items()):
            parts.append(f"<h3>{html.escape(group_name)}</h3>")
            parts.append("<div class='grid'>")
            for it in items:
                rel = html.escape(it["png"])
                parts.append("<div class='tile'>")
                parts.append(f"<div><code>{html.escape(it['tag'])}</code></div>")
                parts.append(f"<img src='{rel}' />")
                parts.append("</div>")
            parts.append("</div>")

    index_html.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[OK] wrote gallery: {index_html}")


if __name__ == "__main__":
    main()
