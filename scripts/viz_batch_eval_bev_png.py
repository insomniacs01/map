#!/usr/bin/env python3
"""
Create a quick BEV PNG overlay for OPV2V batch-eval representatives.

It plots:
  - predicted point cloud (.pcd) in CARLA/UE ego frame (X forward, Y right, Z up)
  - optional GT LiDAR point cloud (.pcd) in the same ego frame (gray)
  - GT vehicle boxes from the corresponding OPV2V YAML
  - optional predicted boxes saved by `scripts/batch_eval.py` (*_pred_boxes.json)

Example:
  cd map-anything
  conda run -n mapanything_ft python scripts/viz_batch_eval_bev_png.py \\
    --pred_pcd eval_runs/.../single_representatives/<seq>_<frame>_best.pcd \\
    --yaml data/opv2v/test/<seq>/<agent>/<frame>.yaml \\
    --out_png eval_runs/.../bev.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing.opv2v_pose_utils import load_frame_metadata
from mapanything.datasets.opv2v import _extract_vehicle_boxes_in_ego


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        pts = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return pts


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


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


def _bev_cell_indices(
    points_xy: np.ndarray,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return BEV grid indices (ix, iy) and valid mask for XY points."""
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


def _box_to_bev_poly(box: np.ndarray) -> np.ndarray:
    x, y, _z, l, w, _h, yaw = [float(v) for v in box]
    dx = l / 2.0
    dy = w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _box_corners_bev_xy(box: np.ndarray) -> np.ndarray:
    """Return 4x2 BEV corners (CCW) for box [x,y,z,l,w,h,yaw]."""
    x, y, _, l, w, _, yaw = [float(v) for v in box]
    dx = l * 0.5
    dy = w * 0.5
    corners = np.array(
        [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy],
        ],
        dtype=np.float32,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def _polygon_area(poly: List[List[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    x = np.asarray([p[0] for p in poly], dtype=np.float64)
    y = np.asarray([p[1] for p in poly], dtype=np.float64)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(p: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> bool:
    # left-of test for CCW clip polygon
    return (
        float((edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0]))
        >= 0.0
    )


def _segment_intersection(s: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersect segment s->e with infinite line a->b."""
    se = e - s
    ab = b - a
    denom = float(se[0] * ab[1] - se[1] * ab[0])
    if abs(denom) < 1e-9:
        return e.copy()
    t = float(((a[0] - s[0]) * ab[1] - (a[1] - s[1]) * ab[0]) / denom)
    return s + t * se


def _convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping for convex polygons (Nx2)."""
    output = subject
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        if output.size == 0:
            break
        input_list = output
        output_list: list[np.ndarray] = []
        s = input_list[-1]
        for e in input_list:
            e_in = _inside(e, a, b)
            s_in = _inside(s, a, b)
            if e_in:
                if not s_in:
                    output_list.append(_segment_intersection(s, e, a, b))
                output_list.append(e)
            elif s_in:
                output_list.append(_segment_intersection(s, e, a, b))
            s = e
        output = np.stack(output_list, axis=0) if output_list else np.zeros((0, 2), dtype=np.float32)
    return output


def _oriented_bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    # Keep this identical to batch_eval.py to avoid visualization/eval drift.
    poly_a = _box_corners_bev_xy(box_a)
    poly_b = _box_corners_bev_xy(box_b)
    inter_poly = _convex_clip(poly_a, poly_b)
    inter = _polygon_area(inter_poly.tolist())
    area_a = _polygon_area(poly_a.tolist())
    area_b = _polygon_area(poly_b.tolist())
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def _load_pred_boxes(pred_json: Path, *, min_score: float) -> Tuple[List[Dict[str, object]], float | None]:
    if not pred_json.is_file():
        return [], None
    try:
        data = json.loads(pred_json.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [], None
    iou_thresh = None
    try:
        if "iou_thresh" in data:
            iou_thresh = float(data["iou_thresh"])
    except Exception:
        iou_thresh = None
    boxes = data.get("boxes", [])
    if not isinstance(boxes, list):
        return [], iou_thresh
    out: List[Dict[str, object]] = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score", 0.0))
        except Exception:
            score = 0.0
        if score < float(min_score):
            continue
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 7:
            continue
        arr = np.asarray(box, dtype=np.float32)
        if arr.shape != (7,):
            continue
        out.append({"score": float(score), "box": arr})
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out, iou_thresh


def _plot_boxes(
    ax,
    boxes: Iterable[np.ndarray],
    *,
    color: str,
    lw: float,
    label: str | None,
    ls: str = "-",
) -> None:
    first = True
    for box in boxes:
        poly = _box_to_bev_poly(box)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=lw, linestyle=ls, label=label if first else None)
        first = False


def _match_pred_boxes_greedy(
    pred_boxes: List[Dict[str, object]],
    gt_boxes: np.ndarray,
    *,
    iou_thresh: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Greedy score-sorted TP/FP matching with oriented BEV IoU.

    Matching semantics mirror `scripts/batch_eval.py`:
      - predictions are processed in descending score order
      - each GT can match at most one prediction
      - a prediction is TP if its best-IoU unmatched GT is >= threshold, else FP

    Returns:
      - tp_pred_boxes: list of (7,) boxes
      - fp_pred_boxes: list of (7,) boxes
      - gt_matched_mask: (G,) bool
    """
    if gt_boxes.size == 0:
        tp_boxes: List[np.ndarray] = []
        fp_boxes = [item["box"] for item in pred_boxes if isinstance(item.get("box"), np.ndarray)]
        return tp_boxes, fp_boxes, np.zeros((0,), dtype=bool)

    gt_matched = np.zeros((gt_boxes.shape[0],), dtype=bool)
    tp_pred: List[np.ndarray] = []
    fp_pred: List[np.ndarray] = []
    for item in pred_boxes:
        box = item.get("box")
        if not isinstance(box, np.ndarray) or box.shape != (7,):
            continue
        best_iou = 0.0
        best_j = -1
        for j in range(gt_boxes.shape[0]):
            if gt_matched[j]:
                continue
            iou = _oriented_bev_iou(box, gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= float(iou_thresh) and best_j >= 0:
            gt_matched[best_j] = True
            tp_pred.append(box)
        else:
            fp_pred.append(box)
    return tp_pred, fp_pred, gt_matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_pcd", type=Path, required=True)
    parser.add_argument(
        "--gt_pcd",
        type=Path,
        default=None,
        help="Optional OPV2V GT LiDAR .pcd. If omitted, defaults to `<yaml>.pcd` when it exists.",
    )
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--pred_boxes_json", type=Path, default=None)
    parser.add_argument("--out_png", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=80_000)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--radius_max", type=float, default=60.0)
    parser.add_argument("--gt_bbox_range", type=float, default=120.0)
    parser.add_argument("--gt_max_boxes", type=int, default=128)
    parser.add_argument("--pred_min_score", type=float, default=0.05)
    parser.add_argument("--det_x_min", type=float, default=0.0, help="Fixed BEV x_min (meters) (match det range)")
    parser.add_argument("--det_x_max", type=float, default=120.0, help="Fixed BEV x_max (meters) (match det range)")
    parser.add_argument("--det_y_min", type=float, default=-50.0, help="Fixed BEV y_min (meters) (match det range)")
    parser.add_argument("--det_y_max", type=float, default=50.0, help="Fixed BEV y_max (meters) (match det range)")
    parser.add_argument(
        "--pred_point_color_by",
        type=str,
        default="auto",
        choices=("auto", "plain", "z", "match"),
        help=(
            "How to color predicted points. "
            "'plain' uses a single color. "
            "'z' colors by height (turbo). "
            "'match' colors pred points by BEV cell overlap with GT LiDAR (green/red). "
            "'auto' selects 'match' if GT points are available, else 'plain'."
        ),
    )
    parser.add_argument("--match_res", type=float, default=0.5, help="BEV resolution (m) used for match-based point coloring.")
    parser.add_argument(
        "--color_pred_by_match",
        action="store_true",
        help="Color predicted boxes by TP/FP using oriented-IoU matching against GT.",
    )
    parser.add_argument(
        "--match_iou_thresh",
        type=float,
        default=None,
        help="IoU threshold for TP/FP matching (default: read from *_pred_boxes.json, else 0.5).",
    )
    args = parser.parse_args()

    pred_pcd = args.pred_pcd.expanduser().resolve()
    yaml_path = args.yaml.expanduser().resolve()
    if not pred_pcd.is_file():
        raise FileNotFoundError(pred_pcd)
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)

    gt_pcd = args.gt_pcd
    if gt_pcd is None:
        cand = yaml_path.with_suffix(".pcd")
        gt_pcd = cand if cand.is_file() else None
    if gt_pcd is not None:
        gt_pcd = gt_pcd.expanduser().resolve()
        if not gt_pcd.is_file():
            gt_pcd = None

    pred_json = args.pred_boxes_json
    if pred_json is None:
        pred_json = pred_pcd.with_name(pred_pcd.stem + "_pred_boxes.json")
    pred_json = pred_json.expanduser().resolve()

    rng = np.random.default_rng(int(args.seed))

    points = _load_ascii_pcd_xyz(pred_pcd)
    points = _filter_points(points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
    points = _downsample(points, int(args.max_points), rng)

    gt_points = None
    if gt_pcd is not None:
        gt_points = _load_ascii_pcd_xyz(gt_pcd)
        gt_points = _filter_points(gt_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
        gt_points = _downsample(gt_points, int(args.max_points), rng)

    meta = load_frame_metadata(yaml_path)
    gt_padded, gt_mask = _extract_vehicle_boxes_in_ego(
        meta, max_range=float(args.gt_bbox_range) if args.gt_bbox_range is not None else None, max_num_boxes=int(args.gt_max_boxes)
    )
    gt_boxes = gt_padded[gt_mask]

    pred_items, pred_iou_from_json = _load_pred_boxes(pred_json, min_score=float(args.pred_min_score))
    match_iou = float(args.match_iou_thresh) if args.match_iou_thresh is not None else float(pred_iou_from_json or 0.5)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    xlim = (float(args.det_x_min), float(args.det_x_max))
    ylim = (float(args.det_y_min), float(args.det_y_max))
    if gt_points is not None and gt_points.size:
        ax.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            s=0.10,
            c="0.35",
            alpha=0.12,
            rasterized=True,
            label="GT LiDAR",
        )

    pred_color = str(args.pred_point_color_by)
    if pred_color == "auto":
        pred_color = "match" if (gt_points is not None and gt_points.size) else "plain"

    match_str = None
    if pred_color == "plain":
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c="tab:blue", alpha=0.30, rasterized=True, label="Pred pts")
    elif pred_color == "z":
        if points.size:
            zc = np.clip(points[:, 2], -2.5, 2.5)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=0.15,
                c=zc,
                cmap="turbo",
                alpha=0.55,
                rasterized=True,
                label="Pred pts (z)",
            )
    elif pred_color == "match":
        if gt_points is not None and gt_points.size and points.size:
            mm = _bev_match_mask(gt_points=gt_points, pred_points=points, xlim=xlim, ylim=ylim, resolution=float(args.match_res))
            n = int(points.shape[0])
            n_match = int(mm.sum())
            match_str = f"bev_match={100.0 * n_match / n:.0f}% (res={float(args.match_res):.2g}m)" if n > 0 else None
            if bool(mm.any()):
                ax.scatter(
                    points[mm, 0],
                    points[mm, 1],
                    s=0.15,
                    c="tab:green",
                    alpha=0.26,
                    rasterized=True,
                    label=f"Pred match ({n_match}/{n})",
                )
            if bool((~mm).any()):
                ax.scatter(
                    points[~mm, 0],
                    points[~mm, 1],
                    s=0.16,
                    c="tab:red",
                    alpha=0.34,
                    rasterized=True,
                    label=f"Pred mismatch ({n - n_match}/{n})",
                )
        else:
            ax.scatter(points[:, 0], points[:, 1], s=0.15, c="tab:blue", alpha=0.30, rasterized=True, label="Pred pts")
    else:
        raise ValueError(f"unknown --pred_point_color_by: {pred_color}")

    _plot_boxes(ax, gt_boxes, color="black", lw=1.2, label=f"GT boxes ({gt_boxes.shape[0]})")
    if args.color_pred_by_match and pred_items:
        tp_boxes, fp_boxes, gt_matched = _match_pred_boxes_greedy(pred_items, gt_boxes, iou_thresh=match_iou)
        fn = 0 if gt_boxes.size == 0 else int((~gt_matched).sum())
        if fn > 0 and gt_boxes.size:
            _plot_boxes(
                ax,
                gt_boxes[~gt_matched],
                color="tab:red",
                lw=1.4,
                ls="--",
                label=f"GT FN ({fn})",
            )
        if tp_boxes:
            _plot_boxes(ax, tp_boxes, color="tab:green", lw=1.2, label=f"Pred TP ({len(tp_boxes)})")
        if fp_boxes:
            _plot_boxes(ax, fp_boxes, color="tab:orange", lw=1.0, label=f"Pred FP ({len(fp_boxes)})")
        top_left = f"TP={len(tp_boxes)}  FP={len(fp_boxes)}  FN={fn}  (IoU>={match_iou:.2f})"
        if match_str:
            top_left += "\n" + str(match_str)
        ax.text(
            0.01,
            0.99,
            top_left,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            family="monospace",
            bbox=dict(facecolor="white", edgecolor="0.2", alpha=0.92, boxstyle="round,pad=0.30"),
        )
    else:
        pred_boxes = [item["box"] for item in pred_items if isinstance(item.get("box"), np.ndarray)]
        if pred_boxes:
            _plot_boxes(ax, pred_boxes, color="tab:orange", lw=1.0, label=f"Pred boxes ({len(pred_boxes)})")
        if match_str:
            ax.text(
                0.01,
                0.99,
                match_str,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.2", alpha=0.92, boxstyle="round,pad=0.30"),
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("+X forward (m)")
    ax.set_ylabel("+Y right (m)")
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"BEV overlay: {pred_pcd.name}", fontsize=10)

    out_png = args.out_png.expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
