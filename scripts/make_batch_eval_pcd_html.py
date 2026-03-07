#!/usr/bin/env python3
"""
Generate offline HTML visualizations for `scripts/batch_eval.py` outputs.

This script scans `<eval_root>/*/*_representatives/*.pcd`, loads the predicted
point clouds, overlays them with the corresponding OPV2V GT LiDAR point cloud,
and writes Plotly HTML files plus an `index.html`.

Important: OPV2V GT LiDAR `.pcd` files are in the CARLA/UE coordinate convention
(X forward, Y right, Z up). MapAnything predictions are produced in the OpenCV
convention (X right, Y down, Z forward). Before overlaying, we convert the
predicted points back to CARLA/UE so that height/radius filters and BEV axes are
interpretable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_processing.opv2v_pose_utils import (  # noqa: E402
    get_vehicle_bboxes_in_ego,
    load_frame_metadata,
)

# Carla/UE (X forward, Y right, Z up) -> OpenCV (X right, Y down, Z forward)
_CARLA_TO_OPENCV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _infer_coord_convention(points: np.ndarray) -> str:
    """Heuristic to detect whether `points` look like CARLA or OpenCV convention.

    In OPV2V driving scenes, CARLA points tend to have a small Z span (height),
    while OpenCV points (produced by MapAnything) tend to have a small Y span
    (since OpenCV Y points down, corresponding to -Z in CARLA).
    """

    if points.size == 0:
        return "unknown"
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    span = hi - lo
    if span[2] <= span[1]:
        return "carla"
    return "opencv"


def _opencv_to_carla(points_cv: np.ndarray) -> np.ndarray:
    """Convert Nx3 points from OpenCV convention to CARLA/UE convention."""

    if points_cv.size == 0:
        return points_cv
    # Column-vector form: p_cv = S * p_carla  =>  p_carla = S^T * p_cv
    return (_CARLA_TO_OPENCV.T @ points_cv.T).T


@dataclass(frozen=True)
class FrameInfo:
    sequence: str
    frame: str
    main_agent: str


def _load_gt_boxes(
    *,
    images_root: Path,
    split: str,
    sequence: str,
    frame: str,
    main_agent: str,
    bbox_range: float | None,
    max_boxes: int | None,
) -> List[np.ndarray]:
    yaml_path = images_root / split / sequence / main_agent / f"{frame}.yaml"
    if not yaml_path.is_file():
        return []
    try:
        meta = load_frame_metadata(yaml_path)
        boxes = get_vehicle_bboxes_in_ego(meta, max_range=bbox_range)
    except Exception:  # noqa: BLE001
        return []

    corners = [
        v["corners"].astype(np.float32, copy=False)
        for v in boxes.values()
        if isinstance(v, dict) and "corners" in v
    ]
    if max_boxes is not None and max_boxes > 0:
        corners = corners[: int(max_boxes)]
    return corners


def _edge_lengths(corners: np.ndarray) -> tuple[float, float, float]:
    # Corners are generated in `box_to_corners_ego` order: sx, sy, sz.
    # So edges (0-4, 0-2, 0-1) correspond to length/width/height respectively.
    length = float(np.linalg.norm(corners[0] - corners[4]))
    width = float(np.linalg.norm(corners[0] - corners[2]))
    height = float(np.linalg.norm(corners[0] - corners[1]))
    return length, width, height


def _load_pred_boxes(
    rep_path: Path,
    *,
    min_score: float,
    max_boxes: int | None,
    min_extent: float | None,
    max_extent: float | None,
) -> List[np.ndarray]:
    pred_path = rep_path.with_name(rep_path.stem + "_pred_boxes.json")
    if not pred_path.is_file():
        return []
    try:
        data = json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    boxes = data.get("boxes", [])
    if not isinstance(boxes, list):
        return []
    corners_list: List[np.ndarray] = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        score = item.get("score", 0.0)
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        if score_f < float(min_score):
            continue
        corners = item.get("corners")
        if not isinstance(corners, list) or len(corners) != 8:
            continue
        arr = np.asarray(corners, dtype=np.float32)
        if arr.shape != (8, 3):
            continue
        if min_extent is not None or max_extent is not None:
            length, width, height = _edge_lengths(arr)
            if min_extent is not None:
                if length < min_extent or width < min_extent or height < min_extent:
                    continue
            if max_extent is not None:
                if length > max_extent or width > max_extent or height > max_extent:
                    continue
        corners_list.append(arr)
        if max_boxes is not None and max_boxes > 0 and len(corners_list) >= int(max_boxes):
            break
    return corners_list


_BOX_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _box_edges_to_trace(
    boxes: Iterable[np.ndarray],
    *,
    name: str,
    color: str,
    width: int = 4,
    opacity: float = 0.9,
) -> go.Scatter3d | None:
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    count = 0
    for corners in boxes:
        if corners.shape != (8, 3):
            continue
        for a, b in _BOX_EDGES:
            xs.extend([float(corners[a, 0]), float(corners[b, 0]), None])
            ys.extend([float(corners[a, 1]), float(corners[b, 1]), None])
            zs.extend([float(corners[a, 2]), float(corners[b, 2]), None])
        count += 1
    if count == 0:
        return None
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        opacity=opacity,
    )


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    try:
        with pcd_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("DATA"):
                    break
            points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    except (UnicodeDecodeError, ValueError):
        points = _load_binary_pcd_xyz(pcd_path)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points


def _load_binary_pcd_xyz(pcd_path: Path) -> np.ndarray:
    try:
        import open3d as o3d  # local import to keep optional dependency
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Binary PCD parsing requires open3d. Install open3d or export ASCII PCDs."
        ) from exc
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd is None:
        raise RuntimeError(f"Failed to load PCD: {pcd_path}")
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return points


def _filter_points(
    points: np.ndarray,
    *,
    z_min: float | None,
    z_max: float | None,
    radius_max: float | None,
) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    if radius_max is not None:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= radius_max
    return points[mask]


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _parse_rep_stem(stem: str) -> Tuple[str, str, str]:
    # Robust parsing for tags that may contain underscores (e.g. "worst_scale").
    # OPV2V frame ids are 6-digit strings, which we use as an anchor.
    m = re.match(r"^(?P<sequence>.+)_(?P<frame>\d{6})_(?P<tag>.+)$", stem)
    if m:
        return m.group("sequence"), m.group("frame"), m.group("tag")

    # Fallback: assume `<sequence>_<frame>_<tag>` with no underscores in tag.
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected representative filename stem: {stem}")
    tag = parts[-1]
    frame = parts[-2]
    sequence = "_".join(parts[:-2])
    return sequence, frame, tag


def _pick_summary(eval_root: Path) -> Path:
    candidates = sorted(eval_root.glob("summary_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary_*.json found under {eval_root}")
    return candidates[0]


def _load_frame_index(summary_json: Path) -> Tuple[str, Dict[Tuple[str, str], FrameInfo]]:
    data = json.loads(summary_json.read_text())
    split = data.get("split")
    frames = data.get("frames", [])
    index: Dict[Tuple[str, str], FrameInfo] = {}
    for item in frames:
        seq = item.get("sequence")
        frame = item.get("frame")
        main_agent = item.get("main_agent")
        if not (seq and frame and main_agent):
            continue
        index[(seq, frame)] = FrameInfo(sequence=seq, frame=frame, main_agent=str(main_agent))
    if not isinstance(split, str) or not split:
        raise ValueError(f"Invalid split in {summary_json}")
    return split, index


def _make_figure(
    *,
    pred_points: np.ndarray,
    gt_points: np.ndarray | None,
    gt_boxes: List[np.ndarray] | None,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    if gt_points is not None and gt_points.size:
        fig.add_trace(
            go.Scatter3d(
                x=gt_points[:, 0],
                y=gt_points[:, 1],
                z=gt_points[:, 2],
                mode="markers",
                name="GT LiDAR",
                marker=dict(size=1, color="rgba(140,140,140,0.35)"),
            )
        )
    if gt_boxes:
        trace = _box_edges_to_trace(
            gt_boxes,
            name="GT Boxes",
            color="rgba(213,94,0,0.9)",
            width=4,
            opacity=0.9,
        )
        if trace is not None:
            fig.add_trace(trace)
    fig.add_trace(
        go.Scatter3d(
            x=pred_points[:, 0],
            y=pred_points[:, 1],
            z=pred_points[:, 2],
            mode="markers",
            name="Pred",
            marker=dict(size=1, color="rgba(0,114,178,0.55)"),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _discover_representatives(eval_root: Path) -> List[Path]:
    return sorted(eval_root.glob("*/*_representatives/*.pcd"))


def _write_index(out_dir: Path, items: Iterable[Tuple[str, str, str, str, Path]]) -> None:
    rows = []
    for model, mode, sequence, frame, html_path in items:
        rel = html_path.relative_to(out_dir)
        rows.append(
            f"<tr><td>{model}</td><td>{mode}</td><td>{sequence}</td><td>{frame}</td>"
            f"<td><a href='{rel.as_posix()}'>{rel.as_posix()}</a></td></tr>"
        )
    html = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>OPV2V Batch Eval Report</title></head><body>",
            "<h2>OPV2V Batch Eval Report</h2>",
            "<p>Open the links below (offline; uses local <code>plotly.min.js</code>).</p>",
            "<table border='1' cellpadding='6' cellspacing='0'>",
            "<tr><th>Model</th><th>Mode</th><th>Sequence</th><th>Frame</th><th>HTML</th></tr>",
            *rows,
            "</table></body></html>",
        ]
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--images_root", type=Path, default=REPO_ROOT / "data" / "opv2v")
    parser.add_argument("--split", type=str, default=None, help="Override split (otherwise read from summary JSON)")
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: <eval_root>/html")
    parser.add_argument("--max_points", type=int, default=80_000, help="Max points per cloud (after filtering)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--radius_max", type=float, default=None)
    parser.add_argument("--draw_gt_boxes", action="store_true", help="Overlay OPV2V GT vehicle boxes (ego frame)")
    parser.add_argument("--bbox_range", type=float, default=120.0, help="Max range for GT boxes (meters)")
    parser.add_argument("--max_boxes", type=int, default=40, help="Max number of GT boxes to draw per frame")
    parser.add_argument("--draw_pred_boxes", action="store_true", help="Overlay predicted boxes saved by batch_eval (ego frame)")
    parser.add_argument("--pred_min_score", type=float, default=0.3, help="Min score for predicted boxes")
    parser.add_argument("--pred_max_boxes", type=int, default=30, help="Max predicted boxes to draw per frame")
    parser.add_argument("--pred_min_extent", type=float, default=None, help="Min edge length (meters) for predicted boxes")
    parser.add_argument("--pred_max_extent", type=float, default=None, help="Max edge length (meters) for predicted boxes")
    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"eval_root not found: {eval_root}")

    summary_json = args.summary_json.expanduser().resolve() if args.summary_json else _pick_summary(eval_root)
    split_from_summary, frame_index = _load_frame_index(summary_json)
    split = args.split or split_from_summary

    out_dir = (args.out_dir or (eval_root / "html")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = _discover_representatives(eval_root)
    if not reps:
        raise FileNotFoundError(f"No representatives found under {eval_root} (expected */*_representatives/*.pcd)")

    rng = np.random.default_rng(args.seed)
    index_items: List[Tuple[str, str, str, str, Path]] = []

    for rep_path in reps:
        model = rep_path.parent.parent.name
        mode = rep_path.parent.name.split("_", 1)[0]
        sequence, frame, tag = _parse_rep_stem(rep_path.stem)

        pred_points = _load_ascii_pcd_xyz(rep_path)
        pred_coord = _infer_coord_convention(pred_points)
        if pred_coord == "opencv":
            pred_points = _opencv_to_carla(pred_points)
        pred_points = _filter_points(pred_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
        pred_points = _downsample(pred_points, args.max_points, rng)

        gt_points = None
        gt_boxes = None
        pred_boxes = None
        key = (sequence, frame)
        if key in frame_index:
            main_agent = frame_index[key].main_agent
            gt_path = args.images_root / split / sequence / main_agent / f"{frame}.pcd"
            if gt_path.is_file():
                gt_points = _load_ascii_pcd_xyz(gt_path)
                gt_points = _filter_points(gt_points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
                gt_points = _downsample(gt_points, args.max_points, rng)
            if args.draw_gt_boxes:
                gt_boxes = _load_gt_boxes(
                    images_root=args.images_root,
                    split=split,
                    sequence=sequence,
                    frame=frame,
                    main_agent=main_agent,
                    bbox_range=float(args.bbox_range) if args.bbox_range is not None else None,
                    max_boxes=args.max_boxes,
                )
        if args.draw_pred_boxes:
            pred_boxes = _load_pred_boxes(
                rep_path,
                min_score=float(args.pred_min_score),
                max_boxes=int(args.pred_max_boxes) if args.pred_max_boxes is not None else None,
                min_extent=float(args.pred_min_extent) if args.pred_min_extent is not None else None,
                max_extent=float(args.pred_max_extent) if args.pred_max_extent is not None else None,
            )

        title = f"{model} / {mode} / {sequence} / {frame} ({tag})"
        fig = _make_figure(pred_points=pred_points, gt_points=gt_points, gt_boxes=gt_boxes, title=title)
        if pred_boxes:
            trace = _box_edges_to_trace(
                pred_boxes,
                name="Pred Boxes",
                color="rgba(0,158,115,0.9)",
                width=3,
                opacity=0.9,
            )
            if trace is not None:
                fig.add_trace(trace)

        html_path = out_dir / model / mode / f"{sequence}_{frame}_{tag}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path), include_plotlyjs="directory", full_html=True)
        index_items.append((model, mode, sequence, frame, html_path))

    _write_index(out_dir, index_items)
    print(f"[OK] Wrote {len(index_items)} HTML files under {out_dir}")


if __name__ == "__main__":
    main()
