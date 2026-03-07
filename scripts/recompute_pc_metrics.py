#!/usr/bin/env python3
"""
Recompute detection-side metrics (Chamfer / BEV IoU) from cached predicted point clouds.

Use this script after running `batch_eval.py` with `--pc_save_dir` so that height/radius
filters can be changed without re-running heavy inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow imports from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from data_processing.opv2v_pose_utils import load_ascii_pcd_xyz  # noqa: E402
import importlib.util

_BATCH_EVAL_SPEC = importlib.util.spec_from_file_location("batch_eval_module", SCRIPTS_ROOT / "batch_eval.py")
if _BATCH_EVAL_SPEC is None or _BATCH_EVAL_SPEC.loader is None:
    raise ImportError("Unable to locate batch_eval.py for import")
batch_eval = importlib.util.module_from_spec(_BATCH_EVAL_SPEC)
sys.modules["batch_eval_module"] = batch_eval
_BATCH_EVAL_SPEC.loader.exec_module(batch_eval)

EvalResult = batch_eval.EvalResult
FrameInfo = batch_eval.FrameInfo
PCMetricConfig = batch_eval.PCMetricConfig
compute_detection_metrics = batch_eval.compute_detection_metrics
load_frames_from_json = batch_eval.load_frames_from_json
summarize_results = batch_eval.summarize_results

CSV_COLUMNS = [
    "sequence",
    "frame",
    "pose_abs_m",
    "pose_rot_deg",
    "depth_rmse",
    "depth_mae",
    "depth_rel",
    "scale_err",
    "scale_log_err",
    "scale_eq_rel_err",
    "scale_ratio_mean",
    "scale_ratio_median",
    "scale_ratio_p90",
    "chamfer_pred_to_gt",
    "chamfer_gt_to_pred",
    "chamfer_filtered_pred_to_gt",
    "chamfer_filtered_gt_to_pred",
    "bev_iou_raw",
    "bev_iou_filtered",
]

def _infer_coord_convention(points: np.ndarray) -> str:
    """Heuristic to detect whether `points` look like CARLA or OpenCV convention.

    Cached point clouds produced by older `batch_eval.py` versions were stored in
    OpenCV convention, while OPV2V GT LiDAR uses CARLA/UE convention (Z-up).
    """

    if points.size == 0:
        return "unknown"
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    span = hi - lo
    if span[2] <= span[1]:
        return "carla"
    return "opencv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="Dataset split (train/validate/test)")
    parser.add_argument("--frames_json", type=Path, required=True, help="JSON describing frames (same format as summary_*.json)")
    parser.add_argument("--pc_cache_dir", type=Path, required=True, help="Directory containing cached predicted point clouds")
    parser.add_argument("--output_root", type=Path, default=Path("/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/xiongyijin_workspace/opv2v_batch_eval"))
    parser.add_argument("--images_root", type=Path, default=Path("/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V"))
    parser.add_argument("--models", nargs="*", help="Optional subset of model directory names to update (default: all in cache dir)")
    parser.add_argument("--modes", nargs="*", choices=("single", "coop"), default=("single", "coop"), help="Modes (single/coop) to update")
    parser.add_argument("--pc_filter_z_min", type=float, default=None, help="Z-min threshold for filtered Chamfer/IoU")
    parser.add_argument("--pc_filter_z_max", type=float, default=None, help="Z-max threshold for filtered Chamfer/IoU")
    parser.add_argument("--pc_filter_radius", type=float, default=None, help="Radius threshold for filtered Chamfer/IoU")
    parser.add_argument("--pc_bev_range", type=float, default=120.0, help="BEV grid range for detection metrics")
    parser.add_argument("--pc_bev_resolution", type=float, default=0.5, help="BEV grid resolution for detection metrics")
    return parser.parse_args()


def _format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _load_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"CSV {csv_path} is empty")
    return rows, reader.fieldnames or CSV_COLUMNS


def _write_csv(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str] | None = None) -> None:
    ordered_fields = fieldnames or CSV_COLUMNS
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ordered_fields)
        for row in rows:
            writer.writerow([row.get(col, "nan") for col in ordered_fields])


def _row_to_eval_result(row: Dict[str, str], frame: FrameInfo, mode: str) -> EvalResult:
    def _float_from(row_key: str) -> float:
        try:
            return float(row[row_key])
        except (KeyError, ValueError):
            return float("nan")

    def _float_or_none(row_key: str) -> float | None:
        try:
            value = float(row[row_key])
        except (KeyError, ValueError):
            return None
        return None if math.isnan(value) else value

    return EvalResult(
        frame=frame,
        mode=mode,
        pose_abs=_float_from("pose_abs_m"),
        pose_rot=_float_from("pose_rot_deg"),
        depth_rmse=_float_from("depth_rmse"),
        depth_mae=_float_from("depth_mae"),
        depth_rel=_float_from("depth_rel"),
        scale_err=_float_or_none("scale_err"),
        scale_log_err=_float_or_none("scale_log_err"),
        scale_eq_rel_err=_float_or_none("scale_eq_rel_err"),
        scale_ratio_mean=_float_or_none("scale_ratio_mean"),
        scale_ratio_median=_float_or_none("scale_ratio_median"),
        scale_ratio_p90=_float_or_none("scale_ratio_p90"),
        chamfer_pred_to_gt=_float_or_none("chamfer_pred_to_gt"),
        chamfer_gt_to_pred=_float_or_none("chamfer_gt_to_pred"),
        chamfer_filtered_pred_to_gt=_float_or_none("chamfer_filtered_pred_to_gt"),
        chamfer_filtered_gt_to_pred=_float_or_none("chamfer_filtered_gt_to_pred"),
        bev_iou_raw=_float_or_none("bev_iou_raw"),
        bev_iou_filtered=_float_or_none("bev_iou_filtered"),
    )


def _update_detection_metrics_for_mode(
    model: str,
    mode: str,
    frames: List[FrameInfo],
    cache_root: Path,
    csv_path: Path,
    gt_root: Path,
    split: str,
    pc_cfg: PCMetricConfig,
) -> Tuple[List[Dict[str, str]], List[EvalResult]]:
    rows, header = _load_csv(csv_path)
    row_lookup: Dict[Tuple[str, str], Dict[str, str]] = {(row["sequence"], row["frame"]): row for row in rows}
    frame_lookup = {(frame.sequence, frame.frame): frame for frame in frames}
    updated = 0
    for frame in frames:
        key = (frame.sequence, frame.frame)
        if key not in row_lookup:
            print(f"[WARN] {model}/{mode}: frame {frame.sequence}/{frame.frame} missing from CSV; skipping")
            continue
        cache_path = cache_root / model / mode / f"{frame.sequence}_{frame.frame}.npy"
        if not cache_path.is_file():
            print(f"[WARN] Cached point cloud missing: {cache_path}")
            continue
        gt_pcd_path = gt_root / split / frame.sequence / frame.main_agent / f"{frame.frame}.pcd"
        if not gt_pcd_path.is_file():
            print(f"[WARN] GT point cloud missing: {gt_pcd_path}")
            continue
        pred_points = np.load(cache_path).astype(np.float32)
        if _infer_coord_convention(pred_points) == "opencv":
            pred_points = batch_eval._points_opencv_to_carla(pred_points)
        gt_points = load_ascii_pcd_xyz(gt_pcd_path)
        metrics = compute_detection_metrics(pred_points, gt_points, pc_cfg)
        row = row_lookup[key]
        row["chamfer_pred_to_gt"] = _format_float(metrics["chamfer_pred_to_gt"])
        row["chamfer_gt_to_pred"] = _format_float(metrics["chamfer_gt_to_pred"])
        row["chamfer_filtered_pred_to_gt"] = _format_float(metrics.get("chamfer_filtered_pred_to_gt"))
        row["chamfer_filtered_gt_to_pred"] = _format_float(metrics.get("chamfer_filtered_gt_to_pred"))
        row["bev_iou_raw"] = _format_float(metrics.get("bev_iou_raw"))
        row["bev_iou_filtered"] = _format_float(metrics.get("bev_iou_filtered"))
        updated += 1
    if not updated:
        print(f"[WARN] {model}/{mode}: no rows updated; leave CSV untouched")
        eval_results = []
        for row in rows:
            key = (row["sequence"], row["frame"])
            frame = frame_lookup.get(key)
            if frame:
                eval_results.append(_row_to_eval_result(row, frame, mode))
        return rows, eval_results

    rows_sorted = sorted(rows, key=lambda r: (r["sequence"], r["frame"]))
    _write_csv(csv_path, rows_sorted, header)
    eval_results: List[EvalResult] = []
    for row in rows_sorted:
        key = (row["sequence"], row["frame"])
        frame = frame_lookup.get(key)
        if not frame:
            continue
        eval_results.append(_row_to_eval_result(row, frame, mode))
    print(f"[INFO] {model}/{mode}: updated detection metrics for {updated} frames -> {csv_path}")
    return rows_sorted, eval_results


def main() -> None:
    args = parse_args()
    frames = load_frames_from_json(args.frames_json)
    if not frames:
        raise RuntimeError("frames_json did not contain any entries")
    frame_lookup = {(f.sequence, f.frame): f for f in frames}
    cache_root = args.pc_cache_dir
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_root}")
    models = args.models or sorted([p.name for p in cache_root.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError(f"No model directories present in {cache_root}")
    modes = tuple(dict.fromkeys(args.modes))
    pc_cfg = PCMetricConfig(
        enabled=True,
        z_min=args.pc_filter_z_min,
        z_max=args.pc_filter_z_max,
        radius_max=args.pc_filter_radius,
        bev_range=args.pc_bev_range,
        bev_resolution=args.pc_bev_resolution,
    )
    overall_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in models:
        per_mode_results: Dict[str, List[EvalResult]] = {}
        for mode in modes:
            csv_path = args.output_root / model / f"{mode}_metrics.csv"
            cache_dir = cache_root / model / mode
            if not cache_dir.is_dir():
                print(f"[WARN] Cache missing for {model}/{mode}: {cache_dir}")
                continue
            rows, eval_results = _update_detection_metrics_for_mode(
                model,
                mode,
                frames,
                cache_root,
                csv_path,
                args.images_root,
                args.split,
                pc_cfg,
            )
            # eval_results already built; store if not empty
            if eval_results:
                per_mode_results[mode] = eval_results
        if per_mode_results:
            overall_summary[model] = summarize_results(per_mode_results)
    summary_path = args.output_root / f"summary_{args.split}.json"
    existing_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    existing_meta = None
    existing_model_paths = None
    existing_frames = [
        {
            "sequence": f.sequence,
            "frame": f.frame,
            "main_agent": f.main_agent,
            "coop_agents": list(f.coop_agents),
        }
        for f in frames
    ]
    if summary_path.is_file():
        with summary_path.open() as fh:
            prev = json.load(fh)
        existing_metrics = prev.get("metrics", {})
        existing_meta = prev.get("meta")
        existing_model_paths = prev.get("model_paths")
        if prev.get("frames"):
            existing_frames = prev["frames"]
    combined_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    if isinstance(existing_metrics, dict):
        combined_metrics.update(existing_metrics)
    combined_metrics.update(overall_summary)
    summary_payload = {
        "split": args.split,
        "frames": existing_frames,
        "metrics": combined_metrics,
    }
    # Preserve non-metric metadata written by batch_eval.py (fairness lock, det cfg, etc.).
    if existing_meta is not None:
        summary_payload["meta"] = existing_meta
    if existing_model_paths is not None:
        summary_payload["model_paths"] = existing_model_paths
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        json.dump(summary_payload, fh, indent=2)
    print(f"[INFO] Summary updated at {summary_path}")


if __name__ == "__main__":
    main()
