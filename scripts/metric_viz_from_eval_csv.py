#!/usr/bin/env python3
"""Generate representative point-cloud visualizations from an existing batch_eval metrics CSV.

This is a *no-re-eval* utility:
  1) Read per-frame metrics from `<eval_run_dir>/<model_key>/<mode>_metrics.csv`
  2) Select best / median / worst frames by a chosen metric key
  3) Re-run inference *only on those selected frames*
  4) Export predicted PCDs under:
       `<out_root>/<out_model_name>/<mode>_<metric>_representatives/*.pcd`

Then you can create an offline HTML index (pred vs GT LiDAR + GT boxes) with:
  python scripts/make_batch_eval_pcd_html.py --eval_root <out_root> --summary_json <eval_run_dir>/summary_test.json --draw_gt_boxes

Notes
-----
- Default protocol is deployment-style: keep intrinsics, strip GT extrinsics (`--keep_camera_poses` is False).
- Supports both `model_arch=mapanything` (uses `.infer()`) and `model_arch=vggt` (uses `forward()`).
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
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "metric"


def _pick_summary(eval_run_dir: Path) -> Path:
    candidates = sorted(eval_run_dir.glob("summary_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary_*.json found under {eval_run_dir}")
    return candidates[0]


def _load_frame_map(summary_json: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    doc = json.loads(summary_json.read_text(encoding="utf-8"))
    frames = doc.get("frames") or []
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not isinstance(frames, list):
        return out
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        if not (isinstance(seq, str) and isinstance(frame, str)):
            continue
        out[(seq, frame)] = fr
    return out


def _get_metric_value(row: Dict[str, str], metric_key: str) -> float | None:
    # Direct CSV column
    if metric_key in row:
        return _safe_float(row.get(metric_key))

    # Derived keys (keep minimal + explicit to avoid surprises)
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
class MetricRow:
    sequence: str
    frame: str
    value: float


def _load_metric_rows(csv_path: Path, metric_key: str) -> List[MetricRow]:
    rows: List[MetricRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for raw in rd:
            seq = raw.get("sequence")
            frame = raw.get("frame")
            if not seq or not frame:
                continue
            val = _get_metric_value(raw, metric_key)
            if val is None:
                continue
            rows.append(MetricRow(sequence=str(seq), frame=str(frame), value=float(val)))
    return rows


def _select_best_median_worst(rows: List[MetricRow], *, higher_is_better: bool) -> Dict[str, MetricRow]:
    if not rows:
        raise ValueError("No valid metric rows to select from (metric missing or all NaN).")
    rows_sorted = sorted(rows, key=lambda r: r.value, reverse=bool(higher_is_better))
    best = rows_sorted[0]
    worst = rows_sorted[-1]
    median = rows_sorted[len(rows_sorted) // 2]
    return {"best": best, "median": median, "worst": worst}


def _load_batch_eval_module() -> Any:
    batch_eval_path = Path(__file__).resolve().parent / "batch_eval.py"
    spec = importlib.util.spec_from_file_location("_batch_eval_mod_metric_viz", batch_eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load batch_eval module from {batch_eval_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_batch_eval_mod_metric_viz"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_run_dir", type=Path, required=True, help="Existing eval run dir with summary_*.json + <model_key>/*_metrics.csv")
    ap.add_argument("--model_key", type=str, default="geom_model")
    ap.add_argument("--mode", type=str, default="coop", choices=("single", "coop"))
    ap.add_argument("--metric_key", type=str, required=True, help="CSV column name, or one of: scale_to_gt_mult_err, scale_to_gt_abs_ratio_err")
    ap.add_argument("--higher_is_better", action="store_true", help="Select best=max(metric) instead of min(metric)")

    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint to re-run inference on selected frames")
    ap.add_argument("--model_arch", type=str, default="mapanything", choices=("mapanything", "vggt"))
    ap.add_argument("--out_root", type=Path, required=True, help="Output root: writes <out_root>/<out_model_name>/<mode>_<metric>_representatives/*.pcd")
    ap.add_argument("--out_model_name", type=str, required=True, help="Model label for visualization output directory")

    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--summary_json", type=Path, default=None, help="Override summary JSON (default: pick first summary_*.json in eval_run_dir)")
    ap.add_argument("--images_root", type=Path, default=Path("map-anything/data/opv2v"))
    ap.add_argument("--depth_root", type=Path, default=Path("map-anything/data/opv2v_depth"))
    ap.add_argument("--device", type=str, default="cuda:0")

    # MapAnything-specific
    ap.add_argument("--model_task", type=str, default="calibrated_sfm")
    ap.add_argument("--keep_camera_poses", action="store_true", help="Do NOT strip GT camera poses from inputs (NOT deployment-style)")
    ap.add_argument("--data_norm_type", type=str, default=None, help="Override image norm type: dinov2 or identity (default: dinov2 for mapanything, identity for vggt)")

    # VGGT-specific (optional knobs)
    ap.add_argument("--vggt_enable_metric_scale_head", action="store_true")
    ap.add_argument("--vggt_autocast_dtype", type=str, default=None, help="auto|bf16|fp16|fp32")
    args = ap.parse_args()

    eval_run_dir = args.eval_run_dir.expanduser().resolve()
    model_key = str(args.model_key)
    mode = str(args.mode)
    metric_key = str(args.metric_key)
    out_model_name = str(args.out_model_name)

    summary_json = args.summary_json.expanduser().resolve() if args.summary_json else _pick_summary(eval_run_dir)
    frame_map = _load_frame_map(summary_json)

    csv_path = eval_run_dir / model_key / f"{mode}_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing metrics csv: {csv_path}")

    rows = _load_metric_rows(csv_path, metric_key=metric_key)
    selected = _select_best_median_worst(rows, higher_is_better=bool(args.higher_is_better))

    # Load batch_eval utilities.
    be = _load_batch_eval_module()

    device = torch.device(str(args.device))
    model_arch = str(args.model_arch)
    norm_type = str(args.data_norm_type).strip() if args.data_norm_type else ""
    if not norm_type:
        norm_type = "identity" if model_arch == "vggt" else "dinov2"

    ckpt = args.ckpt.expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    # Init model once.
    if model_arch == "vggt":
        cfg_overrides = [
            "machine=local_a800",
            "dataset=opv2v_ft_a800",
            "model=vggt",
            "model.model_config.load_pretrained_weights=false",
            f"model.model_config.enable_metric_scale_head={'true' if args.vggt_enable_metric_scale_head else 'false'}",
            f"model.model_config.autocast_dtype={args.vggt_autocast_dtype}" if args.vggt_autocast_dtype else None,
            "loss=overall_loss",
        ]
    else:
        cfg_overrides = [
            "machine=local_a800",
            "dataset=opv2v_ft_a800",
            "model=mapanything",
            f"model/task={args.model_task}",
            "model.encoder.uses_torch_hub=false",
            "loss=overall_loss",
        ]
    cfg = {
        "path": str(be.REPO_ROOT / "configs/train.yaml"),
        "checkpoint_path": str(ckpt),
        "config_overrides": [v for v in cfg_overrides if v],
    }
    model = be.initialize_mapanything_local(cfg, device)
    model.eval()

    metric_slug = _slug(metric_key)
    out_rep_dir = args.out_root / out_model_name / f"{mode}_{metric_slug}_representatives"
    out_rep_dir.mkdir(parents=True, exist_ok=True)

    selection_payload: Dict[str, Any] = {
        "eval_run_dir": str(eval_run_dir),
        "summary_json": str(summary_json),
        "model_key": model_key,
        "mode": mode,
        "metric_key": metric_key,
        "higher_is_better": bool(args.higher_is_better),
        "ckpt": str(ckpt),
        "model_arch": model_arch,
        "model_task": str(args.model_task),
        "keep_camera_poses": bool(args.keep_camera_poses),
        "data_norm_type": norm_type,
        "out_model_name": out_model_name,
        "outputs": {},
    }

    for tag, item in selected.items():
        seq, frame = item.sequence, item.frame
        meta = frame_map.get((seq, frame)) or {}
        info = be.FrameInfo(
            sequence=seq,
            frame=frame,
            main_agent=str(meta.get("main_agent") or "641"),
            coop_agents=tuple(meta.get("coop_agents") or ("641", "650")),
        )

        if mode == "single":
            raw_views, _camera_info = be.build_single_raw(args.images_root, args.depth_root, args.split, info)
        else:
            raw_views, _camera_info = be.build_coop_raw(args.images_root, args.depth_root, args.split, info)

        gt_ref_pose_cv = None
        if raw_views and "camera_poses" in raw_views[0]:
            gt_ref_pose_cv = np.asarray(raw_views[0]["camera_poses"], dtype=np.float32)

        processed = be.preprocess_inputs(raw_views, norm_type=norm_type)
        for view in processed:
            view.pop("depth_z", None)
        if not args.keep_camera_poses:
            be.strip_external_calibration_inputs(processed)

        with torch.no_grad():
            if model_arch == "vggt":
                be._move_views_to_device(processed, device)
                raw_preds = model(processed)
                preds = be._vggt_preds_to_batch_eval_format(raw_preds, processed)
            else:
                preds = model.infer(processed, memory_efficient_inference=True)

        if gt_ref_pose_cv is None:
            pts, _ = be.predictions_to_pointcloud(preds, colorize=False)
        else:
            pts, _ = be.predictions_to_ego_pointcloud(preds, gt_ref_pose_cv, colorize=False)
        pts = be._points_opencv_to_carla(pts)

        out_path = out_rep_dir / f"{seq}_{frame}_{tag}.pcd"
        be.save_point_cloud(out_path, pts)
        selection_payload["outputs"][tag] = {
            "sequence": seq,
            "frame": frame,
            "metric_value": float(item.value),
            "pcd": str(out_path),
        }

    sel_path = args.out_root / out_model_name / f"selected_{mode}_{metric_slug}.json"
    sel_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote representatives: {out_rep_dir}")


if __name__ == "__main__":
    main()

