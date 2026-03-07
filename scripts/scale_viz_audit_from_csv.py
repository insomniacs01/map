#!/usr/bin/env python3
"""Generate scale-focused representative visualizations from an existing eval CSV.

Use case
--------
You already ran `scripts/batch_eval.py` on a fixed contract and have:
  - `<eval_run>/<model>/<mode>_metrics.csv`
  - (optional) a GT scale contract JSON with per-frame `gt_scale_single/coop`

This script:
1) derives per-frame ratio-to-GT scale metrics (`s/g`)
2) selects best/median/worst frames by `|s/g - 1|`
3) re-runs inference only on those selected frames
4) exports PCDs under `<out_dir>/<model>/{single,coop}_scale_representatives/*.pcd`

Then you can run:
  python scripts/make_batch_eval_pcd_html.py --eval_root <out_dir> --draw_gt_boxes
to get an offline HTML index overlaying pred PCD with GT LiDAR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


@dataclass(frozen=True)
class Row:
    sequence: str
    frame: str
    main_agent: str
    coop_agents: Tuple[str, ...]
    pose_abs_m: float | None
    pose_rot_deg: float | None
    depth_rmse: float | None
    depth_mae: float | None
    depth_rel: float | None
    chamfer_pred_to_gt: float | None
    chamfer_gt_to_pred: float | None
    bev_iou_raw: float | None

    # legacy fields from eval CSV
    scale_ratio_mean: float | None  # may be missing in older CSVs
    scale_err: float | None  # may be present instead

    # derived
    gt_scale: float | None
    scale_to_gt_ratio: float | None  # s/g
    scale_to_gt_err: float | None  # |s/g - 1|
    scale_to_gt_log_err: float | None  # |log(s/g)|
    scale_to_gt_mult_err: float | None  # exp(|log(s/g)|)


def _load_contract_gt(contract_json: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    doc = _load_json(contract_json)
    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"bad contract json (missing frames): {contract_json}")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        if seq is None or frame is None:
            continue
        entry: Dict[str, Any] = {}
        entry["main_agent"] = str(fr.get("main_agent") or "641")
        coop_agents = fr.get("coop_agents") or []
        if isinstance(coop_agents, (list, tuple)):
            entry["coop_agents"] = [str(a) for a in coop_agents]
        else:
            entry["coop_agents"] = ["641", "650"]
        gs = _safe_float(fr.get("gt_scale_single"))
        gc = _safe_float(fr.get("gt_scale_coop"))
        if gs is not None:
            entry["gt_scale_single"] = gs
        if gc is not None:
            entry["gt_scale_coop"] = gc
        out[(str(seq), str(frame))] = entry
    return out


def _infer_scale_from_csv_row(row: Dict[str, str]) -> float | None:
    """Return predicted scale `s` for this frame.

    Prefer `scale_ratio_mean` if present. Otherwise use `scale_err` if it
    is >=1 (then s = 1 + err is uniquely determined).
    """

    if "scale_ratio_mean" in row:
        v = _safe_float(row.get("scale_ratio_mean"))
        if v is not None and v > 0.0:
            return v
    if "scale_err" in row:
        e = _safe_float(row.get("scale_err"))
        if e is None:
            return None
        if e >= 1.0:
            return 1.0 + e
        # ambiguous when err<1 (could be 1+e or 1-e). Skip.
        return None
    return None


def _parse_metrics_csv(
    csv_path: Path,
    *,
    mode: str,
    gt_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Row]:
    out: List[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return out
        for raw in r:
            seq = raw.get("sequence")
            frame = raw.get("frame")
            if not seq or not frame:
                continue

            meta = gt_map.get((seq, frame), {})
            main_agent = str(meta.get("main_agent") or "641")
            coop_agents = meta.get("coop_agents") or ["641", "650"]
            if not isinstance(coop_agents, list):
                coop_agents = ["641", "650"]
            coop_agents_t = tuple(str(a) for a in coop_agents)

            s = _infer_scale_from_csv_row(raw)
            gt_key = "gt_scale_single" if mode == "single" else "gt_scale_coop"
            gt = _safe_float(meta.get(gt_key))
            ratio = None
            err = None
            log_err = None
            mult_err = None
            if s is not None and gt is not None and gt > 1e-8:
                ratio = s / gt
                if ratio > 0.0 and math.isfinite(ratio):
                    err = abs(ratio - 1.0)
                    log_err = abs(math.log(max(ratio, 1e-8)))
                    mult_err = float(math.exp(log_err))

            out.append(
                Row(
                    sequence=seq,
                    frame=frame,
                    main_agent=main_agent,
                    coop_agents=coop_agents_t,
                    pose_abs_m=_safe_float(raw.get("pose_abs_m")),
                    pose_rot_deg=_safe_float(raw.get("pose_rot_deg")),
                    depth_rmse=_safe_float(raw.get("depth_rmse")),
                    depth_mae=_safe_float(raw.get("depth_mae")),
                    depth_rel=_safe_float(raw.get("depth_rel")),
                    chamfer_pred_to_gt=_safe_float(raw.get("chamfer_pred_to_gt")),
                    chamfer_gt_to_pred=_safe_float(raw.get("chamfer_gt_to_pred")),
                    bev_iou_raw=_safe_float(raw.get("bev_iou_raw")),
                    scale_ratio_mean=_safe_float(raw.get("scale_ratio_mean")),
                    scale_err=_safe_float(raw.get("scale_err")),
                    gt_scale=gt,
                    scale_to_gt_ratio=ratio,
                    scale_to_gt_err=err,
                    scale_to_gt_log_err=log_err,
                    scale_to_gt_mult_err=mult_err,
                )
            )
    return out


def _select_representatives(rows: List[Row]) -> Dict[str, Row]:
    valid = [r for r in rows if r.scale_to_gt_err is not None and math.isfinite(float(r.scale_to_gt_err))]
    if not valid:
        raise ValueError("no valid rows with scale_to_gt_err (missing GT? missing scale?)")
    valid.sort(key=lambda r: float(r.scale_to_gt_err))  # best -> worst
    best = valid[0]
    worst = valid[-1]
    median = valid[len(valid) // 2]
    return {"best_scale": best, "median_scale": median, "worst_scale": worst}


def _as_dict(r: Row) -> Dict[str, Any]:
    return {
        "sequence": r.sequence,
        "frame": r.frame,
        "main_agent": r.main_agent,
        "coop_agents": list(r.coop_agents),
        "pose_abs_m": r.pose_abs_m,
        "pose_rot_deg": r.pose_rot_deg,
        "depth_rmse": r.depth_rmse,
        "depth_mae": r.depth_mae,
        "depth_rel": r.depth_rel,
        "chamfer_pred_to_gt": r.chamfer_pred_to_gt,
        "chamfer_gt_to_pred": r.chamfer_gt_to_pred,
        "bev_iou_raw": r.bev_iou_raw,
        "scale_ratio_mean": r.scale_ratio_mean,
        "scale_err": r.scale_err,
        "gt_scale": r.gt_scale,
        "scale_to_gt_ratio": r.scale_to_gt_ratio,
        "scale_to_gt_err": r.scale_to_gt_err,
        "scale_to_gt_log_err": r.scale_to_gt_log_err,
        "scale_to_gt_mult_err": r.scale_to_gt_mult_err,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_run_dir", type=Path, required=True, help="Existing eval run dir containing <model>/*_metrics.csv")
    ap.add_argument("--model", type=str, default="geom_model")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--contract_json", type=Path, required=True, help="GT scale contract JSON (gt_scale_single/coop)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output root for representatives + selection json")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--images_root", type=Path, default=Path("map-anything/data/opv2v_images"))
    ap.add_argument("--depth_root", type=Path, default=Path("map-anything/data/opv2v_depth"))
    ap.add_argument("--modes", nargs="+", default=["single", "coop"], choices=["single", "coop"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model_task", type=str, default="posed_sfm")
    ap.add_argument("--keep_camera_poses", action="store_true")
    args = ap.parse_args()

    eval_run_dir = args.eval_run_dir
    model_key = str(args.model)
    gt_map = _load_contract_gt(args.contract_json)

    # Dynamically import `batch_eval.py` so we reuse exact data preprocessing and PCD export logic.
    batch_eval_path = Path(__file__).resolve().parent / "batch_eval.py"
    import importlib.util  # noqa: E402

    spec = importlib.util.spec_from_file_location("_batch_eval_mod", batch_eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load batch_eval module from {batch_eval_path}")
    be = importlib.util.module_from_spec(spec)
    sys.modules["_batch_eval_mod"] = be
    spec.loader.exec_module(be)  # type: ignore[attr-defined]

    device = torch.device(str(args.device))

    # Init model once.
    cfg = {
        "path": str(be.REPO_ROOT / "configs/train.yaml"),
        "checkpoint_path": str(args.ckpt),
        "config_overrides": [
            "machine=local_a800",
            "dataset=opv2v_ft_a800",
            "model=mapanything",
            f"model/task={args.model_task}",
            "model.encoder.uses_torch_hub=false",
            "loss=overall_loss",
        ],
    }
    model = be.initialize_mapanything_local(cfg, device)
    model.eval()

    selection: Dict[str, Any] = {
        "eval_run_dir": str(eval_run_dir),
        "model": model_key,
        "ckpt": str(args.ckpt),
        "contract_json": str(args.contract_json),
        "split": str(args.split),
        "modes": list(args.modes),
        "representatives": {},
    }

    for mode in args.modes:
        csv_path = eval_run_dir / model_key / f"{mode}_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing metrics csv: {csv_path}")
        rows = _parse_metrics_csv(csv_path, mode=mode, gt_map=gt_map)
        reps = _select_representatives(rows)
        selection["representatives"][mode] = {k: _as_dict(v) for k, v in reps.items()}

        out_rep_dir = args.out_dir / model_key / f"{mode}_scale_representatives"
        out_rep_dir.mkdir(parents=True, exist_ok=True)

        for tag, r in reps.items():
            info = be.FrameInfo(
                sequence=r.sequence,
                frame=r.frame,
                main_agent=r.main_agent,
                coop_agents=tuple(r.coop_agents),
            )
            if mode == "single":
                raw_views, _camera_info = be.build_single_raw(args.images_root, args.depth_root, args.split, info)
            else:
                raw_views, _camera_info = be.build_coop_raw(args.images_root, args.depth_root, args.split, info)

            gt_ref_pose_cv = None
            if raw_views and "camera_poses" in raw_views[0]:
                gt_ref_pose_cv = np.asarray(raw_views[0]["camera_poses"], dtype=np.float32)

            processed = be.preprocess_inputs(raw_views)
            for view in processed:
                view.pop("depth_z", None)
            if not args.keep_camera_poses:
                be.strip_external_calibration_inputs(processed)

            with torch.no_grad():
                preds = model.infer(processed, memory_efficient_inference=True)

            if gt_ref_pose_cv is None:
                pts, _ = be.predictions_to_pointcloud(preds, colorize=False)
            else:
                pts, _ = be.predictions_to_ego_pointcloud(preds, gt_ref_pose_cv, colorize=False)
            pts = be._points_opencv_to_carla(pts)

            out_path = out_rep_dir / f"{r.sequence}_{r.frame}_{tag}.pcd"
            be.save_point_cloud(out_path, pts)

    _dump_json(args.out_dir / "selected_scale_representatives.json", selection)
    print(f"[OK] wrote representatives under: {args.out_dir}")


if __name__ == "__main__":
    main()
