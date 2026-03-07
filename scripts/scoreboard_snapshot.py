#!/usr/bin/env python3
"""
Build a compact metrics scoreboard from existing eval artifacts (read-only).

This script does NOT run model inference. It only reads JSON outputs under
`map-anything/eval_runs/**` and prints a markdown scoreboard.

It also performs basic fairness checks for the shared test50 sample:
- pinhole test50 frames (sequence, frame, main_agent, coop_agents)
- cyl test50 per_frame frames (sequence, frame)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt(x: Any, *, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if x != x:
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def _sf_set_from_pinhole_frames(frames: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
    out = set()
    for it in frames:
        if not isinstance(it, dict):
            continue
        seq = it.get("sequence")
        frame = it.get("frame")
        if seq is None or frame is None:
            continue
        out.add((str(seq), str(frame)))
    return out


def _sf_set_from_cyl_per_frame(per_frame: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
    out = set()
    for it in per_frame:
        if not isinstance(it, dict):
            continue
        seq = it.get("sequence")
        frame = it.get("frame")
        if seq is None or frame is None:
            continue
        out.add((str(seq), str(frame)))
    return out


def _check_main_agent_is_min(frames: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (total, violations)."""
    total = 0
    viol = 0
    for it in frames:
        if not isinstance(it, dict):
            continue
        ma = it.get("main_agent")
        coop = it.get("coop_agents") or []
        if ma is None or not coop:
            continue
        total += 1
        coop_ids = [str(x) for x in coop]
        if str(ma) != min(coop_ids):
            viol += 1
    return total, viol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pinhole_single",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "det_e2e_v5_eval_t005" / "summary_test.json",
    )
    ap.add_argument(
        "--pinhole_coop",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "det_e2e_v5_coop_eval_t005" / "summary_test.json",
    )
    ap.add_argument(
        "--pinhole_stage2_baseline",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "stage2_baseline_eval" / "summary_test.json",
    )
    ap.add_argument(
        "--cyl_test50_coop",
        type=Path,
        default=REPO_ROOT
        / "eval_runs"
        / "cyl_det_eval"
        / "det_head_test50_coop_v3_hardneg_t05_imgonly.json",
    )
    ap.add_argument(
        "--cyl_test50_single",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "cyl_det_eval" / "det_head_test50_single_v2.json",
    )
    ap.add_argument(
        "--cyl_pose_demo",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "demo_alignment" / "metrics_pose_r60_v1_recheck.json",
    )
    ap.add_argument(
        "--cyl_geom_demo",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "demo_alignment" / "metrics_geom_lidar_v1_r60_baseline.json",
    )
    args = ap.parse_args()

    pin_single = _load_json(args.pinhole_single)
    pin_coop = _load_json(args.pinhole_coop)
    cyl_coop = _load_json(args.cyl_test50_coop)
    cyl_single = _load_json(args.cyl_test50_single)
    pin_stage2 = _load_json(args.pinhole_stage2_baseline)
    cyl_pose = _load_json(args.cyl_pose_demo)
    cyl_geom = _load_json(args.cyl_geom_demo)

    pin_frames = pin_single.get("frames") or []
    cyl_sf_coop = _sf_set_from_cyl_per_frame(cyl_coop.get("per_frame") or [])
    cyl_sf_single = _sf_set_from_cyl_per_frame(cyl_single.get("per_frame") or [])
    pin_sf = _sf_set_from_pinhole_frames(pin_frames if isinstance(pin_frames, list) else [])

    total, viol = _check_main_agent_is_min(pin_frames if isinstance(pin_frames, list) else [])

    print("## Fairness Checks")
    print(f"- pinhole test50 frames: {len(pin_sf)}")
    print(f"- cyl test50 per_frame (coop): {len(cyl_sf_coop)} match={pin_sf == cyl_sf_coop}")
    print(f"- cyl test50 per_frame (single): {len(cyl_sf_single)} match={pin_sf == cyl_sf_single}")
    print(f"- pinhole main_agent == min(coop_agents): total={total} violations={viol}")
    print("")

    print("## (A) Test50 Detection (Most Comparable)")
    print("| Route | Mode | det_ap_iou | det_precision_iou | det_recall_iou | det_mean_iou | Evidence |")
    print("| --- | --- | ---:| ---:| ---:| ---:| --- |")
    print(
        "| Pinhole MapAnythingDet (e2e_v5) | single | "
        + " | ".join(
            [
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "det_ap_iou"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "det_precision_iou"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "det_recall_iou"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "det_mean_iou"]), nd=6),
                f"`{args.pinhole_single.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print(
        "| Pinhole MapAnythingDet (e2e_v5) | coop | "
        + " | ".join(
            [
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "det_ap_iou"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "det_precision_iou"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "det_recall_iou"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "det_mean_iou"]), nd=6),
                f"`{args.pinhole_coop.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print(
        "| Cyl det head (test50 finetune) | coop (num_views=2) | "
        + " | ".join(
            [
                _fmt(cyl_coop.get("det_ap_iou"), nd=6),
                _fmt(cyl_coop.get("det_precision_iou"), nd=6),
                _fmt(cyl_coop.get("det_recall_iou"), nd=6),
                _fmt(cyl_coop.get("det_mean_iou"), nd=6),
                f"`{args.cyl_test50_coop.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print(
        "| Cyl det head (test50) | single (num_views=1) | "
        + " | ".join(
            [
                _fmt(cyl_single.get("det_ap_iou"), nd=6),
                _fmt(cyl_single.get("det_precision_iou"), nd=6),
                _fmt(cyl_single.get("det_recall_iou"), nd=6),
                _fmt(cyl_single.get("det_mean_iou"), nd=6),
                f"`{args.cyl_test50_single.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print("")

    print("## (B) Pinhole Geometry (Same Test50)")
    print(
        "| Route | Mode | pose_abs_mean(m) | pose_rot_mean(deg) | depth_rel_mean | depth_rmse_mean | scale_err_mean | scale_ratio_mean | Evidence |"
    )
    print("| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| --- |")
    print(
        "| Pinhole MapAnythingDet (e2e_v5) | single | "
        + " | ".join(
            [
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "pose_abs_mean"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "pose_rot_mean"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "depth_rel_mean"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "depth_rmse_mean"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "scale_err_mean"]), nd=6),
                _fmt(_get_nested(pin_single, ["metrics", "e2e_v5", "single", "scale_ratio_mean"]), nd=6),
                f"`{args.pinhole_single.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print(
        "| Pinhole MapAnythingDet (e2e_v5) | coop | "
        + " | ".join(
            [
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "pose_abs_mean"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "pose_rot_mean"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "depth_rel_mean"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "depth_rmse_mean"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "scale_err_mean"]), nd=6),
                _fmt(_get_nested(pin_coop, ["metrics", "e2e_v5", "coop", "scale_ratio_mean"]), nd=6),
                f"`{args.pinhole_coop.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print("")

    print("## (C) Pinhole Stage2 Coop Geometry Baseline (20 frames)")
    print("| Route | frames | chamfer_pred_to_gt_mean | bev_iou_raw_mean | pose_abs_mean(m) | scale_err_mean | Evidence |")
    print("| --- | ---:| ---:| ---:| ---:| ---:| --- |")
    print(
        "| Pinhole stage2 baseline | "
        + " | ".join(
            [
                _fmt(_get_nested(pin_stage2, ["metrics", "stage2", "coop", "frames"]), nd=0),
                _fmt(_get_nested(pin_stage2, ["metrics", "stage2", "coop", "chamfer_pred_to_gt_mean"]), nd=6),
                _fmt(_get_nested(pin_stage2, ["metrics", "stage2", "coop", "bev_iou_raw_mean"]), nd=6),
                _fmt(_get_nested(pin_stage2, ["metrics", "stage2", "coop", "pose_abs_mean"]), nd=6),
                _fmt(_get_nested(pin_stage2, ["metrics", "stage2", "coop", "scale_err_mean"]), nd=6),
                f"`{args.pinhole_stage2_baseline.relative_to(REPO_ROOT)}`",
            ]
        )
        + " |"
    )
    print("")

    print("## (D) Cyl Coop Pose/Geom Decomposition (Validate 5-frame demo)")
    print("| ckpt | pose_err_rel_trans_mean(m) | pose_err_rel_rot_mean(deg) | chamfer_predpose_mean | chamfer_gtpose_mean | bev_iou_predpose_mean | bev_iou_gtpose_mean | Evidence |")
    print("| --- | ---:| ---:| ---:| ---:| ---:| ---:| --- |")
    for name, d, path in [
        ("pose_r60_v1", cyl_pose, args.cyl_pose_demo),
        ("geom_lidar_v1", cyl_geom, args.cyl_geom_demo),
    ]:
        print(
            f"| {name} | "
            + " | ".join(
                [
                    _fmt(d.get("pose_err_rel_trans_mean"), nd=6),
                    _fmt(d.get("pose_err_rel_rot_mean"), nd=6),
                    _fmt(d.get("chamfer_predpose_mean"), nd=6),
                    _fmt(d.get("chamfer_gtpose_mean"), nd=6),
                    _fmt(d.get("bev_iou_predpose_mean"), nd=6),
                    _fmt(d.get("bev_iou_gtpose_mean"), nd=6),
                    f"`{path.relative_to(REPO_ROOT)}`",
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
