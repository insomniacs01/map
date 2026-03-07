#!/usr/bin/env python3
"""Write a small, stable "quick report" comparing two batch-eval summaries.

This is a lightweight helper for iterative geometry ablations where we want:
  - an easy-to-skim text report
  - consistent coop-first pass/fail logic
  - guardrails against depth / (cross-agent) pose regressions

It is intentionally opinionated and matches the logic in
`scripts/run_geom_ablation_test500.sh`, but can be used standalone for eval-only runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_mode(doc: Dict[str, Any], *, model: str, mode: str) -> bool:
    try:
        m = doc["metrics"][model][mode]
    except Exception:
        return False
    return isinstance(m, dict) and len(m) > 0


def _m(doc: Dict[str, Any], *, model: str, mode: str, key: str) -> float:
    return float(doc["metrics"][model][mode][key])


def _mf(doc: Dict[str, Any], *, model: str, mode: str, key: str) -> float:
    try:
        val = doc["metrics"][model][mode].get(key)
    except Exception:
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", type=str, default="geom_model")
    ap.add_argument("--modes", nargs="*", default=["coop"])
    ap.add_argument("--coop_first", action="store_true", default=True)
    ap.add_argument("--keep_camera_poses", action="store_true", default=False)
    ap.add_argument("--target_scale_mult", type=float, default=1.3)
    ap.add_argument("--depth_guard_factor", type=float, default=1.10)
    ap.add_argument("--pose_guard_factor", type=float, default=1.10)
    args = ap.parse_args()

    summary_path = args.summary.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    report_path = args.out
    if report_path is None:
        report_path = summary_path.parent / "quick_report.txt"
    report_path = report_path.expanduser().resolve()

    cur = _load_json(summary_path)
    base = _load_json(baseline_path)

    eval_modes = [m for m in args.modes if m]
    if not eval_modes:
        eval_modes = ["single", "coop"]

    modes_to_check = ["coop"] if args.coop_first else eval_modes

    lines: List[str] = []
    lines.append(f"summary={summary_path}")
    lines.append(f"baseline={baseline_path}")
    lines.append(f"eval_modes={' '.join(eval_modes)}")
    lines.append(f"keep_camera_poses={bool(args.keep_camera_poses)}")
    lines.append(f"coop_first={bool(args.coop_first)}")

    ok = True
    for mode in modes_to_check:
        if not _has_mode(cur, model=args.model, mode=mode):
            ok = False
            lines.append(f"{mode}: missing metrics (all frames failed or filtered); pass=False")
            continue

        cur_mult = _m(cur, model=args.model, mode=mode, key="scale_to_gt_mult_err_mean")
        cur_ratio = _m(cur, model=args.model, mode=mode, key="scale_to_gt_ratio_mean")
        cur_depth = _m(cur, model=args.model, mode=mode, key="depth_rel_mean")
        base_mult = _m(base, model=args.model, mode=mode, key="scale_to_gt_mult_err_mean")
        base_depth = _m(base, model=args.model, mode=mode, key="depth_rel_mean")

        improve = (base_mult - cur_mult) / max(1e-8, base_mult)

        # Scale criterion: absolute threshold (OPV2V-correct, ratio-to-GT).
        scale_ok = bool(cur_mult <= float(args.target_scale_mult))
        scale_details = f"scale_ok={scale_ok} (<= {float(args.target_scale_mult):.4f})"

        # Pose guardrail:
        # - Coop-first deployments care about cross-agent relative pose, not single pose.
        # - If cross-agent metrics are missing, fall back to pose_abs_mean.
        pose_ok = True
        pose_details = "pose_guard=ignored"
        if mode == "coop" and (not args.keep_camera_poses):
            cur_cross_t = _mf(cur, model=args.model, mode=mode, key="cross_agent_pose_trans_mean")
            cur_cross_r = _mf(cur, model=args.model, mode=mode, key="cross_agent_pose_rot_mean")
            base_cross_t = _mf(base, model=args.model, mode=mode, key="cross_agent_pose_trans_mean")
            base_cross_r = _mf(base, model=args.model, mode=mode, key="cross_agent_pose_rot_mean")
            if math.isfinite(cur_cross_t) and math.isfinite(base_cross_t):
                thr_t = base_cross_t * float(args.pose_guard_factor)
                ok_t = cur_cross_t <= thr_t
                pose_details = f"cross_trans={cur_cross_t:.4f} (<= {thr_t:.4f} => {ok_t})"
                ok_r = True
                if math.isfinite(cur_cross_r) and math.isfinite(base_cross_r):
                    thr_r = base_cross_r * float(args.pose_guard_factor)
                    ok_r = cur_cross_r <= thr_r
                    pose_details += f", cross_rot={cur_cross_r:.3f} (<= {thr_r:.3f} => {ok_r})"
                pose_ok = bool(ok_t and ok_r)
            else:
                cur_pose = _m(cur, model=args.model, mode=mode, key="pose_abs_mean")
                base_pose = _m(base, model=args.model, mode=mode, key="pose_abs_mean")
                thr_pose = base_pose * float(args.pose_guard_factor)
                pose_ok = bool(cur_pose <= thr_pose)
                pose_details = f"pose_abs={cur_pose:.4f} (<= {thr_pose:.4f} => {pose_ok})"

        # Depth guardrail.
        depth_thr = base_depth * float(args.depth_guard_factor)
        depth_ok = bool(cur_depth <= depth_thr)

        trial_ok = scale_ok and pose_ok and depth_ok
        ok = ok and trial_ok
        lines.append(
            f"{mode}: mult={cur_mult:.4f} (base={base_mult:.4f}, improve={improve*100:.1f}%), "
            f"ratio={cur_ratio:.4f}, {scale_details}, {pose_details}, "
            f"depth={cur_depth:.4f} (<= {depth_thr:.4f} => {depth_ok}), pass={trial_ok}"
        )

    lines.append(f"overall_pass={ok}")
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

