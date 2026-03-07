#!/usr/bin/env python3
"""
Gate a run based on summary.json metrics.

Usage:
  gate_metrics.py --summary /path/to/summary.json [--split validate,test,train]

Config (env):
  GATE_MAX="loss=50,depth_z_mae_m=2,pose_trans_l2_m=1"
  GATE_MIN="metric_scaling_factor=0.5"
  GATE_SPLIT="validate,test,train"   # override split order
  GATE_REQUIRE_KEYS=1                # fail if a threshold key is missing

Scale metric convention:
  - `scale_to_gt_err_mean` is OPV2V-correct ratio-to-GT key (smaller, toward 0, is better).
  - `scale_err_mean` is a ratio-space key (smaller, toward 0, is better).
  - `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` is loss-space and should be
    treated as optimization-only (not ratio).
  - Optional fallback for legacy logs:
      GATE_SCALE_FALLBACK=1
      GATE_SCALE_FALLBACK_KEY=FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale
      GATE_SCALE_FALLBACK_MAX=0.0065
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_kv(s: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            continue
    return out


def _select_metrics(summary: dict, split_order: list[str]) -> tuple[str, dict]:
    eval_last = summary.get("eval_last") or {}
    for key in split_order:
        if key in eval_last and "metrics" in eval_last[key]:
            return key, eval_last[key]["metrics"]
    train_last = summary.get("train_last") or {}
    if "metrics" in train_last:
        return "train", train_last["metrics"]
    return "none", {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--split", type=str, default=os.environ.get("GATE_SPLIT", "validate,test,train"))
    args = p.parse_args()

    if not args.summary.is_file():
        print(f"[gate] summary not found: {args.summary}", file=sys.stderr)
        return 2

    data = json.loads(args.summary.read_text(encoding="utf-8"))

    split_order = [s.strip() for s in args.split.split(",") if s.strip()]
    split, metrics = _select_metrics(data, split_order)

    max_thr = _parse_kv(os.environ.get("GATE_MAX"))
    min_thr = _parse_kv(os.environ.get("GATE_MIN"))
    require_keys = os.environ.get("GATE_REQUIRE_KEYS", "0") == "1"
    scale_fallback = os.environ.get("GATE_SCALE_FALLBACK", "0") == "1"
    scale_fallback_key = os.environ.get(
        "GATE_SCALE_FALLBACK_KEY", "FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale"
    )
    scale_fallback_max = os.environ.get("GATE_SCALE_FALLBACK_MAX")
    scale_fallback_min = os.environ.get("GATE_SCALE_FALLBACK_MIN")

    failures: list[str] = []
    scale_fallback_used = False
    for k, v in max_thr.items():
        if k not in metrics:
            if (
                k == "scale_err_mean"
                and scale_fallback
                and scale_fallback_max is not None
                and scale_fallback_key in metrics
            ):
                try:
                    fb_max = float(scale_fallback_max)
                except ValueError:
                    fb_max = None
                if fb_max is not None and metrics[scale_fallback_key] > fb_max:
                    failures.append(
                        f"{scale_fallback_key}={metrics[scale_fallback_key]:.4f} > "
                        f"fallback_max={fb_max} (scale_err_mean missing)"
                    )
                scale_fallback_used = True
                continue
            if require_keys:
                failures.append(f"{k}=missing > max={v}")
            continue
        if metrics[k] > v:
            failures.append(f"{k}={metrics[k]:.4f} > max={v}")
    for k, v in min_thr.items():
        if k not in metrics:
            if (
                k == "scale_err_mean"
                and scale_fallback
                and scale_fallback_min is not None
                and scale_fallback_key in metrics
            ):
                try:
                    fb_min = float(scale_fallback_min)
                except ValueError:
                    fb_min = None
                if fb_min is not None and metrics[scale_fallback_key] < fb_min:
                    failures.append(
                        f"{scale_fallback_key}={metrics[scale_fallback_key]:.4f} < "
                        f"fallback_min={fb_min} (scale_err_mean missing)"
                    )
                scale_fallback_used = True
                continue
            if require_keys:
                failures.append(f"{k}=missing < min={v}")
            continue
        if metrics[k] < v:
            failures.append(f"{k}={metrics[k]:.4f} < min={v}")

    print(f"[gate] split={split} metrics_keys={sorted(metrics.keys())}")
    if scale_fallback_used:
        print(f"[gate] note: used fallback scale key `{scale_fallback_key}` (scale_err_mean missing)")
    if "scale_err_mean" in max_thr:
        print("[gate] note: `scale_err_mean` is ratio-space (lower toward 0)")
    if "scale_to_gt_err_mean" in max_thr:
        print("[gate] note: `scale_to_gt_err_mean` is ratio-to-GT (lower toward 0)")
    if "FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale" in max_thr:
        print(
            "[gate] note: `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` "
            "is loss-space (lower toward 0), not ratio-space"
        )
    if failures:
        print("[gate] FAIL: " + "; ".join(failures))
        return 10
    print("[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
