#!/usr/bin/env python3
"""
Gate a run based on summary.json metrics.

Env thresholds (optional):
  GATE_VAL_POSE  (pose_trans_l2_m max)
  GATE_VAL_DEPTH (depth_z_mae_m max)
  GATE_TEST_POSE
  GATE_TEST_DEPTH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _get_thr(name: str):
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()

    if not args.summary.is_file():
        print("gate: summary missing", file=sys.stderr)
        sys.exit(2)

    data = json.loads(args.summary.read_text())
    eval_last = data.get("eval_last", {})
    val = (eval_last.get("validate") or eval_last.get("val") or {}).get("metrics", {}) or {}
    test = (eval_last.get("test") or {}).get("metrics", {}) or {}

    checks = []
    thr = _get_thr("GATE_VAL_POSE")
    if thr is not None:
        checks.append(("val_pose", val.get("pose_trans_l2_m"), thr))
    thr = _get_thr("GATE_VAL_DEPTH")
    if thr is not None:
        checks.append(("val_depth", val.get("depth_z_mae_m"), thr))
    thr = _get_thr("GATE_TEST_POSE")
    if thr is not None:
        checks.append(("test_pose", test.get("pose_trans_l2_m"), thr))
    thr = _get_thr("GATE_TEST_DEPTH")
    if thr is not None:
        checks.append(("test_depth", test.get("depth_z_mae_m"), thr))

    failed = []
    for name, value, limit in checks:
        if value is None:
            failed.append(f"{name}:missing")
        elif float(value) > float(limit):
            failed.append(f"{name}:{value:.4f}>{limit:.4f}")

    if failed:
        print("gate: FAIL " + ", ".join(failed), file=sys.stderr)
        sys.exit(3)
    print("gate: OK")


if __name__ == "__main__":
    main()
