#!/usr/bin/env python3
"""Backfill human-friendly scale multiplicative error fields in eval summaries.

Why
---
Historically we stored:
  - `scale_eq_rel_err_mean = exp(scale_log_err_mean) - 1`
  - `scale_to_gt_eq_rel_err_mean = exp(scale_to_gt_log_err_mean) - 1`

These are perfectly valid, but many users prefer a "1 is best" metric:
  - `scale_mult_err_mean = exp(scale_log_err_mean)`
  - `scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)`

Since per-frame `*_eq_rel_err = expm1(...)`, the mean multiplicative error is simply:
  - `*_mult_err_mean = *_eq_rel_err_mean + 1`

This script adds the missing keys into `eval_runs/**/summary_test.json`
without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _maybe_backfill_mode(m: Dict[str, Any]) -> int:
    changed = 0

    if "scale_mult_err_mean" not in m:
        eq_rel = m.get("scale_eq_rel_err_mean")
        if _finite(eq_rel):
            m["scale_mult_err_mean"] = float(eq_rel) + 1.0
            changed += 1

    if "scale_to_gt_mult_err_mean" not in m:
        eq_rel_gt = m.get("scale_to_gt_eq_rel_err_mean")
        if _finite(eq_rel_gt):
            m["scale_to_gt_mult_err_mean"] = float(eq_rel_gt) + 1.0
            changed += 1

    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "eval_runs")
    ap.add_argument("--run_prefix", type=str, default="", help="Only process runs whose dir name contains this substring")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    total_files = 0
    total_changed = 0

    for summary_path in root.rglob("summary_test.json"):
        run_dir = summary_path.parent.name
        if args.run_prefix and args.run_prefix not in run_dir:
            continue

        try:
            payload = _load_json(summary_path)
        except Exception:
            continue

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue

        changed = 0
        for _model_key, per_mode in metrics.items():
            if not isinstance(per_mode, dict):
                continue
            for _mode, m in per_mode.items():
                if not isinstance(m, dict):
                    continue
                changed += _maybe_backfill_mode(m)

        if changed > 0:
            total_files += 1
            total_changed += changed
            if not args.dry_run:
                payload.setdefault("run_info", {})
                if isinstance(payload["run_info"], dict):
                    payload["run_info"]["scale_mult_err_backfilled"] = "2026-02-17"
                _dump_json(summary_path, payload)
                print(f"[OK] backfilled {changed} fields: {summary_path.relative_to(REPO_ROOT)}")
            else:
                print(f"[DRY] would backfill {changed} fields: {summary_path.relative_to(REPO_ROOT)}")

    print(f"done. files_changed={total_files} fields_added={total_changed}")


if __name__ == "__main__":
    main()

