#!/usr/bin/env python3
"""
Scan `eval_runs/**/summary_test.json` and build a fair pinhole test50 scoreboard.

Definition of "fair" here:
  - split == "test"
  - frames list matches the canonical 50-frame set from `det_e2e_v5_eval_t005/summary_test.json`

This script is read-only (no inference). It is intended to answer:
  - Among all pinhole eval runs that used the SAME 50 frames, which configs/models actually improved det AP?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frames_hash(frames: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        seq = f.get("sequence")
        fr = f.get("frame")
        if seq is None or fr is None:
            continue
        items.append(f"{seq}/{fr}")
    items.sort()
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if x != x:
            return "nan"
        if not math.isfinite(x):
            return "inf"
        return f"{x:.{nd}f}"
    return str(x)


@dataclass(frozen=True)
class Row:
    path: Path
    model: str
    mode: str
    det_ap_iou: float
    det_precision_iou: Optional[float]
    det_recall_iou: Optional[float]
    det_mean_iou: Optional[float]
    pose_abs_mean: Optional[float]
    pose_rot_mean: Optional[float]
    scale_err_mean: Optional[float]
    scale_ratio_mean: Optional[float]


def _extract_rows(summary_path: Path, d: Dict[str, Any]) -> List[Row]:
    out: List[Row] = []
    metrics = d.get("metrics") or {}
    if not isinstance(metrics, dict):
        return out
    for model_key, per_mode in metrics.items():
        if not isinstance(per_mode, dict):
            continue
        for mode, m in per_mode.items():
            if not isinstance(m, dict):
                continue
            ap = m.get("det_ap_iou")
            if not isinstance(ap, (int, float)) or not math.isfinite(ap):
                continue
            out.append(
                Row(
                    path=summary_path,
                    model=str(model_key),
                    mode=str(mode),
                    det_ap_iou=float(ap),
                    det_precision_iou=m.get("det_precision_iou"),
                    det_recall_iou=m.get("det_recall_iou"),
                    det_mean_iou=m.get("det_mean_iou"),
                    pose_abs_mean=m.get("pose_abs_mean"),
                    pose_rot_mean=m.get("pose_rot_mean"),
                    scale_err_mean=m.get("scale_err_mean"),
                    scale_ratio_mean=m.get("scale_ratio_mean"),
                )
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--canonical",
        type=Path,
        default=REPO_ROOT / "eval_runs" / "det_e2e_v5_eval_t005" / "summary_test.json",
        help="Canonical summary whose `frames` defines the test50 sample.",
    )
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "eval_runs")
    args = ap.parse_args()

    can = _load_json(args.canonical)
    can_frames = can.get("frames") or []
    if not isinstance(can_frames, list) or not can_frames:
        raise SystemExit(f"canonical summary has no frames list: {args.canonical}")
    can_hash = _frames_hash(can_frames)

    rows: List[Row] = []
    for summary_path in args.root.rglob("summary_test.json"):
        try:
            d = _load_json(summary_path)
        except Exception:
            continue
        if d.get("split") != "test":
            continue
        frames = d.get("frames") or []
        if not isinstance(frames, list) or len(frames) != len(can_frames):
            continue
        if _frames_hash(frames) != can_hash:
            continue
        rows.extend(_extract_rows(summary_path, d))

    print("## Canonical Test50 Contract")
    print(f"- canonical: `{args.canonical.relative_to(REPO_ROOT)}`")
    print(f"- frames: {len(can_frames)}")
    print(f"- frames_hash_md5: `{can_hash}`")
    print("")

    def show(mode: str) -> None:
        rs = [r for r in rows if r.mode == mode]
        rs.sort(key=lambda r: r.det_ap_iou, reverse=True)
        print(f"## Top-{args.topk} det_ap_iou (mode={mode}, same frames)")
        print(
            "| det_ap_iou | model | run_dir | det_precision_iou | det_recall_iou | det_mean_iou | pose_abs_mean | scale_err_mean | scale_ratio_mean | evidence |"
        )
        print("| ---:| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| --- |")
        for r in rs[: args.topk]:
            run_dir = r.path.parent.name
            print(
                "| "
                + " | ".join(
                    [
                        _fmt(r.det_ap_iou),
                        r.model,
                        run_dir,
                        _fmt(r.det_precision_iou),
                        _fmt(r.det_recall_iou),
                        _fmt(r.det_mean_iou),
                        _fmt(r.pose_abs_mean),
                        _fmt(r.scale_err_mean),
                        _fmt(r.scale_ratio_mean),
                        f"`{r.path.relative_to(REPO_ROOT)}`",
                    ]
                )
                + " |"
            )
        print("")

    show("single")
    show("coop")


if __name__ == "__main__":
    main()

