#!/usr/bin/env python3
"""Estimate det AP stability vs subset size from a saved det cache (.npz).

This is a *no-inference* analysis tool. It consumes the compact caches produced by:
  `scripts/batch_eval.py --det_metrics --save_det_cache`

and then re-scores AP on many random subsets of frames (without replacement).

Why this matters
----------------
If your main det contract is small (e.g., 50 frames), AP can have non-trivial
sampling variance. This tool helps answer "is 50 enough?" quantitatively.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _ap_from_scores_tp(
    scores: np.ndarray,
    tp: np.ndarray,
    tie_break: np.ndarray | None = None,
    *,
    total_gt: int,
) -> float:
    """Compute AP (single IoU threshold) given per-pred TP flags and scores."""
    total_gt = int(total_gt)
    if total_gt <= 0:
        return float("nan")
    if scores.size == 0:
        return 0.0

    if tie_break is None:
        tie_break = np.arange(scores.size, dtype=np.int64)
    tie_break = tie_break.astype(np.int64, copy=False)

    # Stable tie-break: use the original insertion order as secondary key.
    sort_idx = np.lexsort((tie_break, -scores.astype(np.float64)))
    tp_sorted = tp[sort_idx].astype(np.float64)
    fp_sorted = 1.0 - tp_sorted

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)

    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _stats(xs: List[float]) -> dict:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    ys.sort()
    if not ys:
        return {}
    def q(p: float) -> float:
        i = int(round((len(ys) - 1) * p))
        return float(ys[i])
    mean = float(sum(ys) / len(ys))
    var = float(sum((y - mean) ** 2 for y in ys) / len(ys))
    return {
        "n": int(len(ys)),
        "mean": mean,
        "std": float(var ** 0.5),
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "min": float(ys[0]),
        "max": float(ys[-1]),
    }


def _load_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    npz = np.load(path, allow_pickle=False)
    scores = npz["pred_scores"].astype(np.float64)
    tp = npz["pred_tp"].astype(np.int8)
    frame_idx = npz["pred_frame_idx"].astype(np.int32)
    pred_order = npz["pred_order"].astype(np.int64) if "pred_order" in npz else np.arange(scores.size, dtype=np.int64)
    gt_counts = npz["gt_counts"].astype(np.int32)
    iou_thresh = float(npz["iou_thresh"][0]) if "iou_thresh" in npz else float("nan")
    # Guard: some predictions might have frame_idx=-1 if cache was built with mismatched keys.
    keep = frame_idx >= 0
    return scores[keep], tp[keep], frame_idx[keep], gt_counts, pred_order[keep], iou_thresh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det_cache_npz", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--subset_sizes", type=str, default="50,100,200")
    ap.add_argument("--num_trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scores, tp, frame_idx, gt_counts, pred_order, iou_thresh = _load_cache(args.det_cache_npz)
    num_frames = int(gt_counts.shape[0])

    # Pre-group preds by frame for fast subset concat.
    scores_by_frame: List[List[float]] = [[] for _ in range(num_frames)]
    tp_by_frame: List[List[int]] = [[] for _ in range(num_frames)]
    order_by_frame: List[List[int]] = [[] for _ in range(num_frames)]
    for s, t, fi, oi in zip(scores.tolist(), tp.tolist(), frame_idx.tolist(), pred_order.tolist()):
        if 0 <= int(fi) < num_frames:
            scores_by_frame[int(fi)].append(float(s))
            tp_by_frame[int(fi)].append(int(t))
            order_by_frame[int(fi)].append(int(oi))

    scores_by_frame_np = [np.asarray(v, dtype=np.float64) for v in scores_by_frame]
    tp_by_frame_np = [np.asarray(v, dtype=np.int8) for v in tp_by_frame]
    order_by_frame_np = [np.asarray(v, dtype=np.int64) for v in order_by_frame]

    total_gt_full = int(gt_counts.sum())
    ap_full = _ap_from_scores_tp(scores, tp, pred_order, total_gt=total_gt_full)

    subset_sizes = [int(x) for x in args.subset_sizes.split(",") if x.strip()]
    rng = random.Random(int(args.seed))

    lines: List[str] = []
    lines.append("# Det AP Subset Sensitivity (No-Inference)")
    lines.append("")
    lines.append(f"- cache: `{args.det_cache_npz}`")
    lines.append(f"- iou_thresh: `{iou_thresh}`")
    lines.append(f"- frames: `{num_frames}`")
    lines.append(f"- total_gt: `{total_gt_full}`")
    lines.append(f"- total_preds: `{int(scores.shape[0])}`")
    lines.append(f"- AP(full): `{ap_full:.6f}`")
    lines.append("")

    lines.append("## Monte Carlo Subset Results (without replacement)")
    lines.append("")
    lines.append("| subset_frames | trials | AP mean | AP std | p05 | p50 | p95 | min | max |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for n in subset_sizes:
        n = int(n)
        if n <= 0:
            continue
        if n > num_frames:
            lines.append(f"| {n} | 0 | NA | NA | NA | NA | NA | NA | NA |")
            continue
        aps = []
        for _ in range(int(args.num_trials)):
            subset = rng.sample(range(num_frames), n)
            total_gt = int(gt_counts[subset].sum())
            if total_gt <= 0:
                aps.append(float("nan"))
                continue
            sc = np.concatenate([scores_by_frame_np[i] for i in subset], axis=0) if subset else np.zeros((0,))
            tp2 = np.concatenate([tp_by_frame_np[i] for i in subset], axis=0) if subset else np.zeros((0,), dtype=np.int8)
            oi2 = np.concatenate([order_by_frame_np[i] for i in subset], axis=0) if subset else np.zeros((0,), dtype=np.int64)
            aps.append(_ap_from_scores_tp(sc, tp2, oi2, total_gt=total_gt))
        st = _stats(aps)
        if not st:
            lines.append(f"| {n} | {args.num_trials} | NA | NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {n} | {st['n']} | {st['mean']:.6f} | {st['std']:.6f} | {st['p05']:.6f} | {st['p50']:.6f} | {st['p95']:.6f} | {st['min']:.6f} | {st['max']:.6f} |"
        )

    lines.append("")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] wrote:", args.out_md)


if __name__ == "__main__":
    main()
