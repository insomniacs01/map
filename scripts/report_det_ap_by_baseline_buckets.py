#!/usr/bin/env python3
"""Report det AP by OPV2V cross-agent baseline-distance buckets (no inference).

This script consumes the compact caches produced by:
  `map-anything/scripts/batch_eval.py --det_metrics --save_det_cache`

and re-scores AP on bucket subsets of the *same fixed frame contract*.

Why this matters
----------------
- OPV2V coop difficulty is strongly correlated with cross-agent baseline distance.
- A small det contract (e.g. 50 frames) will under-sample the long-baseline tail
  and can hide regressions that only show up when vehicles are far apart.
- By caching TP flags + scores, we can do fair bucket analysis without re-running
  expensive inference.

Inputs
------
- --det_cache_npz: one of det_ap_cache_{single,coop}.npz
- --frames_json: the frames contract used to build the cache (ordering must match)
- --bucket_report_json: produced by bucket_frames_by_baseline_distance.py
  (contains bucket_paths to per-bucket frame jsons)

Output
------
Writes a markdown report with:
  - AP(full)
  - per-bucket AP + frame/GT/pred counts (+ optional distance stats if present)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


FrameKey = Tuple[str, str, str, Tuple[str, ...]]  # (sequence, frame, main_agent, coop_agents tuple)


def _frame_key(fr: Mapping[str, Any]) -> FrameKey:
    seq = str(fr.get("sequence", ""))
    frame = str(fr.get("frame", ""))
    main = str(fr.get("main_agent", ""))
    coop_raw = fr.get("coop_agents") or []
    if isinstance(coop_raw, (list, tuple)):
        coop = tuple(str(a) for a in coop_raw)
    else:
        coop = (str(coop_raw),)
    return seq, frame, main, coop


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _quantile(sorted_xs: List[float], p: float) -> float:
    if not sorted_xs:
        return float("nan")
    i = int(round((len(sorted_xs) - 1) * p))
    return float(sorted_xs[i])


def _stats(xs: Iterable[float]) -> Dict[str, float]:
    xs2 = [float(x) for x in xs if math.isfinite(float(x))]
    xs2.sort()
    if not xs2:
        return {}
    return {
        "n": float(len(xs2)),
        "mean": float(sum(xs2) / len(xs2)),
        "p50": _quantile(xs2, 0.50),
        "p95": _quantile(xs2, 0.95),
        "min": float(xs2[0]),
        "max": float(xs2[-1]),
    }


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

    # Stable tie-break: use original insertion order as secondary key.
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


def _load_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    npz = np.load(path, allow_pickle=False)
    scores = npz["pred_scores"].astype(np.float64)
    tp = npz["pred_tp"].astype(np.int8)
    frame_idx = npz["pred_frame_idx"].astype(np.int32)
    pred_order = npz["pred_order"].astype(np.int64) if "pred_order" in npz else np.arange(scores.size, dtype=np.int64)
    gt_counts = npz["gt_counts"].astype(np.int32)
    iou_thresh = float(npz["iou_thresh"][0]) if "iou_thresh" in npz else float("nan")
    keep = frame_idx >= 0
    return scores[keep], tp[keep], frame_idx[keep], gt_counts, pred_order[keep], iou_thresh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det_cache_npz", type=Path, required=True)
    ap.add_argument("--frames_json", type=Path, required=True)
    ap.add_argument("--bucket_report_json", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None, help="Optional machine-readable JSON output path.")
    args = ap.parse_args()

    cache_path = args.det_cache_npz.expanduser().resolve()
    frames_path = args.frames_json.expanduser().resolve()
    bucket_report_path = args.bucket_report_json.expanduser().resolve()

    scores, tp, pred_frame_idx, gt_counts, pred_order, iou_thresh = _load_cache(cache_path)
    num_frames = int(gt_counts.shape[0])

    contract = _load_json(frames_path)
    frames = contract.get("frames", contract)
    if not isinstance(frames, list):
        raise SystemExit(f"bad --frames_json (expected list under 'frames'): {frames_path}")
    if len(frames) != num_frames:
        raise SystemExit(
            f"cache/contract size mismatch: cache has {num_frames} frames but frames_json has {len(frames)} frames. "
            f"cache={cache_path} frames_json={frames_path}"
        )

    contract_keys: List[FrameKey] = []
    for fr in frames:
        if not isinstance(fr, Mapping):
            raise SystemExit(f"bad frames_json entry (not a dict): {frames_path}")
        contract_keys.append(_frame_key(fr))

    bucket_report = _load_json(bucket_report_path)
    bucket_paths = [Path(p) for p in (bucket_report.get("bucket_paths") or [])]
    if not bucket_paths:
        raise SystemExit(f"bucket_report_json has no bucket_paths: {bucket_report_path}")

    # Build key->bucket mapping using per-bucket frame jsons.
    key_to_bucket: Dict[FrameKey, str] = {}
    bucket_meta: List[Dict[str, Any]] = []
    for p in bucket_paths:
        doc = _load_json(p)
        frs = doc.get("frames", doc)
        if not isinstance(frs, list):
            frs = []
        keys: List[FrameKey] = []
        dists: List[float] = []
        for fr in frs:
            if not isinstance(fr, Mapping):
                continue
            k = _frame_key(fr)
            keys.append(k)
            dist = _safe_float(fr.get("baseline_distance_m"))
            if dist is not None:
                dists.append(float(dist))

        # Prefer a stable, human-readable label if present (produced by the bucketer).
        bucket_name = str(doc.get("bucket_label") or p.stem)
        for k in keys:
            prev = key_to_bucket.get(k)
            if prev is not None and prev != bucket_name:
                raise SystemExit(f"frame appears in multiple buckets: {k} -> {prev} and {bucket_name}")
            key_to_bucket[k] = bucket_name

        bucket_meta.append(
            {
                "name": bucket_name,
                "path": str(p),
                "frames": int(len(keys)),
                "dist_stats": _stats(dists) if dists else {},
            }
        )

    # Map contract indices to buckets.
    bucket_to_indices: Dict[str, List[int]] = {}
    missing = 0
    for i, k in enumerate(contract_keys):
        b = key_to_bucket.get(k)
        if b is None:
            missing += 1
            continue
        bucket_to_indices.setdefault(b, []).append(int(i))
    if missing:
        raise SystemExit(
            f"{missing}/{len(contract_keys)} frames from frames_json are not covered by bucket_report_json. "
            f"(Likely pairing drift / wrong bucket_report_json.)"
        )

    total_gt_full = int(gt_counts.sum())
    ap_full = _ap_from_scores_tp(scores, tp, pred_order, total_gt=total_gt_full)

    # Report.
    lines: List[str] = []
    lines.append("# Det AP by Cross-Agent Baseline Distance Buckets (No-Inference)")
    lines.append("")
    lines.append(f"- cache: `{cache_path}`")
    lines.append(f"- frames_json: `{frames_path}`")
    lines.append(f"- bucket_report: `{bucket_report_path}`")
    lines.append(f"- iou_thresh: `{iou_thresh}`")
    lines.append(f"- frames: `{num_frames}`")
    lines.append(f"- total_gt: `{total_gt_full}`")
    lines.append(f"- total_preds: `{int(scores.shape[0])}`")
    lines.append(f"- AP(full): `{ap_full:.6f}`")
    lines.append("")

    edges = bucket_report.get("bucket_edges_m")
    if isinstance(edges, list) and edges:
        lines.append(f"- bucket_edges_m: `{edges}`")
        lines.append("")

    lines.append("## Per-Bucket AP")
    lines.append("")
    lines.append("| bucket | frames | total_gt | total_preds | dist_mean | dist_p50 | dist_p95 | AP |")
    lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|")

    report: Dict[str, Any] = {
        "det_cache_npz": str(cache_path),
        "frames_json": str(frames_path),
        "bucket_report_json": str(bucket_report_path),
        "iou_thresh": float(iou_thresh),
        "frames": int(num_frames),
        "total_gt": int(total_gt_full),
        "total_preds": int(scores.shape[0]),
        "ap_full": float(ap_full),
        "bucket_edges_m": edges if isinstance(edges, list) else None,
        "bucket_order": [str(m.get("name")) for m in bucket_meta],
        "buckets": [],
    }

    # Stable bucket order as in the bucket_report (bucket_paths order).
    for meta in bucket_meta:
        name = str(meta["name"])
        subset = bucket_to_indices.get(name, [])
        subset = sorted(set(int(i) for i in subset))
        in_subset = np.zeros((num_frames,), dtype=bool)
        in_subset[subset] = True
        mask = in_subset[pred_frame_idx]

        total_gt = int(gt_counts[subset].sum()) if subset else 0
        sc = scores[mask]
        tp2 = tp[mask]
        oi2 = pred_order[mask]
        ap_bucket = _ap_from_scores_tp(sc, tp2, oi2, total_gt=total_gt) if total_gt > 0 else float("nan")

        dst = meta.get("dist_stats") or {}
        dist_mean = dst.get("mean")
        dist_p50 = dst.get("p50")
        dist_p95 = dst.get("p95")

        def fmt(x: Any) -> str:
            v = _safe_float(x)
            return f"{v:.2f}" if v is not None else "NA"

        ap_str = f"{ap_bucket:.6f}" if math.isfinite(float(ap_bucket)) else "NA"
        lines.append(
            f"| {name} | {len(subset)} | {total_gt} | {int(mask.sum())} | {fmt(dist_mean)} | {fmt(dist_p50)} | {fmt(dist_p95)} | {ap_str} |"
        )
        report["buckets"].append(
            {
                "bucket": name,
                "frames": int(len(subset)),
                "total_gt": int(total_gt),
                "total_preds": int(mask.sum()),
                "dist_stats": (meta.get("dist_stats") or {}),
                "ap": float(ap_bucket),
            }
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- This is a no-inference re-score using cached decoded predictions.")
    lines.append("- `AP(full)` is computed on the full contract. Bucket AP is diagnostic; do not average bucket APs to get full AP.")
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] wrote:", args.out_md)

    if args.out_json is not None:
        out_json = args.out_json.expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[OK] wrote:", out_json)


if __name__ == "__main__":
    main()
