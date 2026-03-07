#!/usr/bin/env python3
"""Report baseline-distance bucket stats from a generic coop index parquet.

Given an index parquet with (at least) the columns:
  - split
  - baseline_distance_m
  - (optional) has_assets
  - (optional) dataset

This script prints:
  - counts per baseline-distance bucket,
  - baseline_distance_m quantiles,
  - recommended `--bucket_counts` for a target `--sample_size`
    (proportional allocation, clamped to availability, deterministic).

It can also emit a machine-readable JSON report via `--out_json`.

Self-check
----------
For a pure syntax check:
  python -m py_compile scripts/report_index_bucket_stats.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


from coop_bucket_presets import (  # noqa: E402
    FINE_EVAL_BUCKET_EDGES_STR as DEFAULT_BUCKET_EDGES_STR,
    bucket_index as _bucket_index,
    bucket_label as _bucket_label,
    parse_edges as _parse_edges,
    preset_edges_m,
)


def _allocate_bucket_counts(*, sample_size: int, bucket_sizes: Sequence[int]) -> List[int]:
    """Allocate per-bucket sample counts proportional to availability (deterministic)."""
    total = int(sum(bucket_sizes))
    if total <= 0:
        return [0 for _ in bucket_sizes]
    raw = [float(sample_size) * (float(n) / float(total)) for n in bucket_sizes]
    counts = [int(math.floor(x)) for x in raw]
    remainder = int(sample_size - sum(counts))
    frac = [raw[i] - float(counts[i]) for i in range(len(raw))]
    order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
    for i in order:
        if remainder <= 0:
            break
        counts[i] += 1
        remainder -= 1
    return counts


def _clamp_and_redistribute(*, target: Sequence[int], available: Sequence[int]) -> List[int]:
    target2 = [int(x) for x in target]
    available2 = [int(x) for x in available]
    leftover = 0
    for i in range(len(target2)):
        if target2[i] > available2[i]:
            leftover += target2[i] - available2[i]
            target2[i] = available2[i]
    if leftover <= 0:
        return target2

    cap = [available2[i] - target2[i] for i in range(len(target2))]
    order = sorted(range(len(cap)), key=lambda i: cap[i], reverse=True)
    for i in order:
        if leftover <= 0:
            break
        take = min(cap[i], leftover)
        if take > 0:
            target2[i] += int(take)
            leftover -= int(take)
    return target2


def _quantiles(xs: np.ndarray, ps: Sequence[float]) -> Dict[str, float]:
    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {}
    out: Dict[str, float] = {}
    for p in ps:
        try:
            v = float(np.quantile(xs, p))
        except Exception:
            continue
        out[f"p{int(round(p * 100)):02d}"] = v
    out["min"] = float(np.min(xs))
    out["max"] = float(np.max(xs))
    out["mean"] = float(np.mean(xs))
    out["n"] = int(xs.size)
    return out


def _read_index(index_path: Path) -> "Any":
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required to read parquet index files") from exc

    # Try to read only relevant columns.
    wanted = ["split", "baseline_distance_m", "has_assets", "dataset"]
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(index_path)
        available = set(schema.names)
        cols = [c for c in wanted if c in available]
        df = pd.read_parquet(index_path, columns=cols)
    except Exception:
        df = pd.read_parquet(index_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index_parquet", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test", help="Split name to filter by (matches the index 'split').")
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset filter (matches index column 'dataset' if present).",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        choices=("", "coarse_train", "fine_eval"),
        help=(
            "Optional bucket-edge preset. "
            "coarse_train: [0,30,60,100,inf]. "
            "fine_eval: [0,15,30,60,100,inf]. "
            "Note: cannot be used together with --bucket_edges."
        ),
    )
    ap.add_argument(
        "--bucket_edges",
        type=str,
        default="",
        help=(
            "Comma-separated bucket edges in meters. "
            f"If omitted, defaults to preset fine_eval ({DEFAULT_BUCKET_EDGES_STR})."
        ),
    )
    ap.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Target sample size used to recommend `--bucket_counts` (does not sample).",
    )
    ap.add_argument(
        "--require_assets",
        action="store_true",
        help="If index has `has_assets`, only count rows where has_assets is true (recommended for contract parity).",
    )
    ap.add_argument("--out_json", type=Path, default=None, help="Optional JSON output path for the report.")
    ap.add_argument("--self_check", action="store_true", help="Run basic sanity checks on the report.")
    args = ap.parse_args()

    index_path = args.index_parquet.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"index parquet not found: {index_path}")

    if args.preset and str(args.bucket_edges).strip():
        raise ValueError("Cannot use --preset together with --bucket_edges")
    if args.preset:
        edges = preset_edges_m(str(args.preset))
        edges_source = f"preset:{str(args.preset)}"
    elif str(args.bucket_edges).strip():
        edges = _parse_edges(str(args.bucket_edges))
        edges_source = "cli"
    else:
        edges = preset_edges_m("fine_eval")
        edges_source = "default:fine_eval"
    labels = [_bucket_label(edges, i) for i in range(len(edges) - 1)]

    df = _read_index(index_path)
    if "split" not in df.columns:
        raise ValueError(f"index parquet missing required column 'split': {index_path}")
    if "baseline_distance_m" not in df.columns:
        raise ValueError(f"index parquet missing required column 'baseline_distance_m': {index_path}")

    split = str(args.split)
    df = df[df["split"].astype(str) == split]

    dataset_filter = str(args.dataset).strip()
    if dataset_filter:
        if "dataset" not in df.columns:
            raise ValueError(f"--dataset set but index parquet missing column 'dataset': {index_path}")
        df = df[df["dataset"].astype(str) == dataset_filter]

    if args.require_assets:
        if "has_assets" not in df.columns:
            raise ValueError(f"--require_assets set but index parquet missing column 'has_assets': {index_path}")
        df = df[df["has_assets"].astype(bool)]

    total_rows = int(len(df))
    if total_rows == 0:
        msg = f"No rows for split={split!r}"
        if dataset_filter:
            msg += f" dataset={dataset_filter!r}"
        if args.require_assets:
            msg += " require_assets=True"
        msg += f" in index: {index_path}"
        raise RuntimeError(msg)

    has_assets_stats = None
    if "has_assets" in df.columns:
        raw = df["has_assets"]
        try:
            num_true = int(np.sum(raw.astype(bool).to_numpy()))
            num_false = int(len(raw) - num_true)
            has_assets_stats = {"true": num_true, "false": num_false}
        except Exception:
            has_assets_stats = None

    dist = df["baseline_distance_m"].to_numpy()
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist[np.isfinite(dist)]
    dist = dist[dist >= 0.0]
    if dist.size == 0:
        raise RuntimeError(
            f"No valid baseline_distance_m values (finite, >=0) for split={split!r} in index: {index_path}"
        )

    # Bucket counts.
    bucket_sizes = [0 for _ in range(len(edges) - 1)]
    for x in dist.tolist():
        bi = _bucket_index(edges, float(x))
        if bi is None:
            continue
        bucket_sizes[int(bi)] += 1

    # Recommended counts for make_frames_contract_from_index.py --bucket_counts.
    prop = _allocate_bucket_counts(sample_size=int(args.sample_size), bucket_sizes=bucket_sizes)
    prop_clamped = _clamp_and_redistribute(target=prop, available=bucket_sizes)

    q = _quantiles(dist, ps=(0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99))

    report: Dict[str, Any] = {
        "index_parquet": str(index_path),
        "split": split,
        "dataset": dataset_filter or None,
        "require_assets": bool(args.require_assets),
        "total_rows": total_rows,
        "has_assets": has_assets_stats,
        "bucket_edges_source": edges_source,
        "preset": str(args.preset) if args.preset else None,
        "bucket_edges_m": edges,
        "bucket_labels": labels,
        "bucket_available_counts": bucket_sizes,
        "baseline_distance_m_quantiles": q,
        "recommended": {
            "sample_size": int(args.sample_size),
            "bucket_counts_proportional": prop,
            "bucket_counts_proportional_clamped": prop_clamped,
        },
    }

    # Pretty print.
    print("[INDEX]", index_path)
    print("  split:", split, "rows:", total_rows)
    if dataset_filter:
        print("  dataset:", dataset_filter)
    print("  require_assets:", bool(args.require_assets))
    if has_assets_stats is not None:
        print("  has_assets:", has_assets_stats)
    print("  baseline_distance_m quantiles:", json.dumps(q, indent=None, sort_keys=True))
    print("  buckets:")
    for i, (lab, n) in enumerate(zip(labels, bucket_sizes)):
        frac = float(n) / float(sum(bucket_sizes)) if sum(bucket_sizes) > 0 else 0.0
        print(f"    - {lab}: {n} ({frac:.3%})")
    print("  recommended --bucket_counts (proportional, clamped):", ",".join(str(x) for x in prop_clamped))

    if args.self_check:
        if len(bucket_sizes) != len(edges) - 1:
            raise AssertionError("bucket_sizes length mismatch")
        if len(prop_clamped) != len(bucket_sizes):
            raise AssertionError("recommended counts length mismatch")
        if any(x < 0 for x in prop_clamped):
            raise AssertionError("negative recommended counts")
        if any(prop_clamped[i] > bucket_sizes[i] for i in range(len(bucket_sizes))):
            raise AssertionError("recommended counts exceed availability")

    if args.out_json is not None:
        out = args.out_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print("[OK] wrote:", out)


if __name__ == "__main__":
    main()
