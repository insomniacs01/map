#!/usr/bin/env python3
"""Shared baseline-distance bucket presets/utilities for coop perception scripts.

This module is intentionally lightweight and dependency-free so it can be
imported by small CLI tools under `scripts/` without pulling in the full
training stack.

Presets
-------
- fine_eval: 5 buckets (near is split) for evaluation / reporting
- coarse_train: 4 buckets for training / curriculum (more samples per bucket)
"""

from __future__ import annotations

import math
from typing import List, Sequence


INF_SENTINEL_M = 1.0e9

# Standard 5-bucket preset used for evaluation and bucket reports.
FINE_EVAL_BUCKET_EDGES_M: List[float] = [0.0, 15.0, 30.0, 60.0, 100.0, INF_SENTINEL_M]
FINE_EVAL_BUCKET_EDGES_STR = "0,15,30,60,100,inf"

# Standard 4-bucket preset used for training/curriculum.
COARSE_TRAIN_BUCKET_EDGES_M: List[float] = [0.0, 30.0, 60.0, 100.0, INF_SENTINEL_M]
COARSE_TRAIN_BUCKET_EDGES_STR = "0,30,60,100,inf"


def preset_edges_m(preset: str) -> List[float]:
    p = (preset or "").strip()
    if p in ("fine_eval", "default"):
        return list(FINE_EVAL_BUCKET_EDGES_M)
    if p == "coarse_train":
        return list(COARSE_TRAIN_BUCKET_EDGES_M)
    raise ValueError(f"unknown bucket preset: {preset!r} (expected: fine_eval|coarse_train)")


def parse_edges(s: str) -> List[float]:
    """Parse comma-separated bucket edges in meters.

    Supports "inf" tokens; they are mapped to `INF_SENTINEL_M`.
    """

    out: List[float] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        t = tok.lower()
        if t in ("inf", "+inf", "infty", "infinity"):
            out.append(float(INF_SENTINEL_M))
        else:
            out.append(float(tok))
    if len(out) < 2:
        raise ValueError("bucket_edges must contain at least 2 comma-separated numbers")
    if any(not math.isfinite(x) for x in out):
        raise ValueError("bucket_edges contains non-finite values")
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError("bucket_edges must be strictly increasing")
    return out


def bucket_label(edges: Sequence[float], i: int) -> str:
    lo = float(edges[i])
    hi = float(edges[i + 1])
    if hi >= 1e8:
        return f"B{i}_[{lo:g},inf)"
    return f"B{i}_[{lo:g},{hi:g})"


def bucket_index(edges: Sequence[float], x: float) -> int | None:
    """Return bucket index for x under half-open intervals [edge_i, edge_{i+1}).

    Boundary policy:
    - If x equals the last edge, assign it to the last bucket.
    - If the last edge is an "inf sentinel" (>=1e8) and x is larger, also assign to the last bucket.
    """

    xf = float(x)
    for i in range(len(edges) - 1):
        if float(edges[i]) <= xf < float(edges[i + 1]):
            return i
    if len(edges) >= 2:
        if xf == float(edges[-1]):
            return len(edges) - 2
        if xf > float(edges[-1]) and float(edges[-1]) >= 1e8:
            return len(edges) - 2
    return None

