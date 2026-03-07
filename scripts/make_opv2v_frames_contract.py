#!/usr/bin/env python3
"""Make an OPV2V frames contract JSON from the parquet index (reproducible).

Why this exists
---------------
Historically, `scripts/batch_eval.py::discover_frames()` sampled coop pairs by
simply taking the first two agent IDs under each sequence. This is *not* the
same as the training/eval pairing used by `OPV2VCoopDataset(pair_agents=True,
pair_agent_policy='nearest')`, which pairs the main agent with its nearest
neighbor (with assets).

That mismatch can silently make Test50/Test500 contracts much harder (larger
baseline distances) and therefore makes conclusions about "representativeness"
and "bucket coverage" ambiguous.

This script generates explicit, reproducible frame contracts from the canonical
OPV2V index (`opv2v_index/index_<split>.parquet`) with a selectable pairing
policy and optional baseline-distance stratification.

Contract schema (output)
------------------------
{
  "split": "test",
  "seed": 42,
  "sample_size": 500,
  "pair_policy": "index_nearest",
  "bucket_edges_m": [0, 30, 60, 100, 1e9],          # optional
  "bucket_target_counts": [..],                     # optional
  "frames": [
    {
      "sequence": "...",
      "frame": "000123",
      "main_agent": "1045",
      "coop_agents": ["1045", "1054"],
      "baseline_distance_m": 28.6,                  # optional but recommended
      "baseline_bucket": "B1_[30,60)"               # optional
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from coop_bucket_presets import (  # noqa: E402
    bucket_index as _bucket_index,
    bucket_label as _bucket_label,
    parse_edges as _parse_edges,
    preset_edges_m,
)


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _baseline_distance_xy_m(agent_xy: Mapping[str, Sequence[float]], a: str, b: str) -> float | None:
    try:
        pa = agent_xy.get(str(a))
        pb = agent_xy.get(str(b))
        if pa is None or pb is None:
            return None
        pa2 = np.asarray(pa, dtype=np.float64).reshape(-1)
        pb2 = np.asarray(pb, dtype=np.float64).reshape(-1)
        if pa2.size < 2 or pb2.size < 2:
            return None
        return float(np.linalg.norm(pa2[:2] - pb2[:2]))
    except Exception:
        return None


def _frames_hash_md5_v1(frames: Iterable[Mapping[str, Any]]) -> str:
    """Legacy hash: md5 over newline-joined sorted unique `sequence/frame` keys."""
    items = sorted(
        {
            f"{str(fr.get('sequence'))}/{str(fr.get('frame'))}"
            for fr in frames
            if isinstance(fr, Mapping) and fr.get("sequence") is not None and fr.get("frame") is not None
        }
    )
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _frames_hash_md5_v2(frames: Iterable[Mapping[str, Any]]) -> str:
    """Fair hash: include pairing (main + coop_agents order) to prevent silent contract drift."""
    items = sorted(
        {
            f"{str(fr.get('sequence'))}/{str(fr.get('frame'))}"
            f"/main={str(fr.get('main_agent'))}"
            f"/coop={','.join([str(a) for a in (fr.get('coop_agents') or [])])}"
            for fr in frames
            if isinstance(fr, Mapping) and fr.get("sequence") is not None and fr.get("frame") is not None
        }
    )
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Candidate:
    sequence: str
    frame: str
    main: str
    other: str
    baseline_m: float


def _choose_pair(
    *,
    pair_policy: str,
    agents: Sequence[str],
    main_agent: str,
    pair_agent_from_index: str | None,
    agent_xy: Mapping[str, Sequence[float]],
    rng: random.Random,
) -> Tuple[str, str] | None:
    agents_sorted = sorted({str(a) for a in agents if a is not None})
    main = str(main_agent)
    if main not in agents_sorted:
        # Fallback: deterministic first.
        if not agents_sorted:
            return None
        main = agents_sorted[0]

    remaining = [a for a in agents_sorted if a != main]
    if not remaining:
        return None

    policy = str(pair_policy)
    if policy == "index_nearest":
        if pair_agent_from_index is None:
            return None
        other = str(pair_agent_from_index)
        if other == main:
            return None
        if other not in agents_sorted:
            return None
        return main, other
    if policy == "id_first2":
        # Replicate the legacy `discover_frames()` behavior: (min_id, second_min_id).
        agents_sorted2 = agents_sorted
        if len(agents_sorted2) < 2:
            return None
        main2 = agents_sorted2[0]
        other2 = agents_sorted2[1]
        if main2 == other2:
            return None
        return main2, other2
    if policy == "random":
        other = rng.choice(remaining)
        return main, other
    if policy == "farthest":
        best_other = None
        best_dist = -1.0
        for other in remaining:
            d = _baseline_distance_xy_m(agent_xy, main, other)
            if d is None:
                continue
            if d > best_dist:
                best_dist = float(d)
                best_other = other
        if best_other is None:
            return None
        return main, best_other

    raise ValueError(f"Unsupported --pair_policy: {pair_policy!r}")


def _allocate_bucket_counts(
    *,
    sample_size: int,
    bucket_sizes: Sequence[int],
) -> List[int]:
    """Allocate per-bucket sample counts proportional to availability (deterministic)."""
    total = int(sum(bucket_sizes))
    if total <= 0:
        return [0 for _ in bucket_sizes]
    # Initial floor allocation.
    raw = [float(sample_size) * (float(n) / float(total)) for n in bucket_sizes]
    counts = [int(math.floor(x)) for x in raw]
    # Distribute remainder by largest fractional parts.
    remainder = int(sample_size - sum(counts))
    frac = [raw[i] - float(counts[i]) for i in range(len(raw))]
    order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
    for i in order:
        if remainder <= 0:
            break
        counts[i] += 1
        remainder -= 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=("train", "validate", "test"))
    ap.add_argument("--index_parquet", type=Path, default=None)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--sample_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pair_policy",
        default="index_nearest",
        choices=("index_nearest", "id_first2", "random", "farthest"),
        help="How to choose the coop pair for each frame.",
    )
    ap.add_argument(
        "--require_assets",
        action="store_true",
        help="Require that both selected agents have all camera assets (recommended).",
    )
    ap.add_argument(
        "--bucket_preset",
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
        help="Optional comma-separated baseline-distance edges (meters) for stratified sampling.",
    )
    ap.add_argument(
        "--bucket_counts",
        type=str,
        default="",
        help="Optional comma-separated per-bucket target counts (overrides proportional allocation).",
    )
    args = ap.parse_args()

    split = str(args.split)
    rng = random.Random(int(args.seed))

    index_path = args.index_parquet
    if index_path is None:
        index_path = REPO_ROOT / "data" / "opv2v" / "opv2v_index" / f"index_{split}.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(f"index parquet not found: {index_path}")

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required to read OPV2V index parquet") from exc

    cols = ["sequence", "frame", "agents", "main_agent", "pair_agent", "agent_xy", "agent_has_assets"]
    # Read only available columns to keep memory overhead low.
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(index_path)
        available = set(schema.names)
        read_cols = [c for c in cols if c in available]
    except Exception:
        read_cols = list(cols)
    try:
        df = pd.read_parquet(index_path, columns=read_cols)
    except Exception:
        # Fallback for unexpected parquet reader issues.
        df = pd.read_parquet(index_path)

    # Ensure missing columns are filled.
    for c in cols:
        if c not in df.columns:
            df[c] = None

    candidates: List[Candidate] = []
    skipped = 0
    for row in df.itertuples(index=False):
        seq = getattr(row, "sequence", None)
        frame = getattr(row, "frame", None)
        agents_raw = getattr(row, "agents", None)
        agents = (
            [str(a) for a in list(agents_raw) if a is not None]
            if isinstance(agents_raw, (list, tuple, np.ndarray))
            else []
        )
        main_agent = getattr(row, "main_agent", None)
        pair_agent = getattr(row, "pair_agent", None)
        agent_xy_raw = getattr(row, "agent_xy", None)
        agent_xy = agent_xy_raw if isinstance(agent_xy_raw, dict) else {}
        agent_has_assets_raw = getattr(row, "agent_has_assets", None)
        agent_has_assets = agent_has_assets_raw if isinstance(agent_has_assets_raw, dict) else {}

        if seq is None or frame is None or main_agent is None:
            skipped += 1
            continue
        if len(agents) < 2:
            skipped += 1
            continue

        picked = _choose_pair(
            pair_policy=str(args.pair_policy),
            agents=agents,
            main_agent=str(main_agent),
            pair_agent_from_index=str(pair_agent) if pair_agent is not None else None,
            agent_xy=agent_xy,
            rng=rng,
        )
        if picked is None:
            skipped += 1
            continue
        main, other = picked

        if args.require_assets:
            if not (agent_has_assets.get(main, False) and agent_has_assets.get(other, False)):
                skipped += 1
                continue

        if not isinstance(agent_xy, dict):
            skipped += 1
            continue
        d = _baseline_distance_xy_m(agent_xy, main, other)
        if d is None or not math.isfinite(float(d)):
            skipped += 1
            continue

        candidates.append(Candidate(sequence=str(seq), frame=str(frame), main=main, other=other, baseline_m=float(d)))

    if not candidates:
        raise RuntimeError(f"No usable candidates from index: {index_path}")

    edges: List[float] | None = None
    if args.bucket_preset and str(args.bucket_edges).strip():
        raise ValueError("Cannot use --bucket_preset together with --bucket_edges")
    if args.bucket_preset:
        edges = preset_edges_m(str(args.bucket_preset))
    elif str(args.bucket_edges).strip():
        edges = _parse_edges(str(args.bucket_edges))

    selected: List[Candidate] = []
    bucket_meta = {}
    if edges is None:
        rng.shuffle(candidates)
        selected = candidates[: int(min(len(candidates), int(args.sample_size)))]
    else:
        buckets: List[List[Candidate]] = [[] for _ in range(len(edges) - 1)]
        for c in candidates:
            bi = _bucket_index(edges, c.baseline_m)
            if bi is None:
                continue
            buckets[bi].append(c)

        bucket_sizes = [len(b) for b in buckets]
        req = _parse_int_list(str(args.bucket_counts)) if args.bucket_counts.strip() else []
        if req:
            if len(req) != len(buckets):
                raise ValueError(
                    f"--bucket_counts length mismatch: got {len(req)} counts but edges define {len(buckets)} buckets"
                )
            target = [int(x) for x in req]
        else:
            target = _allocate_bucket_counts(sample_size=int(args.sample_size), bucket_sizes=bucket_sizes)

        # Clamp and redistribute if a bucket has insufficient candidates.
        target2 = list(target)
        leftover = 0
        for i in range(len(target2)):
            if target2[i] > bucket_sizes[i]:
                leftover += target2[i] - bucket_sizes[i]
                target2[i] = bucket_sizes[i]
        if leftover > 0:
            # Redistribute leftover to buckets with remaining capacity, preferring larger buckets.
            cap = [bucket_sizes[i] - target2[i] for i in range(len(target2))]
            order = sorted(range(len(cap)), key=lambda i: cap[i], reverse=True)
            for i in order:
                if leftover <= 0:
                    break
                take = min(cap[i], leftover)
                if take > 0:
                    target2[i] += int(take)
                    leftover -= int(take)

        for i, b in enumerate(buckets):
            rng.shuffle(b)
            selected.extend(b[: int(target2[i])])

        bucket_meta = {
            "bucket_edges_m": edges,
            "bucket_available_counts": bucket_sizes,
            "bucket_target_counts": target2,
            "bucket_labels": [_bucket_label(edges, i) for i in range(len(edges) - 1)],
        }

    # Stable ordering in JSON.
    selected.sort(key=lambda x: (x.sequence, x.frame, x.main, x.other))

    frames_out: List[Dict[str, Any]] = []
    for c in selected:
        fr: Dict[str, Any] = {
            "sequence": c.sequence,
            "frame": c.frame,
            "main_agent": c.main,
            "coop_agents": [c.main, c.other],
            "baseline_distance_m": float(c.baseline_m),
        }
        if edges is not None:
            bi = _bucket_index(edges, c.baseline_m)
            if bi is not None:
                fr["baseline_bucket"] = _bucket_label(edges, int(bi))
        frames_out.append(fr)

    payload: Dict[str, Any] = {
        "split": split,
        "seed": int(args.seed),
        "sample_size": int(len(frames_out)),
        "pair_policy": str(args.pair_policy),
        "index_parquet": str(index_path),
        "require_assets": bool(args.require_assets),
        "frames_hash_md5_v1": _frames_hash_md5_v1(frames_out),
        "frames_hash_md5_v2": _frames_hash_md5_v2(frames_out),
        "skipped_candidates": int(skipped),
        "total_candidates": int(len(candidates)),
        "frames": frames_out,
    }
    payload.update(bucket_meta)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print('[OK] wrote contract:', args.out_json)
    print('  split:', split, 'pair_policy:', str(args.pair_policy))
    print('  frames:', len(frames_out), '/', int(args.sample_size), 'skipped_candidates:', skipped)
    if edges is not None:
        print('  bucket_target_counts:', payload.get('bucket_target_counts'))


if __name__ == "__main__":
    main()
