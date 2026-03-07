#!/usr/bin/env python3
"""Make a generic frames contract JSON from a parquet index (dataset-agnostic).

This script is intended to be the "last mile" contract generator once you have
an index parquet that already encodes:
  - the exact main/cooperative agents to use for each frame, and
  - the precomputed baseline distance used for bucketing.

Required index columns
----------------------
- split: str
- sequence: str
- frame: str (or int; will be stringified)
- main_agent: str (or int; will be stringified)
- coop_agents: list[str] (or list[int]; will be stringified)
- baseline_distance_m: float

Optional index columns
----------------------
- dataset: str
  If present, it will be copied into each frame entry in the output contract.
  This is useful when you merge multiple datasets into a single index and want
  to avoid `(sequence, frame)` key collisions.
  - When working with a merged multi-dataset index, you can also filter by a
    specific dataset via `--dataset ...`.
- has_assets: bool
  If `--require_assets` is set, rows with falsy `has_assets` are skipped to
  avoid generating contracts that will fail at load time.

Output contract schema
----------------------
The output matches the common "frames contract" schema used across this repo:
{
  "split": "test",
  "seed": 42,
  "sample_size": 500,
  "require_assets": false,
  "bucket_edges_m": [0, 15, 30, 60, 100, 1000000000],     # optional
  "bucket_target_counts": [..],                           # optional
  "frames_hash_md5_v1": "...",
  "frames_hash_md5_v2": "...",
  "frames_hash_md5_v3": "...",                           # dataset-aware (when dataset is present)
  "frames": [
    {
      "dataset": "opv2v",                                 # optional
      "sequence": "...",
      "frame": "000123",
      "main_agent": "1045",
      "coop_agents": ["1045", "1054"],
      "baseline_distance_m": 28.6,
      "baseline_bucket": "B1_[15,30)"                    # optional
    }
  ]
}

Notes / self-check
------------------
- You can run a quick schema+hash validation with `--self_check`.
- For a pure syntax check:
    python -m py_compile scripts/make_frames_contract_from_index.py
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
    FINE_EVAL_BUCKET_EDGES_STR as DEFAULT_STRESS_BUCKET_EDGES_STR,
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


def _frames_hash_md5_v3(frames: Iterable[Mapping[str, Any]]) -> str:
    """Dataset-aware hash: include optional per-frame `dataset` + pairing to prevent cross-dataset collisions.

    Backward-compatibility policy:
      - If no frame has a non-empty `dataset`, v3 == v2.
      - Otherwise, include `dataset` (stringified, can be "None") in the hash key.
    """

    frames_list = list(frames)
    has_dataset = any(
        isinstance(fr, Mapping) and fr.get("dataset") not in (None, "", [])
        for fr in frames_list
    )
    if not has_dataset:
        return _frames_hash_md5_v2(frames_list)

    items = sorted(
        {
            f"ds={str(fr.get('dataset'))}/"
            f"{str(fr.get('sequence'))}/{str(fr.get('frame'))}"
            f"/main={str(fr.get('main_agent'))}"
            f"/coop={','.join([str(a) for a in (fr.get('coop_agents') or [])])}"
            for fr in frames_list
            if isinstance(fr, Mapping) and fr.get("sequence") is not None and fr.get("frame") is not None
        }
    )
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


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


def _coerce_str_list(x: Any) -> List[str]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(a) for a in list(x) if a is not None and str(a) != ""]
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Common fallback when list columns get stringified.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed if a is not None and str(a) != ""]
            except Exception:
                pass
        if "," in s:
            return [tok.strip() for tok in s.split(",") if tok.strip()]
        return [s]
    return []


def _normalize_coop_agents(*, main_agent: str, coop_agents: Sequence[str]) -> Tuple[str, ...]:
    main = str(main_agent)
    xs = [str(a) for a in coop_agents if a is not None and str(a) != ""]
    # De-dup while preserving order.
    seen = set()
    xs2: List[str] = []
    for a in xs:
        if a in seen:
            continue
        seen.add(a)
        xs2.append(a)
    if main in xs2:
        xs2.remove(main)
    xs2.insert(0, main)
    return tuple(xs2)


@dataclass(frozen=True)
class Candidate:
    dataset: str | None
    sequence: str
    frame: str
    main_agent: str
    coop_agents: Tuple[str, ...]
    baseline_m: float


def _read_parquet_columns(index_path: Path, wanted: Sequence[str]) -> "Any":
    """Read parquet with column projection when possible (pandas DataFrame)."""
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required to read parquet index files") from exc

    # Best-effort column projection (works with pyarrow backend).
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(index_path)
        available = set(schema.names)
        cols = [c for c in wanted if c in available]
        df = pd.read_parquet(index_path, columns=cols)
    except Exception:
        df = pd.read_parquet(index_path)
    return df


def _load_candidates(
    *,
    index_path: Path,
    split: str,
    dataset: str | None,
    require_assets: bool,
) -> tuple[list[Candidate], int, int, bool]:
    required = ["split", "sequence", "frame", "main_agent", "coop_agents", "baseline_distance_m"]
    optional = ["has_assets", "dataset"]
    df = _read_parquet_columns(index_path, wanted=required + optional)
    for c in required:
        if c not in df.columns:
            raise ValueError(f"index parquet missing required column: {c!r} (path={index_path})")
    if require_assets and "has_assets" not in df.columns:
        raise ValueError(f"--require_assets set but index parquet missing column 'has_assets': {index_path}")

    split_s = str(split)
    df = df[df["split"].astype(str) == split_s]

    dataset_filter = str(dataset).strip() if dataset is not None else ""
    if dataset_filter:
        if "dataset" not in df.columns:
            raise ValueError(f"--dataset set but index parquet missing column 'dataset': {index_path}")
        df = df[df["dataset"].astype(str) == dataset_filter]

    skipped = 0
    kept = 0
    out: list[Candidate] = []
    for row in df.itertuples(index=False):
        ds_raw = getattr(row, "dataset", None) if "dataset" in df.columns else None
        seq = getattr(row, "sequence", None)
        frame = getattr(row, "frame", None)
        main = getattr(row, "main_agent", None)
        coop_raw = getattr(row, "coop_agents", None)
        dist_raw = getattr(row, "baseline_distance_m", None)
        has_assets = getattr(row, "has_assets", None) if "has_assets" in df.columns else None

        if seq is None or frame is None or main is None or dist_raw is None:
            skipped += 1
            continue

        if require_assets:
            if not bool(has_assets):
                skipped += 1
                continue

        try:
            dist = float(dist_raw)
        except Exception:
            skipped += 1
            continue
        if not math.isfinite(dist):
            skipped += 1
            continue
        if dist < 0.0:
            skipped += 1
            continue

        coop_list = _coerce_str_list(coop_raw)
        coop = _normalize_coop_agents(main_agent=str(main), coop_agents=coop_list)
        if not coop:
            skipped += 1
            continue

        ds = None
        if ds_raw is not None:
            try:
                # pandas may store missing strings as NaN; avoid stringifying to "nan".
                if isinstance(ds_raw, float) and math.isnan(float(ds_raw)):
                    ds = None
                else:
                    s = str(ds_raw).strip()
                    if s and s.lower() != "nan":
                        ds = s
            except Exception:
                ds = None

        out.append(
            Candidate(
                dataset=ds,
                sequence=str(seq),
                frame=str(frame),
                main_agent=str(main),
                coop_agents=coop,
                baseline_m=dist,
            )
        )
        kept += 1

    has_dataset_col = "dataset" in df.columns
    return out, kept, skipped, bool(has_dataset_col)


def _candidate_sort_key(c: Candidate) -> tuple[Any, ...]:
    # Deterministic ordering to avoid sampling being sensitive to parquet row order.
    return (
        str(c.dataset or ""),
        str(c.sequence),
        str(c.frame),
        str(c.main_agent),
        tuple(str(a) for a in (c.coop_agents or ())),
        float(c.baseline_m),
    )


def _frame_key_from_candidate(c: Candidate, *, mode: str) -> Tuple[str, ...]:
    seq = str(c.sequence)
    frame = str(c.frame)
    if mode == "v1":
        return (seq, frame)
    main = str(c.main_agent)
    coop = tuple(str(a) for a in (c.coop_agents or ()))
    if mode == "v2":
        return (seq, frame, main, ",".join(coop))
    ds = str(c.dataset or "")
    return (ds, seq, frame, main, ",".join(coop))


def _validate_frame_fields_for_key_mode(*, fr: Mapping[str, Any], mode: str, context: str) -> None:
    required = ["sequence", "frame"]
    if mode in ("v2", "v3"):
        required.extend(["main_agent", "coop_agents"])
    missing = [k for k in required if fr.get(k, None) is None]
    if missing:
        raise ValueError(f"{context} missing required fields for exclude_key_mode={mode}: {missing}")


def _frame_key_from_mapping(fr: Mapping[str, Any], *, mode: str) -> Tuple[str, ...]:
    seq = str(fr.get("sequence"))
    frame = str(fr.get("frame"))
    if mode == "v1":
        return (seq, frame)

    main = str(fr.get("main_agent"))
    coop = tuple(_coerce_str_list(fr.get("coop_agents")))
    if mode == "v2":
        return (seq, frame, main, ",".join(coop))

    ds = str(fr.get("dataset") or "")
    return (ds, seq, frame, main, ",".join(coop))


def _choose_key_mode_auto(
    *, requested: str, candidates: Sequence[Candidate], exclude_frames: Sequence[Mapping[str, Any]]
) -> tuple[str, str | None]:
    """Auto key mode with dataset-asymmetry guard.

    Use v3 only when both candidate pool and exclude contracts contain non-empty dataset values.
    Otherwise fall back to v2. If the dataset field is asymmetric, return a warning string for meta.
    """

    if requested != "auto":
        return requested, None

    cand_has_dataset = any(c.dataset is not None and str(c.dataset).strip() for c in candidates)
    excl_has_dataset = any(bool(str(fr.get("dataset") or "").strip()) for fr in exclude_frames)

    if cand_has_dataset and excl_has_dataset:
        return "v3", None

    if cand_has_dataset != excl_has_dataset:
        warn = (
            "dataset field asymmetric; auto fallback to v2 "
            f"(candidates_has_dataset={str(bool(cand_has_dataset)).lower()}, "
            f"exclude_has_dataset={str(bool(excl_has_dataset)).lower()})"
        )
        return "v2", warn

    # Both sides have no dataset (symmetric); v2 is the intended default.
    return "v2", None


def _filter_candidates_by_exclude_contracts(
    *,
    candidates: Sequence[Candidate],
    split: str,
    exclude_contracts: Sequence[Path],
    requested_mode: str,
) -> tuple[list[Candidate], dict[str, Any]]:
    if not exclude_contracts:
        return list(candidates), {}

    # Load frames from all exclude contracts, but ignore contracts with mismatched split when present.
    exclude_frames: list[Mapping[str, Any]] = []
    used_paths: list[str] = []
    skipped_paths: list[str] = []
    total_exclude_frames = 0
    for p0 in exclude_contracts:
        p = p0.expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--exclude_contract not found: {p}")
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc_split = doc.get("split", None)
        if doc_split is not None and str(doc_split) and str(doc_split) != str(split):
            skipped_paths.append(str(p))
            continue
        frames = [fr for fr in doc.get("frames", []) if isinstance(fr, Mapping)]
        total_exclude_frames += int(len(frames))
        exclude_frames.extend(frames)
        used_paths.append(str(p))

    mode, auto_warn = _choose_key_mode_auto(
        requested=str(requested_mode), candidates=candidates, exclude_frames=exclude_frames
    )

    exclude_keys: set[Tuple[str, ...]] = set()
    for i, fr in enumerate(exclude_frames):
        _validate_frame_fields_for_key_mode(fr=fr, mode=mode, context=f"exclude_contract frame[{i}]")
        exclude_keys.add(_frame_key_from_mapping(fr, mode=mode))

    before = int(len(candidates))
    kept: list[Candidate] = []
    for c in candidates:
        if _frame_key_from_candidate(c, mode=mode) in exclude_keys:
            continue
        kept.append(c)
    excluded = int(before - len(kept))

    meta = {
        "exclude_contracts": used_paths,
        "exclude_contracts_skipped_due_to_split": skipped_paths,
        "exclude_key_mode": mode,
        "exclude_contract_frames_total": int(total_exclude_frames),
        "excluded_candidates": int(excluded),
        "remaining_candidates": int(len(kept)),
    }
    if auto_warn:
        meta["exclude_key_mode_auto_warning"] = str(auto_warn)
    return kept, meta


def _validate_written_contract(payload: Mapping[str, Any]) -> None:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("contract has no frames")

    # Ensure no exact-duplicate entries (common gotcha when merging indices).
    uniq: set[tuple[Any, str, str, str, tuple[str, ...]]] = set()
    for fr in frames:
        if not isinstance(fr, dict):
            raise ValueError(f"bad frame entry (expected dict): {type(fr)}")
        for k in ("sequence", "frame", "main_agent", "coop_agents"):
            if k not in fr:
                raise ValueError(f"frame missing key {k!r}: {fr}")
        if not isinstance(fr.get("coop_agents"), list) or not fr["coop_agents"]:
            raise ValueError(f"bad coop_agents: {fr.get('coop_agents')!r}")
        if fr["main_agent"] not in fr["coop_agents"]:
            raise ValueError(f"main_agent must be included in coop_agents: {fr}")

        ds = fr.get("dataset") if isinstance(fr, Mapping) and "dataset" in fr else None
        key = (
            ds,
            str(fr.get("sequence")),
            str(fr.get("frame")),
            str(fr.get("main_agent")),
            tuple(str(a) for a in (fr.get("coop_agents") or [])),
        )
        if key in uniq:
            raise ValueError(f"duplicate frame entry (dataset/sequence/frame/main/coop): {key}")
        uniq.add(key)

    h1 = _frames_hash_md5_v1(frames)
    h2 = _frames_hash_md5_v2(frames)
    h3 = _frames_hash_md5_v3(frames)
    if payload.get("frames_hash_md5_v1") != h1:
        raise ValueError(f"frames_hash_md5_v1 mismatch: payload={payload.get('frames_hash_md5_v1')} recompute={h1}")
    if payload.get("frames_hash_md5_v2") != h2:
        raise ValueError(f"frames_hash_md5_v2 mismatch: payload={payload.get('frames_hash_md5_v2')} recompute={h2}")
    if payload.get("frames_hash_md5_v3") != h3:
        raise ValueError(f"frames_hash_md5_v3 mismatch: payload={payload.get('frames_hash_md5_v3')} recompute={h3}")
    if int(payload.get("sample_size", -1)) != len(frames):
        raise ValueError(f"sample_size mismatch: payload={payload.get('sample_size')} len(frames)={len(frames)}")


def _balanced_bucket_counts(*, sample_size: int, num_buckets: int) -> List[int]:
    """Balanced target counts used for the default C-stress contract.

    We allocate the remainder to *far* buckets (from the end) to slightly
    overweight tail difficulty when sample_size is not divisible by #buckets.
    """

    if num_buckets <= 0:
        return []
    base = int(sample_size // num_buckets)
    rem = int(sample_size - base * num_buckets)
    out = [base for _ in range(num_buckets)]
    for j in range(rem):
        out[num_buckets - 1 - j] += 1
    return out


def _select_candidates(
    *,
    candidates: Sequence[Candidate],
    rng: random.Random,
    sample_size: int,
    edges: List[float] | None,
    bucket_target_counts: Sequence[int] | None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Select a subset of candidates (optionally stratified by baseline-distance buckets)."""

    # Stable sort to make sampling deterministic w.r.t. parquet row order.
    candidates2 = sorted(list(candidates), key=_candidate_sort_key)

    if edges is None:
        pool = list(candidates2)
        rng.shuffle(pool)
        selected = pool[: int(min(len(pool), int(sample_size)))]
        return selected, {}

    buckets: list[list[Candidate]] = [[] for _ in range(len(edges) - 1)]
    for c in candidates2:
        bi = _bucket_index(edges, c.baseline_m)
        if bi is None:
            continue
        buckets[int(bi)].append(c)

    bucket_sizes = [len(b) for b in buckets]
    if bucket_target_counts is not None:
        target = [int(x) for x in list(bucket_target_counts)]
        if len(target) != len(buckets):
            raise ValueError(
                f"bucket_target_counts length mismatch: got {len(target)} counts but edges define {len(buckets)} buckets"
            )
    else:
        target = _allocate_bucket_counts(sample_size=int(sample_size), bucket_sizes=bucket_sizes)

    # Clamp and redistribute if a bucket has insufficient candidates.
    target2 = list(target)
    leftover = 0
    for i in range(len(target2)):
        if target2[i] > bucket_sizes[i]:
            leftover += target2[i] - bucket_sizes[i]
            target2[i] = bucket_sizes[i]
    if leftover > 0:
        cap = [bucket_sizes[i] - target2[i] for i in range(len(target2))]
        order = sorted(range(len(cap)), key=lambda i: cap[i], reverse=True)
        for i in order:
            if leftover <= 0:
                break
            take = min(cap[i], leftover)
            if take > 0:
                target2[i] += int(take)
                leftover -= int(take)

    selected: list[Candidate] = []
    for i, b in enumerate(buckets):
        rng.shuffle(b)
        selected.extend(b[: int(target2[i])])

    bucket_meta = {
        "bucket_edges_m": edges,
        "bucket_available_counts": bucket_sizes,
        "bucket_target_counts": target2,
        "bucket_labels": [_bucket_label(edges, i) for i in range(len(edges) - 1)],
    }
    return selected, bucket_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index_parquet", type=Path, required=True, help="Parquet index with the required schema.")
    out_g = ap.add_mutually_exclusive_group(required=True)
    out_g.add_argument("--out_json", type=Path, default=None, help="Output contract JSON path (single contract).")
    out_g.add_argument("--out_dir", type=Path, default=None, help="Output directory (writes a standard suite).")
    ap.add_argument("--split", type=str, default="test", help="Split name to filter by (matches the index 'split').")
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset filter (matches index column 'dataset' if present).",
    )
    ap.add_argument(
        "--require_dataset",
        action="store_true",
        help=(
            "When the index has a 'dataset' column and --dataset is not set, "
            "require every selected frame to have a non-empty dataset (else error)."
        ),
    )
    ap.add_argument(
        "--exclude_contract",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exclude candidate frames that already appear in the given contract(s). "
            "Repeatable. Useful to generate disjoint quick/gate/final."
        ),
    )
    ap.add_argument(
        "--exclude_key_mode",
        type=str,
        default="auto",
        choices=("auto", "v1", "v2", "v3"),
        help=(
            "Key mode for --exclude_contract filtering. "
            "v1: sequence/frame; v2: sequence/frame/main_agent/coop_agents; v3: dataset + v2; "
            "auto: v3 when dataset is present, else v2."
        ),
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="",
        help=(
            "Optional comma-separated splits to generate. If set, overrides --split. "
            "In suite mode (--out_dir) this can generate multiple splits in one run."
        ),
    )
    ap.add_argument("--sample_size", type=int, default=500, help="Total frames to sample (ignored if index has fewer).")
    ap.add_argument(
        "--stress_sample_size",
        type=int,
        default=None,
        help="Optional stress sample size used in suite mode (default: same as --sample_size).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--require_assets",
        action="store_true",
        help="If index has `has_assets`, require it to be true for selected rows.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag added to suite filenames (e.g. 'nearest' to match OPV2V naming).",
    )
    ap.add_argument(
        "--suite",
        type=str,
        default="main,stress,full",
        help="Comma-separated suite kinds to write when --out_dir is used (default: main,stress,full).",
    )
    ap.add_argument(
        "--stress_policy",
        type=str,
        default="balanced",
        choices=("balanced", "proportional"),
        help="How to choose stress bucket target counts when --bucket_counts is not provided.",
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
        help=(
            "Optional comma-separated baseline-distance edges (meters) for stratified sampling. "
            f"In suite stress mode, if omitted and --bucket_preset is unset, defaults to {DEFAULT_STRESS_BUCKET_EDGES_STR}."
        ),
    )
    ap.add_argument(
        "--bucket_counts",
        type=str,
        default="",
        help="Optional comma-separated per-bucket target counts (stress contract).",
    )
    ap.add_argument("--self_check", action="store_true", help="Validate the written contract schema and hashes.")
    args = ap.parse_args()

    index_path = args.index_parquet.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"index parquet not found: {index_path}")

    seed = int(args.seed)
    dataset_filter = str(args.dataset).strip()
    exclude_contracts = list(args.exclude_contract or [])
    exclude_key_mode_requested = str(args.exclude_key_mode)

    splits_raw = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    splits = splits_raw if splits_raw else [str(args.split)]
    # De-dup while preserving order for nicer UX.
    seen_splits: set[str] = set()
    splits = [s for s in splits if not (s in seen_splits or seen_splits.add(s))]
    if not splits:
        raise ValueError("No splits selected (use --split or --splits)")

    if args.bucket_preset and str(args.bucket_edges).strip():
        raise ValueError("Cannot use --bucket_preset together with --bucket_edges")

    if args.out_dir is None:
        # Single-contract mode (backward compatible).
        if len(splits) != 1:
            raise ValueError("--out_json mode supports exactly one split (use --split or a single-entry --splits)")
        split = str(splits[0])

        candidates, kept, skipped, has_dataset_col = _load_candidates(
            index_path=index_path,
            split=split,
            dataset=dataset_filter or None,
            require_assets=bool(args.require_assets),
        )
        exclude_meta: dict[str, Any] = {}
        if exclude_contracts:
            candidates, exclude_meta = _filter_candidates_by_exclude_contracts(
                candidates=candidates,
                split=split,
                exclude_contracts=exclude_contracts,
                requested_mode=exclude_key_mode_requested,
            )
        if not candidates:
            raise RuntimeError(f"No usable candidates from index: {index_path} (split={split})")

        def write_contract(
            *,
            kind: str,
            out_path: Path,
            selected: Sequence[Candidate],
            edges: List[float] | None,
            bucket_meta: Mapping[str, Any],
            extra_meta: Mapping[str, Any],
        ) -> Dict[str, Any]:
            # Stable ordering in JSON.
            selected2 = list(selected)
            selected2.sort(key=lambda x: ((x.dataset or ""), x.sequence, x.frame, x.main_agent, x.coop_agents))

            frames_out: List[Dict[str, Any]] = []
            datasets_used: set[str] = set()
            for c in selected2:
                fr: Dict[str, Any] = {
                    "sequence": c.sequence,
                    "frame": c.frame,
                    "main_agent": c.main_agent,
                    "coop_agents": list(c.coop_agents),
                    "baseline_distance_m": float(c.baseline_m),
                }
                if c.dataset is not None:
                    fr["dataset"] = c.dataset
                    datasets_used.add(str(c.dataset))
                if edges is not None:
                    bi = _bucket_index(edges, c.baseline_m)
                    if bi is not None:
                        fr["baseline_bucket"] = _bucket_label(edges, int(bi))
                frames_out.append(fr)

            payload: Dict[str, Any] = {
                "split": split,
                "seed": seed,
                "sample_size": int(len(frames_out)),
                "index_parquet": str(index_path),
                "require_assets": bool(args.require_assets),
                "frames_hash_md5_v1": _frames_hash_md5_v1(frames_out),
                "frames_hash_md5_v2": _frames_hash_md5_v2(frames_out),
                "frames_hash_md5_v3": _frames_hash_md5_v3(frames_out),
                "total_candidates": int(kept),
                "skipped_candidates": int(skipped),
                "contract_kind": str(kind),
                "frames": frames_out,
            }
            if datasets_used:
                payload["datasets"] = sorted(datasets_used)
            payload.update(dict(bucket_meta))
            payload.update(dict(extra_meta))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            if args.self_check:
                _validate_written_contract(payload)
            return payload

        if args.bucket_preset:
            edges = preset_edges_m(str(args.bucket_preset))
        elif str(args.bucket_edges).strip():
            edges = _parse_edges(str(args.bucket_edges))
        else:
            edges = None
        bucket_target = _parse_int_list(str(args.bucket_counts)) if args.bucket_counts.strip() else None
        rng = random.Random(seed)
        selected, bucket_meta = _select_candidates(
            candidates=candidates,
            rng=rng,
            sample_size=int(args.sample_size),
            edges=edges,
            bucket_target_counts=bucket_target,
        )
        if bool(args.require_dataset) and bool(has_dataset_col) and not dataset_filter:
            missing = [c for c in selected if c.dataset is None or not str(c.dataset).strip()]
            if missing:
                preview = [
                    f"{c.sequence}/{c.frame}/main={c.main_agent}/coop={','.join(c.coop_agents)}"
                    for c in missing[:5]
                ]
                raise ValueError(
                    "Selected frames missing dataset but --require_dataset is set "
                    f"(split={split}, missing={len(missing)}/{len(selected)}). "
                    f"Examples: {preview}"
                )

        out_json = args.out_json
        assert out_json is not None
        out_json = out_json.expanduser()
        if not out_json.is_absolute():
            out_json = (REPO_ROOT / out_json).resolve()
        else:
            out_json = out_json.resolve()

        payload = write_contract(
            kind="single",
            out_path=out_json,
            selected=selected,
            edges=edges,
            bucket_meta=bucket_meta,
            extra_meta={
                **({"bucket_preset": str(args.bucket_preset)} if args.bucket_preset else {}),
                "require_dataset": bool(args.require_dataset),
                **exclude_meta,
            },
        )

        print("[OK] wrote contract:", out_json)
        print("  split:", split)
        print("  frames:", int(payload["sample_size"]), "/", int(args.sample_size))
        if edges is not None:
            print("  bucket_target_counts:", payload.get("bucket_target_counts"))
        return

    # Suite mode: write a standard (main, stress, full) contract set.
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_kinds = [s.strip() for s in str(args.suite).split(",") if s.strip()]
    if not suite_kinds:
        raise ValueError("--suite is empty")
    allowed = {"main", "stress", "full"}
    bad = [s for s in suite_kinds if s not in allowed]
    if bad:
        raise ValueError(f"--suite contains unsupported kinds: {bad} (allowed={sorted(allowed)})")

    tag = str(args.tag).strip()
    tag_mid = f"_{tag}" if tag else ""
    dataset_tag = dataset_filter.strip().replace("/", "_").replace("\\", "_")
    dataset_mid = f"_{dataset_tag}" if dataset_tag else ""
    multi_split = len(splits) > 1
    for split in splits:
        candidates, kept, skipped, has_dataset_col = _load_candidates(
            index_path=index_path,
            split=str(split),
            dataset=dataset_filter or None,
            require_assets=bool(args.require_assets),
        )
        exclude_meta: dict[str, Any] = {}
        if exclude_contracts:
            candidates, exclude_meta = _filter_candidates_by_exclude_contracts(
                candidates=candidates,
                split=str(split),
                exclude_contracts=exclude_contracts,
                requested_mode=exclude_key_mode_requested,
            )
        if not candidates:
            raise RuntimeError(f"No usable candidates from index: {index_path} (split={split})")

        def write_contract(
            *,
            kind: str,
            out_path: Path,
            selected: Sequence[Candidate],
            edges: List[float] | None,
            bucket_meta: Mapping[str, Any],
            extra_meta: Mapping[str, Any],
        ) -> Dict[str, Any]:
            # Stable ordering in JSON.
            selected2 = list(selected)
            selected2.sort(key=lambda x: ((x.dataset or ""), x.sequence, x.frame, x.main_agent, x.coop_agents))

            frames_out: List[Dict[str, Any]] = []
            datasets_used: set[str] = set()
            for c in selected2:
                fr: Dict[str, Any] = {
                    "sequence": c.sequence,
                    "frame": c.frame,
                    "main_agent": c.main_agent,
                    "coop_agents": list(c.coop_agents),
                    "baseline_distance_m": float(c.baseline_m),
                }
                if c.dataset is not None:
                    fr["dataset"] = c.dataset
                    datasets_used.add(str(c.dataset))
                if edges is not None:
                    bi = _bucket_index(edges, c.baseline_m)
                    if bi is not None:
                        fr["baseline_bucket"] = _bucket_label(edges, int(bi))
                frames_out.append(fr)

            payload: Dict[str, Any] = {
                "split": str(split),
                "seed": seed,
                "sample_size": int(len(frames_out)),
                "index_parquet": str(index_path),
                "require_assets": bool(args.require_assets),
                "frames_hash_md5_v1": _frames_hash_md5_v1(frames_out),
                "frames_hash_md5_v2": _frames_hash_md5_v2(frames_out),
                "frames_hash_md5_v3": _frames_hash_md5_v3(frames_out),
                "total_candidates": int(kept),
                "skipped_candidates": int(skipped),
                "contract_kind": str(kind),
                "frames": frames_out,
            }
            if datasets_used:
                payload["datasets"] = sorted(datasets_used)
            payload.update(dict(bucket_meta))
            payload.update(dict(extra_meta))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            if args.self_check:
                _validate_written_contract(payload)
            return payload

        report: Dict[str, Any] = {
            "index_parquet": str(index_path),
            "split": str(split),
            "dataset_filter": dataset_filter or None,
            "seed": seed,
            "require_assets": bool(args.require_assets),
            "require_dataset": bool(args.require_dataset),
            "suite": suite_kinds,
            "out_dir": str(out_dir),
            "contracts": {},
        }
        if exclude_meta:
            report.update(dict(exclude_meta))
        if args.bucket_preset:
            report["bucket_preset"] = str(args.bucket_preset)
        if str(args.bucket_edges).strip():
            report["bucket_edges"] = str(args.bucket_edges).strip()

        def name_for(kind: str, n: int) -> str:
            if kind == "main":
                return f"frames_{split}{n}{dataset_mid}{tag_mid}_seed{seed}.json"
            if kind == "stress":
                return f"frames_{split}{n}{dataset_mid}{tag_mid}_stress_seed{seed}.json"
            if kind == "full":
                return f"frames_{split}{n}{dataset_mid}{tag_mid}_seed{seed}_full.json"
            raise ValueError(kind)

        if "main" in suite_kinds:
            rng = random.Random(seed)
            selected, bucket_meta = _select_candidates(
                candidates=candidates, rng=rng, sample_size=int(args.sample_size), edges=None, bucket_target_counts=None
            )
            if bool(args.require_dataset) and bool(has_dataset_col) and not dataset_filter:
                missing = [c for c in selected if c.dataset is None or not str(c.dataset).strip()]
                if missing:
                    preview = [
                        f"{c.sequence}/{c.frame}/main={c.main_agent}/coop={','.join(c.coop_agents)}"
                        for c in missing[:5]
                    ]
                    raise ValueError(
                        "Selected frames missing dataset but --require_dataset is set "
                        f"(split={split}, kind=main, missing={len(missing)}/{len(selected)}). "
                        f"Examples: {preview}"
                    )
            out_path = out_dir / name_for("main", int(args.sample_size))
            payload = write_contract(
                kind="main",
                out_path=out_path,
                selected=selected,
                edges=None,
                bucket_meta=bucket_meta,
                extra_meta={
                    **({"tag": tag} if tag else {}),
                    "require_dataset": bool(args.require_dataset),
                    **exclude_meta,
                },
            )
            report["contracts"]["main"] = {
                "path": str(out_path),
                "frames": int(payload["sample_size"]),
                "hash_v3": payload["frames_hash_md5_v3"],
            }

        if "stress" in suite_kinds:
            stress_n = int(args.stress_sample_size) if args.stress_sample_size is not None else int(args.sample_size)
            if args.bucket_preset:
                edges = preset_edges_m(str(args.bucket_preset))
            elif str(args.bucket_edges).strip():
                edges = _parse_edges(str(args.bucket_edges))
            else:
                edges = _parse_edges(DEFAULT_STRESS_BUCKET_EDGES_STR)

            bucket_target = _parse_int_list(str(args.bucket_counts)) if str(args.bucket_counts).strip() else None
            if bucket_target is None:
                if str(args.stress_policy) == "balanced":
                    bucket_target = _balanced_bucket_counts(sample_size=stress_n, num_buckets=len(edges) - 1)
                else:
                    bucket_target = None  # proportional (computed inside _select_candidates)

            rng = random.Random(seed)
            selected, bucket_meta = _select_candidates(
                candidates=candidates,
                rng=rng,
                sample_size=stress_n,
                edges=edges,
                bucket_target_counts=bucket_target,
            )
            if bool(args.require_dataset) and bool(has_dataset_col) and not dataset_filter:
                missing = [c for c in selected if c.dataset is None or not str(c.dataset).strip()]
                if missing:
                    preview = [
                        f"{c.sequence}/{c.frame}/main={c.main_agent}/coop={','.join(c.coop_agents)}"
                        for c in missing[:5]
                    ]
                    raise ValueError(
                        "Selected frames missing dataset but --require_dataset is set "
                        f"(split={split}, kind=stress, missing={len(missing)}/{len(selected)}). "
                        f"Examples: {preview}"
                    )
            out_path = out_dir / name_for("stress", stress_n)
            extra_meta = {**({"tag": tag} if tag else {}), "stress_policy": str(args.stress_policy)}
            if args.bucket_preset:
                extra_meta["bucket_preset"] = str(args.bucket_preset)
            extra_meta["require_dataset"] = bool(args.require_dataset)
            extra_meta.update(dict(exclude_meta))
            payload = write_contract(
                kind="stress",
                out_path=out_path,
                selected=selected,
                edges=edges,
                bucket_meta=bucket_meta,
                extra_meta=extra_meta,
            )
            report["contracts"]["stress"] = {
                "path": str(out_path),
                "frames": int(payload["sample_size"]),
                "hash_v3": payload["frames_hash_md5_v3"],
            }

        if "full" in suite_kinds:
            selected = list(candidates)
            if bool(args.require_dataset) and bool(has_dataset_col) and not dataset_filter:
                missing = [c for c in selected if c.dataset is None or not str(c.dataset).strip()]
                if missing:
                    preview = [
                        f"{c.sequence}/{c.frame}/main={c.main_agent}/coop={','.join(c.coop_agents)}"
                        for c in missing[:5]
                    ]
                    raise ValueError(
                        "Selected frames missing dataset but --require_dataset is set "
                        f"(split={split}, kind=full, missing={len(missing)}/{len(selected)}). "
                        f"Examples: {preview}"
                    )
            out_path = out_dir / name_for("full", int(len(selected)))
            payload = write_contract(
                kind="full",
                out_path=out_path,
                selected=selected,
                edges=None,
                bucket_meta={},
                extra_meta={
                    **({"tag": tag} if tag else {}),
                    "require_dataset": bool(args.require_dataset),
                    **exclude_meta,
                },
            )
            report["contracts"]["full"] = {
                "path": str(out_path),
                "frames": int(payload["sample_size"]),
                "hash_v3": payload["frames_hash_md5_v3"],
            }

        # Small machine-readable report for downstream scripting.
        if dataset_tag:
            report_name = (
                f"contract_suite_report_{dataset_tag}.json"
                if not multi_split
                else f"contract_suite_report_{dataset_tag}_{split}.json"
            )
        else:
            report_name = "contract_suite_report.json" if not multi_split else f"contract_suite_report_{split}.json"
        (out_dir / report_name).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        print(f"[OK] wrote contract suite under: {out_dir} (split={split})")
        for k in suite_kinds:
            v = report["contracts"].get(k)
            if not isinstance(v, dict):
                continue
            print(f"  - {k}: {v.get('path')} (frames={v.get('frames')})")


if __name__ == "__main__":
    main()
