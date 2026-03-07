#!/usr/bin/env python3
"""Bucket a frames contract JSON by cross-agent baseline distance (OPV2V).

Use case
--------
You have an existing frames contract (e.g. Test500) and want to split it into
sub-contracts by the distance between the main agent and its coop agent(s).

This is useful to:
  - diagnose why coop scale/pose tails blow up (often at large baselines)
  - run per-bucket evals fairly (same metric definition, different difficulty bins)
  - build curriculum / reweighting strategies

Distance definition
-------------------
Horizontal (XY) distance between the *LiDAR poses* (x, y) of the main agent and
each coop agent in world coordinates.

For frames with >2 agents, we use the "hardest" baseline:
  - baseline_distance_m = max_i ||(x_i, y_i) - (x_main, y_main)||_2

Input contract schema (expected)
--------------------------------
This script is designed to work with *any* frames contract that follows the
common schema used in this repo (not OPV2V-only).

Required per-frame keys:
  - sequence: str
  - frame: str
  - main_agent: str
  - coop_agents: list[str] (>=1; can include main_agent or not)

Optional per-frame keys:
  - baseline_distance_m: float
    If present, we will bucket by this value and will not touch dataset files.

If `baseline_distance_m` is missing, we fall back to OPV2V YAML metadata under
`--images_root` to compute baseline distance.

Output
------
Writes one frames JSON per bucket under `out_dir` and a summary report:
  - `bucket_report.json`
  - `bucket_report.md`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


from coop_bucket_presets import (  # noqa: E402
    COARSE_TRAIN_BUCKET_EDGES_M,
    COARSE_TRAIN_BUCKET_EDGES_STR,
    FINE_EVAL_BUCKET_EDGES_M as DEFAULT_BUCKET_EDGES_M,
    FINE_EVAL_BUCKET_EDGES_STR as DEFAULT_BUCKET_EDGES_STR,
    bucket_index as _bucket_index,
    bucket_label as _bucket_label,
    parse_edges as _parse_edges,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_edges_list(xs: Sequence[object]) -> List[float]:
    out: List[float] = []
    for x in xs:
        try:
            out.append(float(x))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"bucket_edges_m contains non-numeric value: {x!r}") from exc
    if len(out) < 2:
        raise ValueError("bucket_edges_m must contain at least 2 numbers")
    if any(not math.isfinite(v) for v in out):
        raise ValueError("bucket_edges_m contains non-finite values")
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError("bucket_edges_m must be strictly increasing")
    return out


@dataclass(frozen=True)
class FrameKey:
    sequence: str
    frame: str
    main_agent: str
    coop_agents: Tuple[str, ...]


def _baseline_distance_xy_m(*, images_root: Path, split: str, key: FrameKey) -> float:
    # Local import to keep this script lightweight for non-OPV2V environments.
    from data_processing.opv2v_pose_utils import load_frame_metadata  # noqa: WPS433

    def lidar_xy(agent: str) -> Tuple[float, float]:
        meta = load_frame_metadata(images_root / split / key.sequence / agent / f"{key.frame}.yaml")
        x, y = meta["lidar_pose"][:2]
        return float(x), float(y)

    x0, y0 = lidar_xy(key.main_agent)
    others = [a for a in key.coop_agents if a != key.main_agent]
    if not others:
        raise ValueError(f"coop_agents has no agent other than main_agent={key.main_agent}: {key.coop_agents}")
    dists = []
    for other in others:
        x1, y1 = lidar_xy(other)
        dists.append(float(math.hypot(x1 - x0, y1 - y0)))
    # For >2 agents, use the max baseline to reflect the hardest cross-agent geometry.
    return float(max(dists))


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
    """Dataset-aware hash: include optional per-frame `dataset` + pairing.

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


def _stats(xs: List[float]) -> Dict[str, float]:
    xs = sorted(xs)
    if not xs:
        return {}

    def q(p: float) -> float:
        i = int(round((len(xs) - 1) * p))
        return float(xs[i])

    return {
        "n": float(len(xs)),
        "mean": float(sum(xs) / len(xs)),
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "min": float(xs[0]),
        "max": float(xs[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames_json", type=Path, required=True)
    ap.add_argument("--images_root", type=Path, default=Path("map-anything/data/opv2v_images"))
    ap.add_argument(
        "--bucket_edges",
        type=str,
        default="",
        help=(
            "Optional comma-separated bucket edges in meters, e.g. '0,15,30,60,100,1e9'. "
            f"If omitted, uses frames_json.bucket_edges_m if present; otherwise defaults to '{DEFAULT_BUCKET_EDGES_STR}'."
        ),
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        choices=("", "coarse_train", "fine_eval"),
        help=(
            "Optional bucket-edge preset. "
            "coarse_train: [0,30,60,100,inf] (near/mid/far/tail). "
            "fine_eval: [0,15,30,60,100,inf]. "
            "Note: cannot be used together with --bucket_edges."
        ),
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="", help="Optional tag for output filenames.")
    ap.add_argument("--self_check", action="store_true", help="Validate written bucket contracts (schema + hashes).")
    args = ap.parse_args()

    doc = _load_json(args.frames_json)
    split = str(doc.get("split") or "")
    if not split:
        raise ValueError(f"bad frames_json (missing split): {args.frames_json}")
    frames = doc.get("frames") or []
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"bad frames_json (missing frames): {args.frames_json}")

    # Choose bucket edges: preset/explicit CLI > contract > protocol default.
    edges_source = "default"
    if args.preset and args.bucket_edges.strip():
        raise ValueError("Cannot use --preset together with --bucket_edges")
    if args.preset:
        if str(args.preset) == "coarse_train":
            edges = list(COARSE_TRAIN_BUCKET_EDGES_M)
            edges_source = "preset:coarse_train"
        else:
            edges = list(DEFAULT_BUCKET_EDGES_M)
            edges_source = "preset:fine_eval"
    elif args.bucket_edges.strip():
        edges = _parse_edges(args.bucket_edges)
        edges_source = "cli"
    else:
        contract_edges = doc.get("bucket_edges_m")
        if isinstance(contract_edges, (list, tuple)) and len(contract_edges) >= 2:
            edges = _parse_edges_list(contract_edges)
            edges_source = "contract"
        else:
            edges = list(DEFAULT_BUCKET_EDGES_M)
            edges_source = "default"

    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(len(edges) - 1)]
    dists: List[float] = []
    dist_source_counts = {"contract": 0, "opv2v_yaml": 0}
    unassigned_count = 0

    for fr in frames:
        if not isinstance(fr, dict):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        main = fr.get("main_agent")
        coop = fr.get("coop_agents") or []
        if not (seq and frame and main) or not isinstance(coop, (list, tuple)) or len(coop) < 1:
            continue
        coop_agents = tuple(str(a) for a in coop if a is not None)
        if not coop_agents:
            continue
        key = FrameKey(
            sequence=str(seq),
            frame=str(frame),
            main_agent=str(main),
            coop_agents=coop_agents,
        )
        dist = None
        dist_raw = fr.get("baseline_distance_m")
        if isinstance(dist_raw, (int, float)) and math.isfinite(float(dist_raw)):
            dist = float(dist_raw)
            dist_source_counts["contract"] += 1
        else:
            dist = _baseline_distance_xy_m(images_root=args.images_root, split=split, key=key)
            dist_source_counts["opv2v_yaml"] += 1
        dists.append(dist)
        bi = _bucket_index(edges, dist)
        if bi is None:
            unassigned_count += 1
            continue
        fr_out = dict(fr)
        fr_out["baseline_distance_m"] = float(dist)
        fr_out["baseline_bucket"] = _bucket_label(edges, int(bi))
        buckets[int(bi)].append(fr_out)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta = {
        k: v
        for k, v in doc.items()
        if k
        not in {
            "frames",
            # These are full-contract stratification metadata; they do not apply to per-bucket subsets.
            "bucket_available_counts",
            "bucket_target_counts",
            "bucket_labels",
        }
    }

    tag = f"_{args.tag}" if args.tag else ""
    out_paths: List[str] = []
    for i, frs in enumerate(buckets):
        lo, hi = edges[i], edges[i + 1]
        if str(args.preset) == "coarse_train" and len(edges) - 1 == 4:
            lane_names = ["near_0_30m", "mid_30_60m", "far_60_100m", "tail_100_inf_m"]
            lane = lane_names[i] if 0 <= i < len(lane_names) else f"B{i}"
            name = f"{args.frames_json.stem}{tag}_{lane}.json"
        else:
            name = f"{args.frames_json.stem}{tag}_dist_{lo:g}_{hi:g}.json"
        out_path = out_dir / name
        payload = dict(base_meta)
        payload["split"] = split
        payload["sample_size"] = int(len(frs))
        payload["bucket_edges_m"] = edges
        payload["bucket_index"] = int(i)
        payload["bucket_label"] = _bucket_label(edges, i)
        payload["bucket_range_m"] = [float(lo), float(hi)]
        payload["frames_hash_md5_v1"] = _frames_hash_md5_v1(frs)
        payload["frames_hash_md5_v2"] = _frames_hash_md5_v2(frs)
        payload["frames_hash_md5_v3"] = _frames_hash_md5_v3(frs)
        payload["frames"] = frs
        _dump_json(out_path, payload)
        if args.self_check:
            # Basic parity checks (subset contracts should be consistent).
            if payload["sample_size"] != len(payload["frames"]):
                raise AssertionError("sample_size mismatch in bucket contract")
            if payload["frames_hash_md5_v1"] != _frames_hash_md5_v1(payload["frames"]):
                raise AssertionError("frames_hash_md5_v1 mismatch in bucket contract")
            if payload["frames_hash_md5_v2"] != _frames_hash_md5_v2(payload["frames"]):
                raise AssertionError("frames_hash_md5_v2 mismatch in bucket contract")
            if payload["frames_hash_md5_v3"] != _frames_hash_md5_v3(payload["frames"]):
                raise AssertionError("frames_hash_md5_v3 mismatch in bucket contract")
        out_paths.append(str(out_path))

    report = {
        "frames_json": str(args.frames_json),
        "split": split,
        "images_root": str(args.images_root),
        "bucket_edges_source": edges_source,
        "preset": str(args.preset) if args.preset else None,
        "distance_sources": dist_source_counts,
        "bucket_edges_m": edges,
        "bucket_labels": [_bucket_label(edges, i) for i in range(len(edges) - 1)],
        "distance_stats_m": _stats(dists),
        "bucket_counts": [len(x) for x in buckets],
        "unassigned_frames": int(unassigned_count),
        "bucket_paths": out_paths,
    }
    _dump_json(out_dir / "bucket_report.json", report)

    # Human-friendly markdown report.
    md = []
    md.append("# OPV2V Frames Contract Buckets (baseline distance)")
    md.append("")
    md.append(f"- source: `{args.frames_json}`")
    md.append(f"- split: `{split}`")
    md.append(f"- images_root: `{args.images_root}`")
    md.append(f"- bucket_edges_m: `{edges}`")
    if args.preset:
        md.append(f"- preset: `{args.preset}`")
    md.append("")
    st = report["distance_stats_m"]
    if st:
        md.append("## Distance Stats (m)")
        md.append("")
        md.append(
            f"- n={int(st['n'])} mean={st['mean']:.2f} p50={st['p50']:.2f} p95={st['p95']:.2f} p99={st['p99']:.2f} "
            f"min={st['min']:.2f} max={st['max']:.2f}"
        )
        md.append("")
    md.append("## Buckets")
    md.append("")
    md.append("| bucket | range(m) | frames | out_json |")
    md.append("| --- | --- | ---:| --- |")
    for i, frs in enumerate(buckets):
        lo, hi = edges[i], edges[i + 1]
        p = Path(out_paths[i])
        try:
            rel = p.resolve().relative_to(Path.cwd().resolve())
            rel_s = rel.as_posix()
        except Exception:
            rel_s = p.as_posix()
        md.append(f"| B{i} | [{lo:g}, {hi:g}) | {len(frs)} | `{rel_s}` |")
    md.append("")
    (out_dir / "bucket_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] wrote bucketed contracts under: {out_dir}")
    print(f"[OK] report: {out_dir / 'bucket_report.md'}")


if __name__ == "__main__":
    main()
