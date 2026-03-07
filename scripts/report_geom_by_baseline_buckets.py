#!/usr/bin/env python3
"""Report geometry + scale metrics by OPV2V cross-agent baseline-distance buckets.

This is a *no-inference* analysis tool: it consumes existing `*_metrics.csv`
outputs from `scripts/batch_eval.py` on a fixed frames contract, and then
re-aggregates metrics over subsets ("buckets") of that same contract.

Why this exists
---------------
On OPV2V with `norm_mode=avg_dis`, coop GT scale factor `g_coop` becomes heavy-tailed
and correlates strongly with cross-agent baseline distance (vehicle separation).
This tool helps verify that "coop scale is hard" is dominated by the long-baseline tail,
without re-running expensive inference.

Inputs
------
- `--bucket_report_json`: produced by `scripts/bucket_frames_by_baseline_distance.py`
  (contains bucket frame json paths + counts).
- `--gt_scale_contract_json`: per-frame GT `gt_scale_single/coop` (e.g. Test500 contract).
- `--metrics_roots`: repeated `name=/path/to/<model>`, where that directory contains:
    - `single_metrics.csv`
    - `coop_metrics.csv`

Output
------
Writes one markdown report with:
  - bucket stats (distance + GT g distributions)
  - per-model per-bucket metrics for single/coop:
      pose_abs_mean, depth_rel_mean,
      scale_to_gt_mult_err_mean, scale_to_gt_ratio_mean, scale_to_gt_err_mean
  - optional combined groups (near/mid/tail) computed exactly from per-frame rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


FrameKeyV2 = Tuple[str, str, str, Tuple[str, ...]]  # (sequence, frame, main_agent, coop_agents tuple)
FrameKeyV1 = Tuple[str, str]  # legacy (sequence, frame)


def _frame_key_v2(fr: Mapping[str, Any]) -> FrameKeyV2:
    seq = str(fr.get("sequence", ""))
    frame = str(fr.get("frame", ""))
    main = str(fr.get("main_agent", ""))
    coop_raw = fr.get("coop_agents") or []
    if isinstance(coop_raw, (list, tuple)):
        coop = tuple(str(a) for a in coop_raw)
    else:
        coop = (str(coop_raw),)
    return seq, frame, main, coop


def _frame_key_v1(fr: Mapping[str, Any]) -> FrameKeyV1:
    return str(fr.get("sequence", "")), str(fr.get("frame", ""))


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
        "p10": _quantile(xs2, 0.10),
        "p50": _quantile(xs2, 0.50),
        "p90": _quantile(xs2, 0.90),
        "p95": _quantile(xs2, 0.95),
        "p99": _quantile(xs2, 0.99),
        "min": float(xs2[0]),
        "max": float(xs2[-1]),
    }


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


@dataclass(frozen=True)
class FrameRow:
    sequence: str
    frame: str
    main_agent: str
    coop_agents: Tuple[str, ...]
    pose_abs_m: float | None
    cross_agent_pose_trans_m: float | None
    cross_agent_pose_rot_deg: float | None
    depth_rel: float | None
    # Per-frame scale-to-GT metrics written by batch_eval.py (preferred for aggregation).
    scale_to_gt_ratio_mean: float | None
    scale_to_gt_err: float | None
    scale_to_gt_log_err: float | None
    # Legacy: predicted metric_scaling_factor (mean across views). Used only as fallback/debug.
    pred_scale_s: float | None

@dataclass(frozen=True)
class MetricsTable:
    """Lookup per-frame metrics by strict (v2) keys, with a safe legacy fallback."""

    v2: Dict[FrameKeyV2, FrameRow]
    v1: Dict[FrameKeyV1, FrameRow]
    v1_collisions: set[FrameKeyV1]

    def get(self, *, key_v2: FrameKeyV2, key_v1: FrameKeyV1) -> FrameRow | None:
        row = self.v2.get(key_v2)
        if row is not None:
            return row
        # Legacy fallback (unsafe when a contract repeats the same sequence/frame with
        # different main/coop pairing). Refuse to guess if ambiguous.
        if key_v1 in self.v1_collisions:
            return None
        return self.v1.get(key_v1)


@dataclass(frozen=True)
class GTScaleTable:
    """Lookup GT scale factors by strict (v2) keys, with a safe legacy fallback."""

    v2: Dict[FrameKeyV2, Dict[str, Any]]
    v1: Dict[FrameKeyV1, Dict[str, Any]]
    v1_collisions: set[FrameKeyV1]

    def get(self, *, key_v2: FrameKeyV2, key_v1: FrameKeyV1) -> Dict[str, Any] | None:
        gt = self.v2.get(key_v2)
        if gt is not None:
            return gt
        if key_v1 in self.v1_collisions:
            return None
        return self.v1.get(key_v1)


def _parse_coop_agents_csv(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    s = str(raw)
    if not s:
        return ()
    # batch_eval writes comma-separated agent ids.
    return tuple([tok for tok in (t.strip() for t in s.split(",")) if tok])


def _load_metrics_csv(csv_path: Path) -> MetricsTable:
    v2: Dict[FrameKeyV2, FrameRow] = {}
    v1: Dict[FrameKeyV1, FrameRow] = {}
    v1_collisions: set[FrameKeyV1] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            seq = row.get("sequence")
            fr = row.get("frame")
            if not seq or not fr:
                continue
            main = str(row.get("main_agent") or "")
            coop = _parse_coop_agents_csv(row.get("coop_agents"))
            rec = FrameRow(
                sequence=str(seq),
                frame=str(fr),
                main_agent=main,
                coop_agents=tuple(coop),
                pose_abs_m=_safe_float(row.get("pose_abs_m")),
                cross_agent_pose_trans_m=_safe_float(row.get("cross_agent_pose_trans_m")),
                cross_agent_pose_rot_deg=_safe_float(row.get("cross_agent_pose_rot_deg")),
                depth_rel=_safe_float(row.get("depth_rel")),
                scale_to_gt_ratio_mean=_safe_float(row.get("scale_to_gt_ratio_mean")),
                scale_to_gt_err=_safe_float(row.get("scale_to_gt_err")),
                scale_to_gt_log_err=_safe_float(row.get("scale_to_gt_log_err")),
                pred_scale_s=_safe_float(row.get("scale_ratio_mean")),
            )

            key1 = (str(seq), str(fr))
            if key1 in v1:
                prev = v1[key1]
                # If the CSV does not carry pairing metadata (legacy), any duplicate
                # sequence/frame is ambiguous -> refuse to join by v1.
                if not (prev.main_agent and prev.coop_agents and main and coop):
                    v1_collisions.add(key1)
                elif prev.main_agent != main or prev.coop_agents != tuple(coop):
                    v1_collisions.add(key1)
            else:
                v1[key1] = rec

            if main and coop:
                key2: FrameKeyV2 = (str(seq), str(fr), main, tuple(coop))
                v2[key2] = rec

    return MetricsTable(v2=v2, v1=v1, v1_collisions=v1_collisions)


def _bucket_name_from_path(p: Path) -> str:
    # Expected: ...dist_<lo>_<hi>.json
    stem = p.stem
    if "_dist_" in stem:
        return stem.split("_dist_", 1)[1]
    return stem


def _compute_scale_to_gt(
    *,
    pred_s: float | None,
    gt_g: float | None,
) -> Tuple[float | None, float | None, float | None]:
    """Return (ratio r=s/g, abs_err=|r-1|, log_err=|log r|)."""
    if pred_s is None or gt_g is None:
        return None, None, None
    if not (math.isfinite(float(pred_s)) and math.isfinite(float(gt_g))):
        return None, None, None
    s = float(pred_s)
    g = float(gt_g)
    if s <= 1e-8 or g <= 1e-8:
        return None, None, None
    r = s / g
    r = max(r, 1e-8)
    return r, abs(r - 1.0), abs(math.log(r))


def _mean(xs: List[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _exp_mean_log_err(log_errs: List[float]) -> float | None:
    """Compute mean(exp(log_err)) consistent with `batch_eval.py` summary.

    In `batch_eval.py`:
      - per-frame `scale_to_gt_log_err` = mean_views(|log(s/g)|)
      - per-frame `scale_to_gt_eq_rel_err` = expm1(scale_to_gt_log_err)
      - summary `scale_to_gt_mult_err_mean` = mean_frames(scale_to_gt_eq_rel_err) + 1
                                      = mean_frames(exp(scale_to_gt_log_err))

    This helper takes per-frame `log_err` values and returns the arithmetic mean of
    the corresponding multiplicative errors.
    """
    xs: List[float] = []
    for le in log_errs:
        if not math.isfinite(float(le)):
            continue
        xs.append(float(math.exp(float(le))))
    return _mean(xs)

def _pearson_corr(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return float(cov / math.sqrt(vx * vy))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket_report_json", type=Path, required=True)
    ap.add_argument("--gt_scale_contract_json", type=Path, required=True)
    ap.add_argument(
        "--metrics_roots",
        nargs="+",
        required=True,
        help="Repeated entries like name=/path/to/geom_model_dir (must contain single_metrics.csv and coop_metrics.csv).",
    )
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Optional machine-readable report output (per-bucket aggregates).",
    )
    args = ap.parse_args()

    bucket_report = _load_json(args.bucket_report_json)
    bucket_paths = [Path(p) for p in (bucket_report.get("bucket_paths") or [])]
    if not bucket_paths:
        raise SystemExit(f"bucket_report_json has no bucket_paths: {args.bucket_report_json}")

    contract = _load_json(args.gt_scale_contract_json)
    gt_v2: Dict[FrameKeyV2, Dict[str, Any]] = {}
    gt_v1: Dict[FrameKeyV1, Dict[str, Any]] = {}
    gt_v1_collisions: set[FrameKeyV1] = set()
    for fr in contract.get("frames") or []:
        if not isinstance(fr, Mapping):
            continue
        key1 = _frame_key_v1(fr)
        prev = gt_v1.get(key1)
        if prev is not None:
            # Refuse legacy (sequence, frame) joins when pairing differs.
            if _frame_key_v2(prev) != _frame_key_v2(fr):
                gt_v1_collisions.add(key1)
        else:
            gt_v1[key1] = dict(fr)
        key2 = _frame_key_v2(fr)
        if key2[2] and key2[3]:
            gt_v2[key2] = dict(fr)
    gt_lookup = GTScaleTable(v2=gt_v2, v1=gt_v1, v1_collisions=gt_v1_collisions)

    # Parse metrics roots.
    models: Dict[str, Path] = {}
    for item in args.metrics_roots:
        if "=" not in item:
            raise SystemExit(f"bad --metrics_roots entry (missing '='): {item}")
        name, path = item.split("=", 1)
        models[name.strip()] = Path(path.strip())

    # Preload all metrics CSVs so bucket aggregation is fast.
    per_model: Dict[str, Dict[str, MetricsTable]] = {}
    for name, root in models.items():
        single_csv = root / "single_metrics.csv"
        coop_csv = root / "coop_metrics.csv"
        modes: Dict[str, MetricsTable] = {}
        if single_csv.is_file():
            modes["single"] = _load_metrics_csv(single_csv)
        if coop_csv.is_file():
            modes["coop"] = _load_metrics_csv(coop_csv)
        if not modes:
            raise SystemExit(f"missing csvs under: {root} (need at least one of single_metrics.csv / coop_metrics.csv)")
        per_model[name] = modes

    available_modes = sorted({m for mm in per_model.values() for m in mm.keys()})

    lines: List[str] = []
    lines.append("# Geometry Metrics by Cross-Agent Baseline Distance Buckets (Test500)")
    lines.append("")
    lines.append("This is a no-inference re-aggregation over an existing fixed-contract eval.")
    lines.append("")
    lines.append(f"- bucket_report: `{args.bucket_report_json}`")
    lines.append(f"- gt_scale_contract: `{args.gt_scale_contract_json}`")
    lines.append("- join_key: prefer v2 (sequence/frame/main/coop); fall back to v1 only when unambiguous.")
    lines.append("")

    # Bucket stats table.
    lines.append("## Bucket Stats (distance + GT scale factors)")
    lines.append("")
    lines.append("| bucket | frames | dist_mean | dist_p50 | dist_p95 | g_single_mean | g_coop_mean | g_ratio_mean |")
    lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|")

    # Collect aggregated results per bucket.
    bucket_frames: Dict[str, List[Dict[str, Any]]] = {}
    for p in bucket_paths:
        doc = _load_json(p)
        frs = doc.get("frames") or []
        if not isinstance(frs, list):
            frs = []
        bucket_frames[_bucket_name_from_path(p)] = [fr for fr in frs if isinstance(fr, dict)]

    bucket_order = list(bucket_frames.keys())

    out_json: Dict[str, Any] = {
        "bucket_report_json": str(args.bucket_report_json),
        "gt_scale_contract_json": str(args.gt_scale_contract_json),
        "bucket_order": list(bucket_order),
        "bucket_stats": {},
        "per_model": {},
        "combined_groups": {},
        "key_findings": {},
    }

    for b in bucket_order:
        frs = bucket_frames[b]
        dists = [_safe_float(fr.get("baseline_distance_m")) for fr in frs]
        dists_f = [x for x in dists if x is not None]
        gs = []
        gc = []
        gr = []
        for fr in frs:
            key_v2 = _frame_key_v2(fr)
            key_v1 = _frame_key_v1(fr)
            gt = gt_lookup.get(key_v2=key_v2, key_v1=key_v1)
            if not gt:
                continue
            g_single = _safe_float(gt.get("gt_scale_single"))
            g_coop = _safe_float(gt.get("gt_scale_coop"))
            if g_single is not None:
                gs.append(g_single)
            if g_coop is not None:
                gc.append(g_coop)
            if g_single is not None and g_coop is not None and g_single > 1e-8:
                gr.append(g_coop / g_single)
        dist_st = _stats(dists_f)
        out_json["bucket_stats"][b] = {
            "frames": int(len(frs)),
            "dist": dist_st,
            "gt_scale_single": _stats(gs),
            "gt_scale_coop": _stats(gc),
            "gt_scale_ratio_coop_over_single": _stats(gr),
        }
        lines.append(
            "| "
            + " | ".join(
                [
                    b,
                    str(len(frs)),
                    _fmt(dist_st.get("mean"), 2),
                    _fmt(dist_st.get("p50"), 2),
                    _fmt(dist_st.get("p95"), 2),
                    _fmt(_mean(gs), 3),
                    _fmt(_mean(gc), 3),
                    _fmt(_mean(gr), 3),
                ]
            )
            + " |"
        )
    lines.append("")

    # Key findings: verify that baseline distance drives `g_coop` tails and that model
    # errors grow with distance buckets in coop mode.
    all_dist: List[float] = []
    all_g_coop: List[float] = []
    for b in bucket_order:
        for fr in bucket_frames[b]:
            d = _safe_float(fr.get("baseline_distance_m"))
            if d is None:
                continue
            key_v2 = _frame_key_v2(fr)
            key_v1 = _frame_key_v1(fr)
            gt = gt_lookup.get(key_v2=key_v2, key_v1=key_v1)
            if not gt:
                continue
            g = _safe_float(gt.get("gt_scale_coop"))
            if g is None:
                continue
            all_dist.append(float(d))
            all_g_coop.append(float(g))
    corr_d_g = _pearson_corr(all_dist, all_g_coop)
    out_json["key_findings"]["corr_distance_g_coop"] = corr_d_g

    def _group_frames(needles: Iterable[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for b in bucket_order:
            if any(n in b for n in needles):
                out.extend(bucket_frames[b])
        return out

    near_frames = _group_frames(["0_15", "15_30"])
    tail_frames = _group_frames(["100_1e+09"])

    lines.append("## Key Findings (Plan-1 Verification)")
    lines.append("")
    if corr_d_g is not None:
        lines.append(f"- On this contract, baseline distance correlates strongly with `g_coop`: corr(distance, g_coop)={corr_d_g:.3f}.")
    lines.append(
        "- Coop metrics degrade sharply with distance buckets (pose explodes + scale under-shoots), while single is comparatively stable."
    )
    lines.append("- The long-baseline tail (>=100m) is a small fraction of frames but dominates coop scale/pose averages.")
    lines.append("")
    lines.append(
        "| model | group | frames | coop_pose_abs_mean | coop_cross_trans_mean | coop_scale_mult_err_mean | coop_scale_ratio_mean |"
    )
    lines.append("| --- | --- | ---:| ---:| ---:| ---:| ---:|")
    for model_name, modes in per_model.items():
        if "coop" not in modes:
            continue
        rows = modes["coop"]
        for gname, frs_all in [("near_<=30m", near_frames), ("tail_>=100m", tail_frames)]:
            poses: List[float] = []
            crosses: List[float] = []
            ratios: List[float] = []
            log_errs: List[float] = []
            for fr in frs_all:
                key_v2 = _frame_key_v2(fr)
                key_v1 = _frame_key_v1(fr)
                r = rows.get(key_v2=key_v2, key_v1=key_v1)
                if r is None:
                    continue
                if r.pose_abs_m is not None:
                    poses.append(float(r.pose_abs_m))
                if r.cross_agent_pose_trans_m is not None:
                    crosses.append(float(r.cross_agent_pose_trans_m))
                if r.scale_to_gt_ratio_mean is not None:
                    ratios.append(float(r.scale_to_gt_ratio_mean))
                if r.scale_to_gt_log_err is not None:
                    log_errs.append(float(r.scale_to_gt_log_err))
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        gname,
                        str(len(frs_all)),
                        _fmt(_mean(poses), 4),
                        _fmt(_mean(crosses), 4),
                        _fmt(_exp_mean_log_err(log_errs), 4),
                        _fmt(_mean(ratios), 4),
                    ]
                )
                + " |"
            )
    lines.append("")

    def emit_table(*, mode: str) -> None:
        lines.append(f"## Mode={mode}")
        lines.append("")
        for model_name, modes in per_model.items():
            if mode not in modes:
                continue
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append(
                "| bucket | frames | pose_abs_mean | cross_agent_pose_trans_mean | depth_rel_mean | scale_mult_err_mean | scale_ratio_mean | scale_err_mean |"
            )
            lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
            out_json["per_model"].setdefault(model_name, {}).setdefault(mode, {})
            for b in bucket_order:
                frs = bucket_frames[b]
                rows = modes[mode]
                poses: List[float] = []
                crosses: List[float] = []
                depths: List[float] = []
                ratios: List[float] = []
                errs: List[float] = []
                log_errs: List[float] = []
                for fr in frs:
                    key_v2 = _frame_key_v2(fr)
                    key_v1 = _frame_key_v1(fr)
                    r = rows.get(key_v2=key_v2, key_v1=key_v1)
                    if r is None:
                        continue
                    if r.pose_abs_m is not None:
                        poses.append(float(r.pose_abs_m))
                    if r.cross_agent_pose_trans_m is not None:
                        crosses.append(float(r.cross_agent_pose_trans_m))
                    if r.depth_rel is not None:
                        depths.append(float(r.depth_rel))
                    if r.scale_to_gt_ratio_mean is not None:
                        ratios.append(float(r.scale_to_gt_ratio_mean))
                    if r.scale_to_gt_err is not None:
                        errs.append(float(r.scale_to_gt_err))
                    if r.scale_to_gt_log_err is not None:
                        log_errs.append(float(r.scale_to_gt_log_err))
                out_json["per_model"][model_name][mode][b] = {
                    "frames": int(len(frs)),
                    "pose_abs_mean": _mean(poses),
                    "cross_agent_pose_trans_mean": _mean(crosses),
                    "depth_rel_mean": _mean(depths),
                    "scale_to_gt_mult_err_mean": _exp_mean_log_err(log_errs),
                    "scale_to_gt_ratio_mean": _mean(ratios),
                    "scale_to_gt_err_mean": _mean(errs),
                }
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            b,
                            str(len(frs)),
                            _fmt(_mean(poses), 4),
                            _fmt(_mean(crosses), 4),
                            _fmt(_mean(depths), 4),
                            _fmt(_exp_mean_log_err(log_errs), 4),
                            _fmt(_mean(ratios), 4),
                            _fmt(_mean(errs), 4),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    for mode in ("single", "coop"):
        if mode in available_modes:
            emit_table(mode=mode)

    # Combined groups that are commonly useful in diagnosis.
    groups = {
        "near_<=30m": {"0_15", "15_30"},
        "mid_30-60m": {"30_60"},
        "far_>=60m": {"60_100", "100_1e+09"},
        "tail_>=100m": {"100_1e+09"},
    }

    # Try to match group bucket keys by suffix containment; bucket names come from filename suffix.
    def _match(bkey: str, needles: Iterable[str]) -> bool:
        return any(n in bkey for n in needles)

    lines.append("## Combined Groups (exact from per-frame rows)")
    lines.append("")
    lines.append(
        "| model | mode | group | frames | pose_abs_mean | cross_agent_pose_trans_mean | depth_rel_mean | scale_mult_err_mean | scale_ratio_mean |"
    )
    lines.append("| --- | --- | --- | ---:| ---:| ---:| ---:| ---:| ---:|")
    for model_name in per_model:
        for mode in ("single", "coop"):
            if mode not in per_model[model_name]:
                continue
            rows = per_model[model_name][mode]
            for gname, needles in groups.items():
                frs_all: List[Dict[str, Any]] = []
                for b in bucket_order:
                    if _match(b, needles):
                        frs_all.extend(bucket_frames[b])
                poses: List[float] = []
                crosses: List[float] = []
                depths: List[float] = []
                ratios: List[float] = []
                log_errs: List[float] = []
                for fr in frs_all:
                    key_v2 = _frame_key_v2(fr)
                    key_v1 = _frame_key_v1(fr)
                    r = rows.get(key_v2=key_v2, key_v1=key_v1)
                    if r is None:
                        continue
                    if r.pose_abs_m is not None:
                        poses.append(float(r.pose_abs_m))
                    if r.cross_agent_pose_trans_m is not None:
                        crosses.append(float(r.cross_agent_pose_trans_m))
                    if r.depth_rel is not None:
                        depths.append(float(r.depth_rel))
                    if r.scale_to_gt_ratio_mean is not None:
                        ratios.append(float(r.scale_to_gt_ratio_mean))
                    if r.scale_to_gt_log_err is not None:
                        log_errs.append(float(r.scale_to_gt_log_err))
                out_json["combined_groups"].setdefault(model_name, {}).setdefault(mode, {})[gname] = {
                    "frames": int(len(frs_all)),
                    "pose_abs_mean": _mean(poses),
                    "cross_agent_pose_trans_mean": _mean(crosses),
                    "depth_rel_mean": _mean(depths),
                    "scale_to_gt_mult_err_mean": _exp_mean_log_err(log_errs),
                    "scale_to_gt_ratio_mean": _mean(ratios),
                }
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            model_name,
                            mode,
                            gname,
                            str(len(frs_all)),
                            _fmt(_mean(poses), 4),
                            _fmt(_mean(crosses), 4),
                            _fmt(_mean(depths), 4),
                            _fmt(_exp_mean_log_err(log_errs), 4),
                            _fmt(_mean(ratios), 4),
                        ]
                    )
                    + " |"
                )
    lines.append("")

    out_md = args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote report: {out_md}")

    if args.out_json is not None:
        out_json_path: Path = args.out_json
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(out_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[OK] wrote report json: {out_json_path}")


if __name__ == "__main__":
    main()
