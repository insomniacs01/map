#!/usr/bin/env python3
"""Compare multiple geometry checkpoints under the baseline-distance bucket suite.

Expected directory layout (produced by `scripts/eval_geom_ckpt_bucket_suite.sh`):

  <suite_root>/
    <model_name>/
      <contract_stem>/               # e.g. frames_test500_nearest_seed42
        summary_test.json
        geom_by_baseline_buckets.json
        geom_by_baseline_buckets.md

This script is contract-driven and enforces fairness by checking `frames_hash_md5_v2`.
It also reports `eval_script_md5` to surface accidental protocol drift.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, float)):
        v = float(x)
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{nd}f}"
    return str(x)


def _rel_ckpt(p: str, *, repo_root: Path) -> str:
    try:
        pp = Path(p)
        return str(pp.relative_to(repo_root))
    except Exception:
        return str(p)


def _discover_models(suite_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in suite_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        # Heuristic: a model dir must contain at least one contract subdir with a summary.
        try:
            has_summary = any((c.is_dir() and (c / "summary_test.json").is_file()) for c in p.iterdir())
        except Exception:
            has_summary = False
        if has_summary:
            out.append(p)
    return sorted(out)


def _discover_contracts(model_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(model_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "summary_test.json").is_file():
            out.append(p.name)
    return out


def _extract_summary_metrics(summary: Dict[str, Any], *, mode: str = "coop") -> Dict[str, Any]:
    metrics = (summary.get("metrics") or {}).get("geom_model") or {}
    m = metrics.get(mode) or {}
    return {
        "frames": m.get("frames"),
        "scale_to_gt_mult_err_mean": m.get("scale_to_gt_mult_err_mean"),
        "scale_to_gt_ratio_mean": m.get("scale_to_gt_ratio_mean"),
        "cross_agent_pose_trans_mean": m.get("cross_agent_pose_trans_mean"),
        "depth_rel_mean": m.get("depth_rel_mean"),
        "pose_abs_mean": m.get("pose_abs_mean"),
    }


def _load_bucket_metrics(bucket_json: Dict[str, Any], *, mode: str = "coop") -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    bucket_order = list(bucket_json.get("bucket_order") or [])
    bucket_stats = bucket_json.get("bucket_stats") or {}
    per_model = bucket_json.get("per_model") or {}
    # `report_geom_by_baseline_buckets.py` is often called with metrics_roots like
    # "geom_model=<path>", so the key is typically "geom_model". Fall back to the
    # first entry if the caller used a custom name.
    if mode:
        if "geom_model" in per_model and isinstance(per_model["geom_model"], dict):
            per_mode = per_model["geom_model"].get(mode) or {}
        else:
            first = next(iter(per_model.values()), {})
            per_mode = (first.get(mode) or {}) if isinstance(first, dict) else {}
    else:
        per_mode = {}
    return bucket_order, bucket_stats, per_mode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite_root", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument(
        "--contracts",
        nargs="*",
        default=None,
        help="Optional list of contract stems to compare (default: auto-discover from the first model dir).",
    )
    ap.add_argument("--mode", type=str, default="coop", choices=["single", "coop"])
    args = ap.parse_args()

    suite_root = args.suite_root
    if not suite_root.is_dir():
        raise SystemExit(f"missing suite_root dir: {suite_root}")

    repo_root = Path(__file__).resolve().parents[1]

    model_dirs = _discover_models(suite_root)
    if not model_dirs:
        raise SystemExit(f"no model dirs under: {suite_root}")

    contracts = args.contracts
    if not contracts:
        contracts = _discover_contracts(model_dirs[0])
    if not contracts:
        raise SystemExit(f"no contract dirs found under: {model_dirs[0]}")

    lines: List[str] = []
    lines.append("# Geometry Bucket Suite Comparison")
    lines.append("")
    lines.append(f"- suite_root: `{suite_root}`")
    lines.append(f"- mode: `{args.mode}`")
    lines.append(f"- models: {len(model_dirs)}")
    lines.append(f"- contracts: {', '.join(contracts)}")
    lines.append("")

    for contract in contracts:
        lines.append(f"## Contract: `{contract}`")
        lines.append("")

        # Load one bucket json for bucket_stats + ordering.
        bucket_order: List[str] = []
        bucket_stats: Dict[str, Any] = {}
        expected_hash_v2: Optional[str] = None

        # Overall metrics table.
        lines.append("### Overall Metrics (from `summary_test.json`)")
        lines.append("")
        lines.append(
            "| model | model_arch | ckpt | frames_hash_md5_v2 | eval_script_md5 | scale_to_gt_mult_err_mean | scale_to_gt_ratio_mean | cross_agent_pose_trans_mean | depth_rel_mean | pose_abs_mean |"
        )
        lines.append("| --- | --- | --- | --- | --- | ---:| ---:| ---:| ---:| ---:|")

        rows: List[Tuple[str, str, str, str, str, Dict[str, Any]]] = []
        for mdir in model_dirs:
            summary_p = mdir / contract / "summary_test.json"
            if not summary_p.is_file():
                continue
            summary = _load_json(summary_p)
            meta = summary.get("meta") or {}
            model_arch = str(meta.get("model_arch") or "NA")
            frames_hash_v2 = str(meta.get("frames_hash_md5_v2") or "")
            if expected_hash_v2 is None and frames_hash_v2:
                expected_hash_v2 = frames_hash_v2
            eval_script_md5 = str(meta.get("eval_script_md5") or "")
            ckpt = (summary.get("model_paths") or {}).get("geom_model") or "NA"
            ckpt_rel = _rel_ckpt(str(ckpt), repo_root=repo_root)
            mm = _extract_summary_metrics(summary, mode=args.mode)
            rows.append((mdir.name, model_arch, ckpt_rel, frames_hash_v2, eval_script_md5, mm))

        # Emit overall table + track fairness.
        for model_name, model_arch, ckpt_rel, frames_hash_v2, eval_script_md5, mm in rows:
            bad = expected_hash_v2 is not None and frames_hash_v2 and frames_hash_v2 != expected_hash_v2
            hash_cell = frames_hash_v2 + (" (!!)" if bad else "")
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        model_arch,
                        f"`{ckpt_rel}`",
                        hash_cell,
                        eval_script_md5 or "NA",
                        _fmt(mm.get("scale_to_gt_mult_err_mean"), 4),
                        _fmt(mm.get("scale_to_gt_ratio_mean"), 4),
                        _fmt(mm.get("cross_agent_pose_trans_mean"), 4),
                        _fmt(mm.get("depth_rel_mean"), 4),
                        _fmt(mm.get("pose_abs_mean"), 4),
                    ]
                )
                + " |"
            )
        lines.append("")
        if expected_hash_v2 is not None:
            lines.append(f"- expected frames_hash_md5_v2: `{expected_hash_v2}`")
        else:
            lines.append("- expected frames_hash_md5_v2: NA (missing in summaries)")
        uniq_md5 = sorted({r[4] for r in rows if r[4]})
        if len(uniq_md5) == 1:
            lines.append(f"- eval_script_md5: `{uniq_md5[0]}`")
        elif len(uniq_md5) > 1:
            lines.append(f"- eval_script_md5: {', '.join(f'`{x}`' for x in uniq_md5)} (!!)")
        else:
            lines.append("- eval_script_md5: NA (missing in summaries)")
        lines.append("")

        # Bucket stats + per-bucket tables (use the first model that has bucket json).
        bucket_src: Optional[Path] = None
        for mdir in model_dirs:
            p = mdir / contract / "geom_by_baseline_buckets.json"
            if p.is_file():
                bucket_src = p
                doc = _load_json(p)
                bucket_order, bucket_stats, _ = _load_bucket_metrics(doc, mode=args.mode)
                break
        if not bucket_order:
            lines.append("### Bucket Breakdown")
            lines.append("")
            lines.append("- Missing `geom_by_baseline_buckets.json` for this contract; skip per-bucket comparison.")
            lines.append("")
            continue

        lines.append("### Bucket Stats (from `geom_by_baseline_buckets.json`)")
        lines.append("")
        lines.append(f"- src: `{bucket_src}`" if bucket_src else "- src: NA")
        lines.append("")
        lines.append("| bucket | frames | dist_mean | g_coop_mean | g_ratio_mean (coop/single) |")
        lines.append("| --- | ---:| ---:| ---:| ---:|")
        for b in bucket_order:
            st = bucket_stats.get(b) or {}
            dist = (st.get("dist") or {}).get("mean")
            g_coop = (st.get("gt_scale_coop") or {}).get("mean")
            g_ratio = (st.get("gt_scale_ratio_coop_over_single") or {}).get("mean")
            lines.append(
                "| "
                + " | ".join(
                    [
                        b,
                        str(st.get("frames") or "NA"),
                        _fmt(dist, 2),
                        _fmt(g_coop, 3),
                        _fmt(g_ratio, 3),
                    ]
                )
                + " |"
            )
        lines.append("")

        def emit_bucket_table(*, title: str, key: str, nd: int = 4) -> None:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| model | " + " | ".join(bucket_order) + " | overall |")
            lines.append("| --- | " + " | ".join(["---:"] * len(bucket_order)) + " | ---:|")
            for mdir in model_dirs:
                bj = mdir / contract / "geom_by_baseline_buckets.json"
                sj = mdir / contract / "summary_test.json"
                if not bj.is_file() or not sj.is_file():
                    continue
                bdoc = _load_json(bj)
                _, _, per_bucket = _load_bucket_metrics(bdoc, mode=args.mode)
                sdoc = _load_json(sj)
                overall = _extract_summary_metrics(sdoc, mode=args.mode).get(key)
                row = [mdir.name]
                for b in bucket_order:
                    row.append(_fmt((per_bucket.get(b) or {}).get(key), nd))
                row.append(_fmt(overall, nd))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        emit_bucket_table(title=f"{args.mode}: scale_to_gt_mult_err_mean by bucket", key="scale_to_gt_mult_err_mean", nd=4)
        emit_bucket_table(title=f"{args.mode}: scale_to_gt_ratio_mean by bucket", key="scale_to_gt_ratio_mean", nd=4)
        emit_bucket_table(
            title=f"{args.mode}: cross_agent_pose_trans_mean by bucket",
            key="cross_agent_pose_trans_mean",
            nd=4,
        )
        emit_bucket_table(title=f"{args.mode}: depth_rel_mean by bucket", key="depth_rel_mean", nd=4)
        emit_bucket_table(title=f"{args.mode}: pose_abs_mean by bucket", key="pose_abs_mean", nd=4)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {args.out_md}")


if __name__ == "__main__":
    main()
