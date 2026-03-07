#!/usr/bin/env python3
"""Compare det/e2e checkpoints under a baseline-distance bucket suite (no inference).

Expected directory layout (produced by `scripts/eval_det_ckpt_bucket_suite.sh`):

  <suite_root>/
    <model_name>/
      <contract_stem>/
        summary_test.json
        det_ap_by_baseline_buckets_<mode>.json

This script is contract-driven and checks fairness by comparing:
  - frames_hash_md5 (legacy)
  - frames_hash_md5_v2 (when present)
  - det_decode_cfg / seed / eval_script_md5 (as recorded in summary meta)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, float)):
        v = float(x)
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{nd}f}"
    return str(x)


def _discover_models(suite_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in suite_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
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


def _pick_det_model_key(metrics: object) -> str | None:
    if not isinstance(metrics, Mapping):
        return None
    if "det_model" in metrics:
        return "det_model"
    for k in metrics.keys():
        if isinstance(k, str):
            return k
    return None


def _extract_summary_metrics(summary: Mapping[str, Any], *, mode: str) -> Dict[str, Any]:
    metrics = summary.get("metrics") or {}
    det_key = _pick_det_model_key(metrics)
    per_mode = (metrics.get(det_key) if det_key and isinstance(metrics, Mapping) else {}) or {}
    mm = per_mode.get(mode) if isinstance(per_mode, Mapping) else {}
    mm = mm if isinstance(mm, Mapping) else {}
    return {
        "frames": mm.get("frames"),
        "det_ap_iou": mm.get("det_ap_iou"),
        "det_precision_iou": mm.get("det_precision_iou"),
        "det_recall_iou": mm.get("det_recall_iou"),
    }


def _load_bucket_report(p: Path) -> Tuple[List[str], Dict[str, Any]]:
    doc = _load_json(p)
    order = list(doc.get("bucket_order") or [])
    per: Dict[str, Any] = {}
    for b in doc.get("buckets") or []:
        if not isinstance(b, dict):
            continue
        name = b.get("bucket")
        if not isinstance(name, str):
            continue
        per[name] = b
        if name not in order:
            order.append(name)
    return order, doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite_root", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    ap.add_argument("--mode", type=str, default="coop", choices=["single", "coop"])
    ap.add_argument(
        "--contracts",
        nargs="*",
        default=None,
        help="Optional list of contract stems to compare (default: auto-discover from the first model dir).",
    )
    args = ap.parse_args(argv)

    suite_root = args.suite_root
    if not suite_root.is_dir():
        raise SystemExit(f"missing suite_root dir: {suite_root}")

    model_dirs = _discover_models(suite_root)
    if not model_dirs:
        raise SystemExit(f"no model dirs under: {suite_root}")

    contracts = args.contracts
    if not contracts:
        contracts = _discover_contracts(model_dirs[0])
    if not contracts:
        raise SystemExit(f"no contract dirs found under: {model_dirs[0]}")

    lines: List[str] = []
    lines.append("# Det Bucket Suite Comparison")
    lines.append("")
    lines.append(f"- suite_root: `{suite_root}`")
    lines.append(f"- mode: `{args.mode}`")
    lines.append(f"- models: {len(model_dirs)}")
    lines.append(f"- contracts: {', '.join(contracts)}")
    lines.append("")

    for contract in contracts:
        lines.append(f"## Contract: `{contract}`")
        lines.append("")

        expected_hash_v1: Optional[str] = None
        expected_hash_v2: Optional[str] = None

        # Overall metrics table.
        lines.append("### Overall Metrics (from `summary_test.json`)")
        lines.append("")
        lines.append(
            "| model | model_arch | frames_hash_md5 | frames_hash_md5_v2 | det_ap_iou | det_precision_iou | det_recall_iou |"
        )
        lines.append("| --- | --- | --- | --- | ---:| ---:| ---:|")

        for mdir in model_dirs:
            summary_p = mdir / contract / "summary_test.json"
            if not summary_p.is_file():
                continue
            summary = _load_json(summary_p)
            meta = summary.get("meta") or {}
            model_arch = str(meta.get("model_arch") or "NA")
            h1 = meta.get("frames_hash_md5")
            h2 = meta.get("frames_hash_md5_v2")
            h1s = str(h1) if isinstance(h1, str) else ""
            h2s = str(h2) if isinstance(h2, str) else ""
            if expected_hash_v1 is None and h1s:
                expected_hash_v1 = h1s
            if expected_hash_v2 is None and h2s:
                expected_hash_v2 = h2s
            mm = _extract_summary_metrics(summary, mode=args.mode)

            bad1 = expected_hash_v1 is not None and h1s and h1s != expected_hash_v1
            bad2 = expected_hash_v2 is not None and h2s and h2s != expected_hash_v2
            h1_cell = h1s + (" (!!)" if bad1 else "")
            h2_cell = h2s + (" (!!)" if bad2 else "")
            lines.append(
                "| "
                + " | ".join(
                    [
                        mdir.name,
                        model_arch,
                        h1_cell or "NA",
                        h2_cell or "NA",
                        _fmt(mm.get("det_ap_iou")),
                        _fmt(mm.get("det_precision_iou")),
                        _fmt(mm.get("det_recall_iou")),
                    ]
                )
                + " |"
            )
        lines.append("")
        if expected_hash_v1:
            lines.append(f"- expected frames_hash_md5: `{expected_hash_v1}`")
        if expected_hash_v2:
            lines.append(f"- expected frames_hash_md5_v2: `{expected_hash_v2}`")
        lines.append("")

        # Bucket report tables (if present).
        bucket_order: List[str] = []
        bucket_doc: Dict[str, Any] = {}
        bucket_src: Optional[Path] = None
        for mdir in model_dirs:
            p = mdir / contract / f"det_ap_by_baseline_buckets_{args.mode}.json"
            if p.is_file():
                bucket_src = p
                bucket_order, bucket_doc = _load_bucket_report(p)
                break

        if not bucket_order:
            lines.append("### Bucket Breakdown")
            lines.append("")
            lines.append(f"- Missing `det_ap_by_baseline_buckets_{args.mode}.json` for this contract; skip per-bucket comparison.")
            lines.append("")
            continue

        lines.append("### Bucket Stats (from det bucket report)")
        lines.append("")
        lines.append(f"- src: `{bucket_src}`" if bucket_src else "- src: NA")
        edges = bucket_doc.get("bucket_edges_m")
        if isinstance(edges, list) and edges:
            lines.append(f"- bucket_edges_m: `{edges}`")
        lines.append("")
        lines.append("| bucket | frames | total_gt | total_preds | dist_mean | AP |")
        lines.append("| --- | ---:| ---:| ---:| ---:| ---:|")
        per_bucket = {b.get("bucket"): b for b in (bucket_doc.get("buckets") or []) if isinstance(b, dict)}
        for b in bucket_order:
            rec = per_bucket.get(b) or {}
            dst = rec.get("dist_stats") or {}
            dist_mean = dst.get("mean")
            lines.append(
                "| "
                + " | ".join(
                    [
                        b,
                        str(rec.get("frames") or "NA"),
                        str(rec.get("total_gt") or "NA"),
                        str(rec.get("total_preds") or "NA"),
                        _fmt(dist_mean, 2),
                        _fmt(rec.get("ap")),
                    ]
                )
                + " |"
            )
        lines.append("")

        def emit_bucket_ap_table() -> None:
            lines.append(f"### {args.mode}: AP by bucket (from `det_ap_by_baseline_buckets_{args.mode}.json`)")
            lines.append("")
            lines.append("| model | " + " | ".join(bucket_order) + " | overall |")
            lines.append("| --- | " + " | ".join(["---:"] * len(bucket_order)) + " | ---:|")
            for mdir in model_dirs:
                p = mdir / contract / f"det_ap_by_baseline_buckets_{args.mode}.json"
                if not p.is_file():
                    continue
                doc = _load_json(p)
                ap_full = doc.get("ap_full")
                per = {b.get("bucket"): b for b in (doc.get("buckets") or []) if isinstance(b, dict)}
                row = [mdir.name]
                for b in bucket_order:
                    row.append(_fmt((per.get(b) or {}).get("ap")))
                row.append(_fmt(ap_full))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        emit_bucket_ap_table()

    out_md = args.out_md.expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

