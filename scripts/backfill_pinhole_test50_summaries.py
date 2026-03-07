#!/usr/bin/env python3
"""
Backfill missing metadata + ratio-space scale diagnostics for pinhole test50 eval summaries.

Why this exists:
  - Some older `eval_runs/**/summary_test.json` artifacts predate `run_info` and the newer
    ratio-space scale diagnostic fields (scale_log_err_mean / scale_eq_rel_err_mean / scale_ratio_mean).
  - Re-running inference just to enrich metadata is expensive; we instead derive what we can
    from existing artifacts in the same eval directory:
      - `*_metrics.csv` provides per-frame `scale_err` (== |ratio-1|) which is enough to
        derive log error and equivalent relative error when `scale_err >= 1` (ratio must be > 0).
      - `*_representatives/*_pred_boxes.json` provides det decode thresholds (iou/score).

Safety / correctness notes:
  - For frames where `scale_err < 1`, the underlying ratio could be either `1+err` or `1-err`.
    We treat those as ambiguous and skip them in the derived ratio/log stats (and record a note).
  - We never overwrite existing keys; only add missing fields.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _frames_hash(frames: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        seq = f.get("sequence")
        fr = f.get("frame")
        if seq is None or fr is None:
            continue
        items.append(f"{seq}/{fr}")
    items.sort()
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _iter_scale_err_from_csv(path: Path) -> Iterable[float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "scale_err" not in r.fieldnames:
            return
        for row in r:
            try:
                v = float(row.get("scale_err", "nan"))
            except Exception:
                continue
            if math.isfinite(v):
                yield v


def _derive_scale_stats_from_scale_err(errs: List[float]) -> Tuple[Optional[Dict[str, float]], str]:
    """Return (stats, note)."""
    if not errs:
        return None, "no_finite_scale_err"

    ratios: List[float] = []
    ambig = 0
    for e in errs:
        # scale_err is |r-1|. If e>=1 then r cannot be 1-e (<=0), so r=1+e is uniquely determined.
        if e >= 1.0:
            ratios.append(1.0 + e)
        else:
            ambig += 1

    if not ratios:
        return None, f"all_scale_err_ambiguous_lt1 (n={ambig})"

    # ratio-space diagnostics matching `scripts/batch_eval.py` conventions.
    log_err = [abs(math.log(max(r, 1e-12))) for r in ratios]
    mean_abs_log = sum(log_err) / len(log_err)
    stats = {
        "scale_log_err_mean": float(mean_abs_log),
        "scale_eq_rel_err_mean": float(math.expm1(mean_abs_log)),
        "scale_ratio_mean": float(sum(ratios) / len(ratios)),
    }

    note = ""
    if ambig:
        note = f"derived_scale_ratio_used_only_scale_err>=1 (skipped_ambiguous={ambig}/{len(errs)})"
    return stats, note


def _find_one_det_decode_cfg(summary_dir: Path) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Try to discover det decode thresholds from any `*_pred_boxes.json` representative file."""
    for p in summary_dir.rglob("*_pred_boxes.json"):
        try:
            d = _load_json(p)
        except Exception:
            continue
        iou = d.get("iou_thresh")
        score = d.get("score_thresh")
        if isinstance(iou, (int, float)) and isinstance(score, (int, float)):
            return {"iou_thresh": float(iou), "score_thresh": float(score)}, str(p)
    return None, None


def _ensure_run_info(summary: Dict[str, Any], *, summary_path: Path) -> None:
    frames = summary.get("frames")
    if not isinstance(frames, list):
        return

    # Preserve existing run_info if present.
    run_info = summary.get("run_info")
    if not isinstance(run_info, dict):
        run_info = {}

    run_info.setdefault("backfilled_by", "scripts/backfill_pinhole_test50_summaries.py")
    run_info.setdefault("backfilled_at", _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z"))
    run_info.setdefault("sample_size", len(frames))
    run_info.setdefault("frames_hash_md5", _frames_hash(frames))

    det_cfg, det_cfg_src = _find_one_det_decode_cfg(summary_path.parent)
    if det_cfg is not None:
        run_info.setdefault("det_decode_cfg", det_cfg)
        run_info.setdefault(
            "det_decode_cfg_source",
            str(Path(det_cfg_src).relative_to(summary_path.parent.parent)) if det_cfg_src else det_cfg_src,
        )

    summary["run_info"] = run_info


def _backfill_scale_diagnostics(summary: Dict[str, Any], *, summary_path: Path) -> None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return

    notes: List[str] = []
    for model_key, per_mode in metrics.items():
        if not isinstance(per_mode, dict):
            continue
        for mode, m in per_mode.items():
            if not isinstance(m, dict):
                continue
            # Only backfill if missing (do not overwrite).
            if all(k in m for k in ("scale_log_err_mean", "scale_eq_rel_err_mean", "scale_ratio_mean")):
                continue

            csv_path = summary_path.parent / model_key / f"{mode}_metrics.csv"
            if not csv_path.exists():
                continue
            errs = list(_iter_scale_err_from_csv(csv_path))
            stats, note = _derive_scale_stats_from_scale_err(errs)
            if stats is None:
                if note:
                    notes.append(f"{model_key}/{mode}: {note}")
                continue
            for k, v in stats.items():
                m.setdefault(k, v)
            if note:
                notes.append(f"{model_key}/{mode}: {note}")

    if notes:
        run_info = summary.get("run_info")
        if not isinstance(run_info, dict):
            run_info = {}
        existing = run_info.get("notes")
        if isinstance(existing, list):
            run_info["notes"] = existing + notes
        elif isinstance(existing, str) and existing.strip():
            run_info["notes"] = [existing] + notes
        else:
            run_info["notes"] = notes
        summary["run_info"] = run_info


def backfill_one(summary_path: Path) -> None:
    summary = _load_json(summary_path)
    _ensure_run_info(summary, summary_path=summary_path)
    _backfill_scale_diagnostics(summary, summary_path=summary_path)
    _dump_json(summary_path, summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", type=Path, help="Paths to summary_test.json files to patch in-place.")
    args = ap.parse_args()

    for p in args.summaries:
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        backfill_one(p)


if __name__ == "__main__":
    main()

