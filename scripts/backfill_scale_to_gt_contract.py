#!/usr/bin/env python3
"""Backfill ratio-to-GT scale metrics into eval summaries from a frame contract.

Why this exists:
- Historical eval summaries often only contain legacy scale metrics based on `|s-1|`
  where `s=metric_scaling_factor`.
- For OPV2V + `norm_mode=avg_dis`, the correct per-frame GT target is not constant 1;
  it is the GT normalization factor computed from valid depth points in view0 frame.

This script reads:
1) a contract JSON with per-frame `gt_scale_single` / `gt_scale_coop`
2) one or more `summary_test.json` files
3) sibling CSV files `<summary_dir>/<model>/<mode>_metrics.csv`

and writes per-mode summary fields:
- `scale_gt_factor_mean`
- `scale_to_gt_err_mean`
- `scale_to_gt_log_err_mean`
- `scale_to_gt_eq_rel_err_mean`
- `scale_to_gt_ratio_mean`
- `scale_to_gt_ratio_median_mean`
- `scale_to_gt_ratio_p90_mean`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _frames_hash_md5(frames: Iterable[Dict[str, Any]]) -> str:
    items: List[str] = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        seq = f.get("sequence")
        frame = f.get("frame")
        if seq is None or frame is None:
            continue
        items.append(f"{seq}/{frame}")
    items.sort()
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    alpha = pos - lo
    return (1.0 - alpha) * xs[lo] + alpha * xs[hi]


def _compute_scale_to_gt_from_csv(
    csv_path: Path,
    *,
    gt_by_frame: Dict[Tuple[str, str], float],
) -> Dict[str, float] | None:
    ratios_to_gt: List[float] = []
    gt_vals: List[float] = []
    skipped_ambiguous = 0
    used_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        if "sequence" not in reader.fieldnames or "frame" not in reader.fieldnames:
            return None
        has_ratio = "scale_ratio_mean" in reader.fieldnames
        has_err = "scale_err" in reader.fieldnames

        for row in reader:
            seq = row.get("sequence")
            frame = row.get("frame")
            if seq is None or frame is None:
                continue
            gt = gt_by_frame.get((seq, frame))
            pred_ratio = None
            if has_ratio:
                pred_ratio = _safe_float(row.get("scale_ratio_mean"))
            elif has_err:
                # Legacy CSVs store `scale_err = |s-1|` without `scale_ratio_mean`.
                # If err>=1, s cannot be (1-err) (<=0), so we can uniquely infer s = 1+err.
                err = _safe_float(row.get("scale_err"))
                if err is None:
                    pred_ratio = None
                elif err >= 1.0:
                    pred_ratio = 1.0 + err
                else:
                    skipped_ambiguous += 1
                    continue
            else:
                return None
            if gt is None or pred_ratio is None or gt <= 1e-8:
                continue
            ratio_to_gt = pred_ratio / gt
            if not math.isfinite(ratio_to_gt) or ratio_to_gt <= 0.0:
                continue
            ratios_to_gt.append(ratio_to_gt)
            gt_vals.append(gt)
            used_rows += 1

    if not ratios_to_gt:
        return None

    abs_rel = [abs(r - 1.0) for r in ratios_to_gt]
    abs_log = [abs(math.log(max(r, 1e-8))) for r in ratios_to_gt]
    mean_abs_log = sum(abs_log) / len(abs_log)

    return {
        "scale_gt_factor_mean": float(sum(gt_vals) / len(gt_vals)),
        "scale_to_gt_err_mean": float(sum(abs_rel) / len(abs_rel)),
        "scale_to_gt_log_err_mean": float(mean_abs_log),
        "scale_to_gt_eq_rel_err_mean": float(math.expm1(mean_abs_log)),
        "scale_to_gt_ratio_mean": float(sum(ratios_to_gt) / len(ratios_to_gt)),
        "scale_to_gt_ratio_median_mean": float(_quantile(ratios_to_gt, 0.5)),
        "scale_to_gt_ratio_p90_mean": float(_quantile(ratios_to_gt, 0.9)),
        "scale_to_gt_rows_used": float(used_rows),
        "scale_to_gt_rows_skipped_ambiguous": float(skipped_ambiguous),
    }


def _contract_gt_maps(contract: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    frames = contract.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("contract json missing non-empty `frames` list")

    gt_single: Dict[Tuple[str, str], float] = {}
    gt_coop: Dict[Tuple[str, str], float] = {}
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        if seq is None or frame is None:
            continue
        key = (str(seq), str(frame))
        gs = _safe_float(fr.get("gt_scale_single"))
        gc = _safe_float(fr.get("gt_scale_coop"))
        if gs is not None:
            gt_single[key] = gs
        if gc is not None:
            gt_coop[key] = gc

    return gt_single, gt_coop


def backfill_one(
    summary_path: Path,
    *,
    contract: Dict[str, Any],
    gt_single: Dict[Tuple[str, str], float],
    gt_coop: Dict[Tuple[str, str], float],
    strict_hash: bool,
    dry_run: bool,
) -> Tuple[bool, str]:
    payload = _load_json(summary_path)
    metrics = payload.get("metrics")
    frames = payload.get("frames")
    if not isinstance(metrics, dict):
        return False, "skip:missing_metrics"
    if not isinstance(frames, list) or not frames:
        return False, "skip:missing_frames"

    summary_hash = _frames_hash_md5(frames)
    contract_hash = str(contract.get("frames_hash_md5") or "")
    if strict_hash and contract_hash and summary_hash != contract_hash:
        return False, f"skip:hash_mismatch summary={summary_hash} contract={contract_hash}"

    changed = False
    summary_dir = summary_path.parent
    for model_key, per_mode in metrics.items():
        if not isinstance(per_mode, dict):
            continue
        model_dir = summary_dir / str(model_key)
        for mode in ("single", "coop"):
            mode_metrics = per_mode.get(mode)
            if not isinstance(mode_metrics, dict):
                continue
            csv_path = model_dir / f"{mode}_metrics.csv"
            if not csv_path.is_file():
                continue
            gt_map = gt_single if mode == "single" else gt_coop
            stats = _compute_scale_to_gt_from_csv(csv_path, gt_by_frame=gt_map)
            if stats is None:
                continue
            for k, v in stats.items():
                prev = mode_metrics.get(k)
                if _safe_float(prev) is None or abs(float(prev) - v) > 1e-12:
                    mode_metrics[k] = v
                    changed = True

    if not changed:
        return False, "noop:no_changes"

    run_info = payload.get("run_info")
    if not isinstance(run_info, dict):
        run_info = {}
    run_info["scale_to_gt_contract"] = {
        "frames_hash_md5": contract_hash or summary_hash,
        "frames_json": contract.get("frames_json"),
        "norm_mode": contract.get("norm_mode"),
        "source": "scripts/backfill_scale_to_gt_contract.py",
    }
    payload["run_info"] = run_info

    if not dry_run:
        _dump_json(summary_path, payload)
    return True, "ok:updated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--contract_json",
        type=Path,
        required=True,
        help="JSON produced by compute_opv2v_gt_scale_contract.py",
    )
    ap.add_argument(
        "--strict_hash",
        action="store_true",
        help="Require summary frame hash to match contract hash.",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("summaries", nargs="+", type=Path)
    args = ap.parse_args()

    contract = _load_json(args.contract_json)
    gt_single, gt_coop = _contract_gt_maps(contract)

    updated = 0
    for p in args.summaries:
        if not p.is_file():
            print(f"[MISS] {p}")
            continue
        changed, status = backfill_one(
            p,
            contract=contract,
            gt_single=gt_single,
            gt_coop=gt_coop,
            strict_hash=bool(args.strict_hash),
            dry_run=bool(args.dry_run),
        )
        if changed:
            updated += 1
        print(f"[{status}] {p}")
    print(f"[DONE] updated={updated}/{len(args.summaries)}")


if __name__ == "__main__":
    main()
