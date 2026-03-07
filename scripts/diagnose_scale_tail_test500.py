#!/usr/bin/env python3
"""Diagnose scale tail failures under a fixed OPV2V contract (Test500).

This script connects three things under ONE contract:
1) per-frame model metrics from `batch_eval.py` CSV (`*_metrics.csv`)
2) per-frame GT scale factor `g` and GT depth valid-pixel counts from the GT scale contract JSON
3) derived OPV2V-correct scale errors in ratio-to-GT space (`s/g`)

It writes:
- a combined CSV with per-frame derived fields (scale_to_gt_* + gt_valid_ratio)
- a concise markdown report (summary + top worst frames)

Usage (example):
  python scripts/diagnose_scale_tail_test500.py \
    --eval_run_dir eval_runs/geom_generalize_test500_20260216_c20260210_sfix_direct \
    --contract_json eval_runs/gt_scale_test500_seed42_avg_dis.json \
    --depth_root data/opv2v_depth \
    --split test \
    --out_dir eval_runs/_diagnose_scale_tail_test500_20260218
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _infer_pred_scale_s(row: Dict[str, str]) -> float | None:
    # `scale_ratio_mean` is the direct model-reported `s` (legacy ratio-to-1).
    if "scale_ratio_mean" in row:
        v = _safe_float(row.get("scale_ratio_mean"))
        if v is not None and v > 0:
            return v
    # Fallback: if we only have `scale_err=|s-1|` and err>=1, s is unambiguous: s=1+err.
    if "scale_err" in row:
        e = _safe_float(row.get("scale_err"))
        if e is None:
            return None
        if e >= 1.0:
            return 1.0 + e
    return None


def _load_metrics_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("sequence") or not row.get("frame"):
                continue
            rows.append(row)
    return rows


def _load_contract_map(contract_json: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    doc = _load_json(contract_json)
    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"bad contract json: {contract_json}")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        if seq is None or frame is None:
            continue
        out[(str(seq), str(frame))] = fr
    return out


def _load_official_summary(eval_run_dir: Path) -> Dict[str, Any] | None:
    p = eval_run_dir / "summary_test.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_official_metrics(summary: Dict[str, Any] | None, *, model: str, mode: str) -> Dict[str, float]:
    if not summary:
        return {}
    try:
        m = summary["metrics"][model][mode]
        out: Dict[str, float] = {}
        for k in [
            "scale_to_gt_mult_err_mean",
            "scale_to_gt_ratio_mean",
            "pose_abs_mean",
            "depth_rel_mean",
        ]:
            v = m.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[k] = float(v)
        return out
    except Exception:
        return {}


def _read_depth_hw(depth_root: Path, split: str, seq: str, agent: str, frame: str) -> Tuple[int, int]:
    # Read one depth file to infer H/W (assumed constant across dataset).
    p = depth_root / split / seq / agent / f"{frame}_camera0_depth.npy"
    arr = np.load(p)
    if arr.ndim != 2:
        raise ValueError(f"depth npy must be 2D; got {arr.shape} at {p}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"bad depth shape: {arr.shape} at {p}")
    return h, w


def _corr_pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan")
    c = np.corrcoef(xv, yv)[0, 1]
    return float(c)


def _corr_spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(xv))
    ry = np.argsort(np.argsort(yv))
    return _corr_pearson(rx.tolist(), ry.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_run_dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="geom_model")
    ap.add_argument("--contract_json", type=Path, required=True)
    ap.add_argument("--depth_root", type=Path, default=REPO_ROOT / "data" / "opv2v_depth")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    # Normalize paths to avoid `Path.relative_to` surprises when users pass relative paths.
    args.eval_run_dir = args.eval_run_dir.resolve()
    args.contract_json = args.contract_json.resolve()
    args.depth_root = args.depth_root.resolve()
    args.out_dir = args.out_dir.resolve()

    eval_run_dir = args.eval_run_dir
    model = str(args.model)
    contract_map = _load_contract_map(args.contract_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    official_summary = _load_official_summary(eval_run_dir)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except Exception:
            return str(p)

    # Infer depth H/W once (use first contract frame).
    any_fr = next(iter(contract_map.values()))
    seq0 = str(any_fr["sequence"])
    frame0 = str(any_fr["frame"])
    agent0 = str(any_fr.get("main_agent") or "641")
    H, W = _read_depth_hw(args.depth_root, args.split, seq0, agent0, frame0)

    md_lines: List[str] = []
    md_lines.append("# Scale Tail Diagnosis (Test500 Contract)")
    md_lines.append("")
    md_lines.append(f"- eval_run_dir: `{_rel(eval_run_dir)}`")
    md_lines.append(f"- model: `{model}`")
    md_lines.append(f"- contract_json: `{_rel(args.contract_json)}`")
    md_lines.append(f"- split: `{args.split}`")
    md_lines.append(f"- depth_root: `{args.depth_root}`")
    md_lines.append(f"- inferred depth shape: H={H}, W={W}")
    md_lines.append("")

    for mode in ["single", "coop"]:
        official = _get_official_metrics(official_summary, model=model, mode=mode)

        csv_path = eval_run_dir / model / f"{mode}_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing metrics csv: {csv_path}")

        rows = _load_metrics_csv(csv_path)
        out_csv = args.out_dir / f"{model}_{mode}_scale_tail.csv"

        joined: List[Dict[str, Any]] = []
        xs_valid_ratio: List[float] = []
        ys_log_err: List[float] = []
        ys_pose_abs: List[float] = []

        for r in rows:
            seq = str(r["sequence"])
            frame = str(r["frame"])
            meta = contract_map.get((seq, frame))
            if meta is None:
                continue

            s = _infer_pred_scale_s(r)
            gt_scale_key = "gt_scale_single" if mode == "single" else "gt_scale_coop"
            gt_valid_key = "gt_valid_pixels_single" if mode == "single" else "gt_valid_pixels_coop"
            g = _safe_float(meta.get(gt_scale_key))
            gt_valid_pixels = _safe_float(meta.get(gt_valid_key))
            coop_agents = meta.get("coop_agents") or []
            if not isinstance(coop_agents, list):
                coop_agents = []
            num_views = 4 if mode == "single" else 4 * max(1, len(coop_agents))

            gt_valid_ratio = None
            if gt_valid_pixels is not None and num_views > 0:
                gt_valid_ratio = float(gt_valid_pixels) / float(num_views * H * W)

            ratio_to_gt = None
            err = None
            log_err = None
            mult_err = None
            if s is not None and g is not None and g > 1e-8 and s > 1e-8:
                ratio_to_gt = float(s / g)
                ratio_to_gt = max(ratio_to_gt, 1e-8)
                err = abs(ratio_to_gt - 1.0)
                log_err = abs(math.log(ratio_to_gt))
                mult_err = math.exp(log_err)

            pose_abs = _safe_float(r.get("pose_abs_m"))
            depth_rel = _safe_float(r.get("depth_rel"))

            rec = {
                "sequence": seq,
                "frame": frame,
                "pose_abs_m": pose_abs,
                "depth_rel": depth_rel,
                "pred_scale_s": s,
                "gt_scale_g": g,
                # NOTE: These are derived from `scale_ratio_mean` (mean across views),
                # so they are an approximation of the official `batch_eval.py` ratio-to-GT keys.
                "scale_to_gt_ratio_approx": ratio_to_gt,
                "scale_to_gt_err_approx": err,
                "scale_to_gt_log_err_approx": log_err,
                "scale_to_gt_mult_err_approx": mult_err,
                "gt_valid_pixels": gt_valid_pixels,
                "gt_valid_ratio": gt_valid_ratio,
                "num_views": num_views,
            }
            joined.append(rec)

            if gt_valid_ratio is not None and log_err is not None and math.isfinite(gt_valid_ratio) and math.isfinite(log_err):
                xs_valid_ratio.append(float(gt_valid_ratio))
                ys_log_err.append(float(log_err))
            if gt_valid_ratio is not None and pose_abs is not None and math.isfinite(gt_valid_ratio) and math.isfinite(pose_abs):
                ys_pose_abs.append(float(pose_abs))

        # Write CSV.
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(joined[0].keys()) if joined else [])
            w.writeheader()
            for rec in joined:
                w.writerow(rec)

        # Report summary.
        md_lines.append(f"## Mode={mode}")
        md_lines.append("")
        md_lines.append(f"- rows: {len(joined)}")
        if official:
            md_lines.append(
                "- official(Test500, from summary_test.json): "
                + ", ".join([f"{k}={v:.6g}" for k, v in official.items()])
            )
        if joined:
            mult_errs = [v["scale_to_gt_mult_err_approx"] for v in joined if isinstance(v.get("scale_to_gt_mult_err_approx"), (int, float)) and math.isfinite(float(v["scale_to_gt_mult_err_approx"]))]  # type: ignore[index]
            ratios = [v["scale_to_gt_ratio_approx"] for v in joined if isinstance(v.get("scale_to_gt_ratio_approx"), (int, float)) and math.isfinite(float(v["scale_to_gt_ratio_approx"]))]  # type: ignore[index]
            valid_ratios = [v["gt_valid_ratio"] for v in joined if isinstance(v.get("gt_valid_ratio"), (int, float)) and math.isfinite(float(v["gt_valid_ratio"]))]  # type: ignore[index]
            pose_abs_list = [v["pose_abs_m"] for v in joined if isinstance(v.get("pose_abs_m"), (int, float)) and math.isfinite(float(v["pose_abs_m"]))]  # type: ignore[index]

            def q(arr: List[float], p: float) -> float:
                if not arr:
                    return float("nan")
                return float(np.quantile(np.asarray(arr, dtype=np.float64), p))

            md_lines.append(
                "- approx(scale_to_gt_mult_err) (1 is best): "
                f"mean={np.mean(mult_errs):.3f} median={np.median(mult_errs):.3f} p90={q(mult_errs,0.9):.3f}"
            )
            md_lines.append(
                "- approx(scale_to_gt_ratio) (1 is best): "
                f"mean={np.mean(ratios):.3f} median={np.median(ratios):.3f} p90={q(ratios,0.9):.3f}"
            )
            md_lines.append(
                "- gt_valid_ratio (depth>0 density proxy): "
                f"mean={np.mean(valid_ratios):.4f} median={np.median(valid_ratios):.4f} p10={q(valid_ratios,0.1):.4f}"
            )
            if pose_abs_list:
                md_lines.append(f"- pose_abs_m: mean={np.mean(pose_abs_list):.3f} p90={q(pose_abs_list,0.9):.3f} max={max(pose_abs_list):.3f}")
            md_lines.append("")

            pear = _corr_pearson(xs_valid_ratio, ys_log_err)
            spear = _corr_spearman(xs_valid_ratio, ys_log_err)
            md_lines.append(
                f"- corr(depth_valid_ratio, approx(scale_to_gt_log_err)): pearson={pear:.3f} spearman={spear:.3f}"
            )
            md_lines.append("")

            # Top worst by scale mult err.
            worst = [v for v in joined if v.get("scale_to_gt_mult_err_approx") is not None]
            worst.sort(key=lambda x: float(x.get("scale_to_gt_mult_err_approx") or -1.0), reverse=True)
            md_lines.append(f"### Top-{args.topk} worst by approx(scale_to_gt_mult_err)")
            md_lines.append("")
            md_lines.append("| rank | sequence/frame | mult_err | ratio | pose_abs_m | depth_rel | gt_valid_ratio |")
            md_lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
            for i, v in enumerate(worst[: args.topk], 1):
                md_lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(i),
                            f"{v['sequence']}/{v['frame']}",
                            f"{float(v['scale_to_gt_mult_err_approx']):.3f}" if v.get("scale_to_gt_mult_err_approx") is not None else "nan",
                            f"{float(v['scale_to_gt_ratio_approx']):.3f}" if v.get("scale_to_gt_ratio_approx") is not None else "nan",
                            f"{float(v['pose_abs_m']):.3f}" if v.get("pose_abs_m") is not None else "nan",
                            f"{float(v['depth_rel']):.3f}" if v.get("depth_rel") is not None else "nan",
                            f"{float(v['gt_valid_ratio']):.4f}" if v.get("gt_valid_ratio") is not None else "nan",
                        ]
                    )
                    + " |"
                )
            md_lines.append("")

        md_lines.append(f"- per-frame CSV: `{out_csv.relative_to(REPO_ROOT)}`")
        md_lines.append("")

    out_md = args.out_dir / "report.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
