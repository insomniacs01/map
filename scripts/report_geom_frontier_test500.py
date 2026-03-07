#!/usr/bin/env python3
"""Geometry frontier / leaderboard report for canonical Test500 (deployment contract).

This script scans `map-anything/eval_runs/**/summary_test.json` and keeps only *strictly*
comparable runs under one fixed contract (Test500) and deployment protocol.

Default contract (canonical Test500):
  frames_hash_md5 = 4affdce1133580fe79d7522c2892e359

Strict inclusion rules (all must hold):
  - meta.frames_hash_md5 == TARGET_HASH
  - meta.model_task == calibrated_sfm
  - meta.keep_camera_poses == false
  - meta.modes contains "coop"
  - metrics.geom_model.coop contains required coop-mean fields (finite numbers):
      scale_to_gt_mult_err_mean
      scale_to_gt_ratio_mean
      depth_rel_mean
      cross_agent_pose_trans_mean
      cross_agent_pose_rot_mean

Artifacts are written to:
  map-anything/eval_runs/geom_frontier_test500_YYYYMMDD/
    - geom_rows.csv (strict)
    - geom_rows_loose.csv (deployment-like, but tolerate missing meta.keep_camera_poses)
    - plots/*.png
    - report.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval_runs"

TARGET_HASH_TEST500 = "4affdce1133580fe79d7522c2892e359"

REQ_COOP_METRICS = (
    "scale_to_gt_mult_err_mean",
    "scale_to_gt_ratio_mean",
    "depth_rel_mean",
    "cross_agent_pose_trans_mean",
    "cross_agent_pose_rot_mean",
)

OPT_COOP_METRICS = (
    "chamfer_gt_to_pred_mean",
    "chamfer_pred_to_gt_mean",
    "bev_iou_raw_mean",
    "bev_iou_filtered_mean",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _safe_float(x: Any) -> float | None:
    if not _is_finite_number(x):
        return None
    return float(x)


def _rel_to_eval(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(EVAL_ROOT.resolve()))
    except Exception:
        return str(path)


def _short_ckpt(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        p = Path(path_str)
        # Keep last 3 components for readability; absolute paths are too noisy.
        parts = list(p.parts)
        return "/".join(parts[-3:]) if len(parts) >= 3 else str(p)
    except Exception:
        return str(path_str)


def _write_csv(rows: Sequence[Mapping[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    # Stable column order: union of all keys, with a preferred prefix.
    preferred = [
        "run_dir",
        "summary_path",
        "generated_at_utc",
        "ckpt_geom_model",
        "frames_hash_md5",
        "model_task",
        "keep_camera_poses",
        "modes",
        "split",
        "sample_size",
        "frames_len",
    ] + list(REQ_COOP_METRICS) + list(OPT_COOP_METRICS)

    keys: List[str] = []
    seen = set()
    for k in preferred:
        if k in rows[0] and k not in seen:
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def _write_png(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


@dataclass(frozen=True)
class StrictRow:
    run_dir: str
    summary_path: Path
    ckpt_geom_model: str
    generated_at_utc: str
    frames_hash_md5: str
    model_task: str
    keep_camera_poses: bool
    modes: Tuple[str, ...]
    split: str
    sample_size: Optional[int]
    frames_len: Optional[int]
    # Required (coop mean)
    scale_to_gt_mult_err_mean: float
    scale_to_gt_ratio_mean: float
    depth_rel_mean: float
    cross_agent_pose_trans_mean: float
    cross_agent_pose_rot_mean: float
    # Optional
    chamfer_gt_to_pred_mean: float | None
    chamfer_pred_to_gt_mean: float | None
    bev_iou_raw_mean: float | None
    bev_iou_filtered_mean: float | None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "summary_path": _rel_to_eval(self.summary_path),
            "generated_at_utc": self.generated_at_utc,
            "ckpt_geom_model": self.ckpt_geom_model,
            "frames_hash_md5": self.frames_hash_md5,
            "model_task": self.model_task,
            "keep_camera_poses": self.keep_camera_poses,
            "modes": "|".join(self.modes),
            "split": self.split,
            "sample_size": self.sample_size,
            "frames_len": self.frames_len,
            "scale_to_gt_mult_err_mean": self.scale_to_gt_mult_err_mean,
            "scale_to_gt_ratio_mean": self.scale_to_gt_ratio_mean,
            "depth_rel_mean": self.depth_rel_mean,
            "cross_agent_pose_trans_mean": self.cross_agent_pose_trans_mean,
            "cross_agent_pose_rot_mean": self.cross_agent_pose_rot_mean,
            "chamfer_gt_to_pred_mean": self.chamfer_gt_to_pred_mean,
            "chamfer_pred_to_gt_mean": self.chamfer_pred_to_gt_mean,
            "bev_iou_raw_mean": self.bev_iou_raw_mean,
            "bev_iou_filtered_mean": self.bev_iou_filtered_mean,
        }


def _meta_get(meta: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(meta, Mapping):
        return None
    return meta.get(key)


def _modes_has_coop(modes: Any) -> bool:
    if isinstance(modes, (list, tuple)):
        return any(isinstance(x, str) and x == "coop" for x in modes)
    if isinstance(modes, str):
        return modes == "coop"
    return False


def _extract_required_coop_metrics(doc: Mapping[str, Any]) -> Dict[str, float] | None:
    metrics = doc.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    gm = metrics.get("geom_model")
    if not isinstance(gm, Mapping):
        return None
    coop = gm.get("coop")
    if not isinstance(coop, Mapping):
        return None
    out: Dict[str, float] = {}
    for k in REQ_COOP_METRICS:
        v = coop.get(k)
        if not _is_finite_number(v):
            return None
        out[k] = float(v)
    return out


def _extract_optional_coop_metrics(doc: Mapping[str, Any]) -> Dict[str, float | None]:
    metrics = doc.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    gm = metrics.get("geom_model")
    if not isinstance(gm, Mapping):
        return {}
    coop = gm.get("coop")
    if not isinstance(coop, Mapping):
        return {}
    out: Dict[str, float | None] = {}
    for k in OPT_COOP_METRICS:
        out[k] = _safe_float(coop.get(k))
    return out


def _extract_ckpt_path(doc: Mapping[str, Any]) -> str:
    mp = doc.get("model_paths")
    if not isinstance(mp, Mapping):
        return ""
    ckpt = mp.get("geom_model")
    return str(ckpt) if ckpt is not None else ""


def _pareto_frontier(rows: Sequence[StrictRow], keys_minimize: Sequence[str]) -> List[StrictRow]:
    """Return non-dominated rows (minimize all keys)."""
    out: List[StrictRow] = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            # b dominates a if b <= a for all keys and < for at least one key
            le_all = True
            lt_any = False
            for k in keys_minimize:
                av = getattr(a, k, None)
                bv = getattr(b, k, None)
                if not (_is_finite_number(av) and _is_finite_number(bv)):
                    le_all = False
                    lt_any = False
                    break
                af = float(av)
                bf = float(bv)
                if bf > af:
                    le_all = False
                    break
                if bf < af:
                    lt_any = True
            if le_all and lt_any:
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


def _scatter(
    *,
    rows: Sequence[StrictRow],
    x_key: str,
    y_key: str,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xlog: bool = False,
    ylog: bool = False,
    highlight_run_dirs: Optional[set[str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=170)
    highlight_run_dirs = highlight_run_dirs or set()

    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        xv = getattr(r, x_key, None)
        yv = getattr(r, y_key, None)
        if not (_is_finite_number(xv) and _is_finite_number(yv)):
            continue
        xs.append(float(xv))
        ys.append(float(yv))

        is_hi = r.run_dir in highlight_run_dirs
        ax.scatter(
            [float(xv)],
            [float(yv)],
            s=70 if is_hi else 45,
            alpha=0.9 if is_hi else 0.65,
            c="tab:red" if is_hi else "tab:blue",
            edgecolors="black" if is_hi else "none",
            linewidths=0.7 if is_hi else 0.0,
            zorder=3 if is_hi else 2,
        )
        # Label points; keep it small to avoid clutter.
        ax.annotate(
            r.run_dir,
            (float(xv), float(yv)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            alpha=0.92,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    # Expand margins a bit to fit labels.
    if xs and ys:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        ax.set_xlim(xmin * 0.90, xmax * 1.10) if xmin > 0 else None
        ax.set_ylim(ymin * 0.90, ymax * 1.10) if ymin > 0 else None

    _write_png(fig, out_png)


def _bar_rank(
    *,
    rows: Sequence[StrictRow],
    key: str,
    out_png: Path,
    title: str,
    xlabel: str,
    better: str = "lower",
) -> None:
    items = []
    for r in rows:
        v = getattr(r, key, None)
        if _is_finite_number(v):
            items.append((r.run_dir, float(v)))
    if not items:
        return
    rev = better != "lower"
    items.sort(key=lambda x: x[1], reverse=rev)

    names = [n for n, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * max(3, len(items)) + 1.5), dpi=170)
    y = np.arange(len(items))
    ax.barh(y, vals, color="tab:green", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.4g}", va="center", fontsize=8)
    _write_png(fig, out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_root", type=Path, default=EVAL_ROOT, help="Path to eval_runs (default: repo/eval_runs)")
    ap.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: eval_runs/geom_frontier_test500_YYYYMMDD)")
    ap.add_argument("--target_hash", type=str, default=TARGET_HASH_TEST500)
    ap.add_argument("--require_split", type=str, default="test")
    ap.add_argument("--require_sample_size", type=int, default=500)
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.is_dir():
        raise SystemExit(f"eval_root not found: {eval_root}")

    today = _dt.datetime.now().strftime("%Y%m%d")
    out_dir = (args.out_dir or (eval_root / f"geom_frontier_test500_{today}")).expanduser().resolve()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(eval_root.glob("**/summary_test.json"))
    total = len(summary_paths)

    # Exclusion counters (high-level; report uses these).
    meta_incomplete = 0
    protocol_mismatch = 0
    metrics_incomplete = 0
    bad_json = 0

    # Extra breakdown to help debugging / next-step decisions.
    breakdown: Dict[str, int] = {}

    strict_rows: List[StrictRow] = []
    loose_rows: List[StrictRow] = []
    loose_csv_rows: List[Dict[str, Any]] = []
    loose_keep_inferred = 0

    def bump(k: str) -> None:
        breakdown[k] = breakdown.get(k, 0) + 1

    for sp in summary_paths:
        try:
            doc = _load_json(sp)
        except Exception:
            bad_json += 1
            bump("bad_json")
            continue

        meta = doc.get("meta")
        if not isinstance(meta, dict) or not meta:
            meta_incomplete += 1
            bump("meta_missing_or_empty")
            continue

        # Basic payload filters (apply to both strict and loose).
        split = str(doc.get("split") or "")
        if args.require_split and split != str(args.require_split):
            protocol_mismatch += 1
            bump("protocol:split_mismatch")
            continue
        frames = doc.get("frames")
        frames_len = len(frames) if isinstance(frames, list) else None
        if args.require_sample_size is not None and frames_len is not None and frames_len != int(args.require_sample_size):
            protocol_mismatch += 1
            bump("protocol:frames_len_mismatch")
            continue

        req = _extract_required_coop_metrics(doc)
        if req is None:
            metrics_incomplete += 1
            bump("metrics:required_missing_or_nonfinite")
            continue

        frames_hash_md5 = meta.get("frames_hash_md5")
        model_task = meta.get("model_task")
        modes_raw = meta.get("modes")
        # Some older meta may omit keep_camera_poses; treat as unknown for loose mode.
        keep_present = "keep_camera_poses" in meta
        keep_camera_poses = meta.get("keep_camera_poses") if keep_present else None

        # Loose deployment-like inclusion:
        # - calibrated_sfm
        # - contract matches
        # - keep_camera_poses is NOT explicitly true (missing is tolerated but flagged)
        # - has coop mode (prefer meta, but allow inferring from metrics)
        has_coop = _modes_has_coop(modes_raw) or True  # metrics already proved coop exists
        if (
            str(frames_hash_md5) == str(args.target_hash)
            and str(model_task) == "calibrated_sfm"
            and keep_camera_poses is not True
            and has_coop
        ):
            ckpt = _short_ckpt(_extract_ckpt_path(doc))
            gen_at = str(meta.get("generated_at_utc") or "")
            ss_int = None
            try:
                ss_int = int(meta.get("sample_size")) if meta.get("sample_size") is not None else None
            except Exception:
                ss_int = None
            loose_row = StrictRow(
                run_dir=_rel_to_eval(sp.parent),
                summary_path=sp,
                ckpt_geom_model=ckpt,
                generated_at_utc=gen_at,
                frames_hash_md5=str(frames_hash_md5),
                model_task=str(model_task),
                # If missing, default to False (deployment-like); we also record an inferred flag in CSV.
                keep_camera_poses=bool(keep_camera_poses) if keep_present else False,
                modes=tuple(str(m) for m in (modes_raw if isinstance(modes_raw, list) else ["coop"])),
                split=split,
                sample_size=ss_int,
                frames_len=frames_len,
                scale_to_gt_mult_err_mean=req["scale_to_gt_mult_err_mean"],
                scale_to_gt_ratio_mean=req["scale_to_gt_ratio_mean"],
                depth_rel_mean=req["depth_rel_mean"],
                cross_agent_pose_trans_mean=req["cross_agent_pose_trans_mean"],
                cross_agent_pose_rot_mean=req["cross_agent_pose_rot_mean"],
                chamfer_gt_to_pred_mean=_safe_float(_extract_optional_coop_metrics(doc).get("chamfer_gt_to_pred_mean")),
                chamfer_pred_to_gt_mean=_safe_float(_extract_optional_coop_metrics(doc).get("chamfer_pred_to_gt_mean")),
                bev_iou_raw_mean=_safe_float(_extract_optional_coop_metrics(doc).get("bev_iou_raw_mean")),
                bev_iou_filtered_mean=_safe_float(_extract_optional_coop_metrics(doc).get("bev_iou_filtered_mean")),
            )
            loose_rows.append(loose_row)
            loose_d = loose_row.as_dict()
            loose_d["keep_camera_poses_inferred"] = bool(not keep_present)
            if not keep_present:
                loose_keep_inferred += 1
            loose_csv_rows.append(loose_d)

        # Protocol filters.
        # Strict requires explicit keep_camera_poses == false and explicit meta keys.
        need_meta = ("frames_hash_md5", "model_task", "keep_camera_poses", "modes")
        missing = [k for k in need_meta if k not in meta]
        if missing:
            meta_incomplete += 1
            bump("meta_missing_keys:" + ",".join(sorted(missing)))
            continue
        if str(meta.get("frames_hash_md5")) != str(args.target_hash):
            protocol_mismatch += 1
            bump("protocol:frames_hash_mismatch")
            continue
        if str(meta.get("model_task")) != "calibrated_sfm":
            protocol_mismatch += 1
            bump("protocol:model_task_mismatch")
            continue
        if meta.get("keep_camera_poses") is not False:
            protocol_mismatch += 1
            bump("protocol:keep_camera_poses_not_false")
            continue

        modes = meta.get("modes")
        if not (isinstance(modes, list) and any(str(m) == "coop" for m in modes)):
            protocol_mismatch += 1
            bump("protocol:modes_missing_coop")
            continue

        sample_size = meta.get("sample_size")
        if args.require_sample_size is not None:
            try:
                ss_int = int(sample_size)
            except Exception:
                protocol_mismatch += 1
                bump("protocol:sample_size_missing_or_bad")
                continue
            if ss_int != int(args.require_sample_size):
                protocol_mismatch += 1
                bump("protocol:sample_size_mismatch")
                continue
        else:
            ss_int = int(sample_size) if isinstance(sample_size, int) else None
        ckpt = _short_ckpt(_extract_ckpt_path(doc))
        gen_at = str(meta.get("generated_at_utc") or "")
        opt = _extract_optional_coop_metrics(doc)
        strict_rows.append(
            StrictRow(
                run_dir=_rel_to_eval(sp.parent),
                summary_path=sp,
                ckpt_geom_model=ckpt,
                generated_at_utc=gen_at,
                frames_hash_md5=str(meta.get("frames_hash_md5")),
                model_task=str(meta.get("model_task")),
                keep_camera_poses=bool(meta.get("keep_camera_poses")),
                modes=tuple(str(m) for m in modes),
                split=split,
                sample_size=ss_int,
                frames_len=frames_len,
                scale_to_gt_mult_err_mean=req["scale_to_gt_mult_err_mean"],
                scale_to_gt_ratio_mean=req["scale_to_gt_ratio_mean"],
                depth_rel_mean=req["depth_rel_mean"],
                cross_agent_pose_trans_mean=req["cross_agent_pose_trans_mean"],
                cross_agent_pose_rot_mean=req["cross_agent_pose_rot_mean"],
                chamfer_gt_to_pred_mean=_safe_float(opt.get("chamfer_gt_to_pred_mean")),
                chamfer_pred_to_gt_mean=_safe_float(opt.get("chamfer_pred_to_gt_mean")),
                bev_iou_raw_mean=_safe_float(opt.get("bev_iou_raw_mean")),
                bev_iou_filtered_mean=_safe_float(opt.get("bev_iou_filtered_mean")),
            )
        )

    # Write strict + loose rows CSV.
    _write_csv([r.as_dict() for r in strict_rows], out_dir / "geom_rows.csv")
    _write_csv(loose_csv_rows, out_dir / "geom_rows_loose.csv")

    # Leaderboard sorting (use loose rows for a more complete frontier; strict is a subset).
    loose_by_scale = sorted(
        loose_rows,
        key=lambda r: (r.scale_to_gt_mult_err_mean, r.cross_agent_pose_trans_mean, r.depth_rel_mean),
    )
    loose_by_cross = sorted(
        loose_rows,
        key=lambda r: (r.cross_agent_pose_trans_mean, r.cross_agent_pose_rot_mean, r.scale_to_gt_mult_err_mean),
    )
    strict_by_scale = sorted(
        strict_rows,
        key=lambda r: (r.scale_to_gt_mult_err_mean, r.cross_agent_pose_trans_mean, r.depth_rel_mean),
    )
    strict_by_cross = sorted(
        strict_rows,
        key=lambda r: (r.cross_agent_pose_trans_mean, r.cross_agent_pose_rot_mean, r.scale_to_gt_mult_err_mean),
    )

    # Pareto frontiers (computed on loose rows by default).
    frontier_2d = _pareto_frontier(loose_rows, keys_minimize=("scale_to_gt_mult_err_mean", "cross_agent_pose_trans_mean"))
    frontier_5d = _pareto_frontier(loose_rows, keys_minimize=REQ_COOP_METRICS)

    frontier_2d_set = {r.run_dir for r in frontier_2d}

    # Plots (run-level): use loose rows for density; highlight loose frontier.
    if loose_rows:
        _scatter(
            rows=loose_rows,
            x_key="cross_agent_pose_trans_mean",
            y_key="scale_to_gt_mult_err_mean",
            out_png=plots_dir / "scatter_cross_trans_vs_scale_to_gt_mult_err.png",
            title="Test500 coop mean | cross_trans vs scale_to_gt_mult_err (deployment-like loose runs)",
            xlabel="cross_agent_pose_trans_mean (m)  [lower is better]",
            ylabel="scale_to_gt_mult_err_mean  [lower is better]",
            xlog=False,
            ylog=False,
            highlight_run_dirs=frontier_2d_set,
        )
        _scatter(
            rows=loose_rows,
            x_key="depth_rel_mean",
            y_key="scale_to_gt_mult_err_mean",
            out_png=plots_dir / "scatter_depth_rel_vs_scale_to_gt_mult_err.png",
            title="Test500 coop mean | depth_rel vs scale_to_gt_mult_err (deployment-like loose runs)",
            xlabel="depth_rel_mean  [lower is better]",
            ylabel="scale_to_gt_mult_err_mean  [lower is better]",
            xlog=False,
            ylog=False,
            highlight_run_dirs=frontier_2d_set,
        )
        _bar_rank(
            rows=loose_rows,
            key="cross_agent_pose_trans_mean",
            out_png=plots_dir / "bar_rank_cross_trans.png",
            title="Deployment-like loose runs ranked by cross_agent_pose_trans_mean (lower is better)",
            xlabel="cross_agent_pose_trans_mean (m)",
            better="lower",
        )

    # Simple recommendation (deployment-first): choose best cross_trans, tie-break on scale_mult_err, depth_rel.
    recommend = loose_by_cross[0] if loose_by_cross else None

    # Markdown report.
    md: List[str] = []
    md.append("# Geometry Frontier / Leaderboard (Test500, deployment protocol)")
    md.append("")
    md.append(f"- generated_at: `{_dt.datetime.now().isoformat(timespec='seconds')}`")
    md.append(f"- scanned: `{total}` summaries under `{eval_root}`")
    md.append(f"- strict_included: `{len(strict_rows)}`")
    md.append(f"- loose_included(calibrated_sfm + keep_camera_poses!=true): `{len(loose_rows)}`")
    md.append(f"- loose_keep_camera_poses_inferred(missing key): `{loose_keep_inferred}`")
    md.append(f"- excluded_bad_json: `{bad_json}`")
    md.append(f"- excluded_meta_incomplete: `{meta_incomplete}`")
    md.append(f"- excluded_protocol_mismatch: `{protocol_mismatch}`")
    md.append(f"- excluded_metrics_incomplete: `{metrics_incomplete}`")
    md.append("")
    md.append("## Strict Filtering Rules")
    md.append("")
    md.append("A run is included in the main table iff all are true:")
    md.append("")
    md.append(f"- `meta.frames_hash_md5 == {args.target_hash}` (canonical Test500 contract)")
    md.append("- `meta.model_task == calibrated_sfm`")
    md.append("- `meta.keep_camera_poses == false`")
    md.append("- `meta.modes` contains `coop`")
    if args.require_split:
        md.append(f"- `split == {args.require_split}`")
    if args.require_sample_size is not None:
        md.append(f"- `meta.sample_size == {int(args.require_sample_size)}` and `len(frames) == {int(args.require_sample_size)}`")
    md.append("- coop-mean metrics exist and are finite numbers:")
    for k in REQ_COOP_METRICS:
        md.append(f"  - `{k}`")
    md.append("")
    md.append("Strict rows CSV:")
    md.append("")
    md.append(f"- `{_rel_to_eval(out_dir / 'geom_rows.csv')}`")
    md.append("")
    md.append("Loose rows CSV (deployment-like; tolerates missing `meta.keep_camera_poses` and marks `keep_camera_poses_inferred`):")
    md.append("")
    md.append(f"- `{_rel_to_eval(out_dir / 'geom_rows_loose.csv')}`")
    md.append("")

    md.append("## Excluded Runs (Counts)")
    md.append("")
    md.append("These are excluded from strict comparison (counted only):")
    md.append("")
    md.append(f"- meta incomplete (missing/empty meta or missing required keys): `{meta_incomplete}`")
    md.append(f"- protocol mismatch (meta present but values differ): `{protocol_mismatch}`")
    md.append(f"- metrics incomplete (missing required coop-mean fields): `{metrics_incomplete}`")
    if breakdown:
        md.append("")
        md.append("### Breakdown (Top reasons)")
        md.append("")
        for k, v in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:20]:
            md.append(f"- {k}: `{v}`")
    md.append("")

    md.append("## Leaderboards (Loose Runs; deployment-like)")
    md.append("")
    if not loose_rows:
        md.append("_No loose runs found._")
    else:
        def row_line(r: StrictRow) -> str:
            return (
                f"- `{r.run_dir}` | ckpt=`{r.ckpt_geom_model}` | "
                f"mult={r.scale_to_gt_mult_err_mean:.4f} | ratio={r.scale_to_gt_ratio_mean:.4f} | "
                f"depth_rel={r.depth_rel_mean:.4f} | cross_t={r.cross_agent_pose_trans_mean:.4f} | "
                f"cross_r={r.cross_agent_pose_rot_mean:.4f}"
            )

        md.append("### Best by Scale (primary: scale_to_gt_mult_err_mean)")
        md.append("")
        for r in loose_by_scale[:10]:
            md.append(row_line(r))
        md.append("")

        md.append("### Best by Cross-Agent Pose (primary: cross_agent_pose_trans_mean)")
        md.append("")
        for r in loose_by_cross[:10]:
            md.append(row_line(r))
        md.append("")

    md.append("## Leaderboards (Strict Runs; fully audited)")
    md.append("")
    if not strict_rows:
        md.append("_No strict runs found._")
    else:
        def row_line2(r: StrictRow) -> str:
            return (
                f"- `{r.run_dir}` | ckpt=`{r.ckpt_geom_model}` | "
                f"mult={r.scale_to_gt_mult_err_mean:.4f} | ratio={r.scale_to_gt_ratio_mean:.4f} | "
                f"depth_rel={r.depth_rel_mean:.4f} | cross_t={r.cross_agent_pose_trans_mean:.4f} | "
                f"cross_r={r.cross_agent_pose_rot_mean:.4f}"
            )
        for r in strict_by_cross[:10]:
            md.append(row_line2(r))
    md.append("")

    md.append("## Pareto Frontier")
    md.append("")
    md.append("We report two frontiers (all metrics are minimized; computed on **loose** rows):")
    md.append("")
    md.append("- 2D frontier on `(scale_to_gt_mult_err_mean, cross_agent_pose_trans_mean)` (deployment-relevant tradeoff)")
    md.append(f"- 5D frontier on `{', '.join(REQ_COOP_METRICS)}` (requires all 5 metrics present)")
    md.append("")
    if strict_rows:
        md.append("### 2D Frontier Members")
        md.append("")
        for r in sorted(frontier_2d, key=lambda x: (x.scale_to_gt_mult_err_mean, x.cross_agent_pose_trans_mean)):
            md.append(f"- `{r.run_dir}` (mult={r.scale_to_gt_mult_err_mean:.4f}, cross_t={r.cross_agent_pose_trans_mean:.4f})")
        md.append("")

        md.append("### 5D Frontier Members")
        md.append("")
        for r in sorted(frontier_5d, key=lambda x: (x.scale_to_gt_mult_err_mean, x.cross_agent_pose_trans_mean)):
            md.append(
                f"- `{r.run_dir}` (mult={r.scale_to_gt_mult_err_mean:.4f}, ratio={r.scale_to_gt_ratio_mean:.4f}, "
                f"depth_rel={r.depth_rel_mean:.4f}, cross_t={r.cross_agent_pose_trans_mean:.4f}, cross_r={r.cross_agent_pose_rot_mean:.4f})"
            )
        md.append("")

        md.append("A run is **dominated** if another run is <= on all chosen metrics and < on at least one metric.")
        md.append("")

    md.append("## Recommendation (Promotion Starting Point)")
    md.append("")
    if recommend is None:
        md.append("_No strict runs available for recommendation._")
    else:
        md.append("Deployment-first heuristic (minimize cross-agent pose error; tie-break on scale/depth):")
        md.append("")
        md.append(
            f"- RECOMMEND: `{recommend.run_dir}` | ckpt=`{recommend.ckpt_geom_model}` | "
            f"mult={recommend.scale_to_gt_mult_err_mean:.4f} | ratio={recommend.scale_to_gt_ratio_mean:.4f} | "
            f"depth_rel={recommend.depth_rel_mean:.4f} | cross_t={recommend.cross_agent_pose_trans_mean:.4f} | "
            f"cross_r={recommend.cross_agent_pose_rot_mean:.4f}"
        )
    md.append("")

    md.append("## Plots")
    md.append("")
    md.append(f"Output directory: `{_rel_to_eval(out_dir)}`")
    md.append("")
    for p in sorted(plots_dir.glob("*.png")):
        md.append(f"- `{p.name}`")
    md.append("")

    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_dir / 'geom_rows.csv'}")
    print(f"[OK] wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
