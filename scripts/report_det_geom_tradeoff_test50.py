#!/usr/bin/env python3
"""Report: geometry completion vs det/e2e AP on the canonical pinhole Test50 contract.

Goal
----
You want to answer:
  - "How good can geometry get?" (proxy metrics available in det summaries)
  - "Under different geometry quality levels, how does coop perception (det AP) change?"

This script is **read-only**: it scans existing `eval_runs/**/summary_test.json` and
builds a unified table and a small report with plots.

Fairness / comparability
------------------------
Strict det fairness requires the following invariants to match:
  - same frames contract (Test50)
  - same det_head_cfg
  - same det_decode_cfg
  - same model_task / keep_camera_poses
  - same eval script version (eval_script_md5)

Unfortunately, many historical summaries do not contain full `meta` or `run_info`.
We therefore export *all* rows but explicitly label each row as:
  - strict_fair (all fields present)
  - partial_fair (some key fields present)
  - legacy_unknown (missing most protocol fields)

Outputs
-------
Writes to: `map-anything/eval_runs/geom_det_tradeoff_test50_YYYYMMDD/`
  - det_geom_rows.csv            (all rows: one per summary/model/mode)
  - det_geom_best_by_ap.csv      (per run_dir/mode: best det_ap_iou row)
  - plots/*.png                  (scatter plots)
  - report.md                    (how to read + key findings)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frames_hash_md5(frames: Sequence[Mapping[str, Any]]) -> str | None:
    items: List[str] = []
    for f in frames:
        seq = f.get("sequence")
        fr = f.get("frame")
        if not isinstance(seq, str) or not isinstance(fr, str):
            return None
        items.append(f"{seq}/{fr}")
    items.sort()
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


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


def _json_compact(x: Any) -> str:
    try:
        return json.dumps(x, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(x)


def _write_csv(rows: Sequence[Mapping[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    # Stable-ish columns: prefer a known list then union.
    preferred = [
        "run_dir",
        "summary_path",
        "model_key",
        "mode",
        "frames_hash_md5",
        "sample_size",
        "has_meta",
        "has_run_info",
        "model_task",
        "keep_camera_poses",
        "keep_main_agent_poses",
        "det_head_cfg",
        "det_head_ckpt",
        "det_decode_cfg",
        "eval_script_md5",
        "protocol_group",
        "fairness_tier",
        # det metrics
        "det_ap_iou",
        "det_precision_iou",
        "det_recall_iou",
        "det_mean_iou",
        # geometry-ish metrics (from same summary)
        "pose_abs_mean",
        "pose_rot_mean",
        "cross_agent_pose_trans_mean",
        "cross_agent_pose_rot_mean",
        "depth_rel_mean",
        "scale_to_gt_mult_err_mean",
        "scale_to_gt_ratio_mean",
        # legacy scale (ratio-to-1) for backward compatibility
        "scale_err_mean",
        "scale_ratio_mean",
    ]

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


def _fairness_tier(*, has_meta: bool, has_run_info: bool, model_task: Any, keep_camera_poses: Any, det_head_cfg: Any, det_decode_cfg: Any, eval_script_md5: Any) -> str:
    # Strict: all key protocol fields exist.
    if (
        has_meta
        and model_task is not None
        and keep_camera_poses is not None
        and det_head_cfg is not None
        and isinstance(det_decode_cfg, dict)
        and eval_script_md5 is not None
    ):
        return "strict_fair"

    # Partial: we at least know det_decode_cfg (score/iou) from run_info backfill.
    if isinstance(det_decode_cfg, dict) and (has_meta or has_run_info):
        return "partial_fair"

    return "legacy_unknown"


def _stage_from_scale_mult(mult: float | None) -> str:
    # Align with docs/master_plan.md thresholds (coarse).
    if mult is None or not math.isfinite(mult):
        return "S_unknown"
    if mult <= 1.3:
        return "S3_<=1.3"
    if mult <= 1.6:
        return "S2_<=1.6"
    if mult <= 2.0:
        return "S1_<=2.0"
    return "S0_>2.0"


def _stage_from_pose_abs(pose_abs: float | None) -> str:
    if pose_abs is None or not math.isfinite(pose_abs):
        return "P_unknown"
    if pose_abs <= 1.0:
        return "P3_<=1m"
    if pose_abs <= 5.0:
        return "P2_<=5m"
    if pose_abs <= 10.0:
        return "P1_<=10m"
    return "P0_>10m"


def _scatter_xy(
    *,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    ax.scatter(xs, ys, s=22, alpha=0.55)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotate a few extreme points for readability (top AP + worst x).
    if xs and ys:
        idx_top = int(np.argmax(np.asarray(ys, dtype=np.float64)))
        idx_worst = int(np.argmax(np.asarray(xs, dtype=np.float64)))
        for idx, color in [(idx_top, "tab:green"), (idx_worst, "tab:red")]:
            ax.scatter([xs[idx]], [ys[idx]], s=55, alpha=0.9, c=color)
            ax.text(xs[idx], ys[idx], labels[idx], fontsize=7, alpha=0.85)

    _write_png(fig, out_png)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--canonical_frames_json",
        type=Path,
        default=EVAL_ROOT / "frames_test50_det_e2e_v5.json",
        help="Canonical Test50 frames contract JSON.",
    )
    ap.add_argument("--root", type=Path, default=EVAL_ROOT, help="Root to scan for summary_test.json")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    canonical = _load_json(args.canonical_frames_json.expanduser().resolve())
    can_frames = canonical.get("frames") or []
    if not isinstance(can_frames, list) or not can_frames:
        print(f"[ERR] canonical frames json has no frames: {args.canonical_frames_json}", file=sys.stderr)
        return 2
    can_hash = _frames_hash_md5(can_frames)
    if not can_hash:
        print(f"[ERR] failed to hash canonical frames: {args.canonical_frames_json}", file=sys.stderr)
        return 2

    date_tag = _dt.datetime.utcnow().strftime("%Y%m%d")
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else (EVAL_ROOT / f"geom_det_tradeoff_test50_{date_tag}")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    scanned = 0
    matched = 0
    bad_json = 0

    for summary_path in sorted(args.root.expanduser().resolve().rglob("summary_test.json")):
        scanned += 1
        try:
            d = _load_json(summary_path)
        except Exception:
            bad_json += 1
            continue
        if d.get("split") != "test":
            continue
        frames = d.get("frames") or []
        if not isinstance(frames, list) or len(frames) != len(can_frames):
            continue
        fr_hash = _frames_hash_md5(frames)
        if fr_hash != can_hash:
            continue

        metrics = d.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue

        meta = d.get("meta")
        run_info = d.get("run_info")
        has_meta = isinstance(meta, dict)
        has_run_info = isinstance(run_info, dict)

        # Best-effort protocol fields.
        model_task = meta.get("model_task") if has_meta else None
        keep_camera_poses = meta.get("keep_camera_poses") if has_meta else None
        keep_main_agent_poses = meta.get("keep_main_agent_poses") if has_meta else None
        det_head_cfg = meta.get("det_head_cfg") if has_meta else None
        det_head_ckpt = meta.get("det_head_ckpt") if has_meta else None
        det_decode_cfg = meta.get("det_decode_cfg") if has_meta else None
        eval_script_md5 = meta.get("eval_script_md5") if has_meta else None
        sample_size = meta.get("sample_size") if has_meta else (run_info.get("sample_size") if has_run_info else None)

        if det_decode_cfg is None and has_run_info:
            det_decode_cfg = run_info.get("det_decode_cfg")

        # Extract rows: any model_key/mode with det_ap_iou.
        any_det = False
        for model_key, per_mode in metrics.items():
            if not isinstance(per_mode, dict):
                continue
            for mode, m in per_mode.items():
                if not isinstance(m, dict):
                    continue
                ap_iou = _safe_float(m.get("det_ap_iou"))
                if ap_iou is None:
                    continue
                any_det = True
                row: Dict[str, Any] = {
                    "run_dir": summary_path.parent.name,
                    "summary_path": _rel_to_eval(summary_path),
                    "model_key": str(model_key),
                    "mode": str(mode),
                    "frames_hash_md5": fr_hash,
                    "sample_size": sample_size,
                    "has_meta": bool(has_meta),
                    "has_run_info": bool(has_run_info),
                    "model_task": model_task,
                    "keep_camera_poses": keep_camera_poses,
                    "keep_main_agent_poses": keep_main_agent_poses,
                    "det_head_cfg": det_head_cfg,
                    "det_head_ckpt": det_head_ckpt,
                    "det_decode_cfg": _json_compact(det_decode_cfg) if det_decode_cfg is not None else None,
                    "eval_script_md5": eval_script_md5,
                    "det_ap_iou": ap_iou,
                    "det_precision_iou": _safe_float(m.get("det_precision_iou")),
                    "det_recall_iou": _safe_float(m.get("det_recall_iou")),
                    "det_mean_iou": _safe_float(m.get("det_mean_iou")),
                    # geometry-ish
                    "pose_abs_mean": _safe_float(m.get("pose_abs_mean")),
                    "pose_rot_mean": _safe_float(m.get("pose_rot_mean")),
                    "cross_agent_pose_trans_mean": _safe_float(m.get("cross_agent_pose_trans_mean")),
                    "cross_agent_pose_rot_mean": _safe_float(m.get("cross_agent_pose_rot_mean")),
                    "depth_rel_mean": _safe_float(m.get("depth_rel_mean")),
                    "scale_to_gt_mult_err_mean": _safe_float(m.get("scale_to_gt_mult_err_mean")),
                    "scale_to_gt_ratio_mean": _safe_float(m.get("scale_to_gt_ratio_mean")),
                    "scale_err_mean": _safe_float(m.get("scale_err_mean")),
                    "scale_ratio_mean": _safe_float(m.get("scale_ratio_mean")),
                }
                row["fairness_tier"] = _fairness_tier(
                    has_meta=bool(has_meta),
                    has_run_info=bool(has_run_info),
                    model_task=model_task,
                    keep_camera_poses=keep_camera_poses,
                    det_head_cfg=det_head_cfg,
                    det_decode_cfg=det_decode_cfg,
                    eval_script_md5=eval_script_md5,
                )
                # Protocol fingerprint for grouping (only meaningful when strict_fair).
                row["protocol_group"] = "|".join(
                    [
                        f"task={model_task}",
                        f"keep_cam={keep_camera_poses}",
                        f"keep_main={keep_main_agent_poses}",
                        f"head={det_head_cfg}",
                        f"head_ckpt={det_head_ckpt}",
                        f"decode={row.get('det_decode_cfg')}",
                        f"eval={eval_script_md5}",
                    ]
                )
                row["stage_scale"] = _stage_from_scale_mult(row["scale_to_gt_mult_err_mean"])
                row["stage_pose"] = _stage_from_pose_abs(row["pose_abs_mean"])
                rows.append(row)

        if any_det:
            matched += 1

    # All rows.
    all_csv = out_dir / "det_geom_rows.csv"
    _write_csv(rows, all_csv)

    # Best-by-AP per (run_dir, mode).
    best_rows: List[Dict[str, Any]] = []
    key_to_best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        k = (str(r.get("run_dir", "")), str(r.get("mode", "")))
        cur = key_to_best.get(k)
        if cur is None or float(r.get("det_ap_iou") or -1.0) > float(cur.get("det_ap_iou") or -1.0):
            key_to_best[k] = r
    for k in sorted(key_to_best.keys()):
        best_rows.append(key_to_best[k])

    best_csv = out_dir / "det_geom_best_by_ap.csv"
    _write_csv(best_rows, best_csv)

    # Plots (use best rows only to avoid double-counting multi-model summaries).
    def make_scatter(*, mode: str, x_key: str, y_key: str, out_name: str, xlabel: str) -> None:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []
        for r in best_rows:
            if r.get("mode") != mode:
                continue
            xv = r.get(x_key)
            yv = r.get(y_key)
            if xv is None or yv is None:
                continue
            if not (_is_finite_number(xv) and _is_finite_number(yv)):
                continue
            xs.append(float(xv))
            ys.append(float(yv))
            labels.append(f"{r.get('run_dir')}:{r.get('model_key')}")
        if not xs:
            return
        _scatter_xy(
            xs=xs,
            ys=ys,
            labels=labels,
            out_png=plots_dir / out_name,
            title=f"{mode}: {y_key} vs {x_key} (best-by-AP per run_dir)",
            xlabel=xlabel,
            ylabel=y_key,
        )

    make_scatter(
        mode="coop",
        x_key="scale_to_gt_mult_err_mean",
        y_key="det_ap_iou",
        out_name="scatter_coop_ap_vs_scale_to_gt_mult_err.png",
        xlabel="scale_to_gt_mult_err_mean (lower is better; 1 is ideal)",
    )
    make_scatter(
        mode="coop",
        x_key="depth_rel_mean",
        y_key="det_ap_iou",
        out_name="scatter_coop_ap_vs_depth_rel.png",
        xlabel="depth_rel_mean (lower is better)",
    )
    make_scatter(
        mode="coop",
        x_key="cross_agent_pose_trans_mean",
        y_key="det_ap_iou",
        out_name="scatter_coop_ap_vs_cross_agent_pose_trans.png",
        xlabel="cross_agent_pose_trans_mean (m) (lower is better)",
    )
    make_scatter(
        mode="coop",
        x_key="pose_abs_mean",
        y_key="det_ap_iou",
        out_name="scatter_coop_ap_vs_pose_abs.png",
        xlabel="pose_abs_mean (m) (lower is better)",
    )

    make_scatter(
        mode="single",
        x_key="scale_to_gt_mult_err_mean",
        y_key="det_ap_iou",
        out_name="scatter_single_ap_vs_scale_to_gt_mult_err.png",
        xlabel="scale_to_gt_mult_err_mean (lower is better; 1 is ideal)",
    )

    # Simple stats + report.
    def _pearson(xs: List[float], ys: List[float]) -> float | None:
        if len(xs) < 3:
            return None
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        if x.std() <= 1e-12 or y.std() <= 1e-12:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    def _collect(mode: str, x_key: str) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for r in best_rows:
            if r.get("mode") != mode:
                continue
            xv = r.get(x_key)
            yv = r.get("det_ap_iou")
            if xv is None or yv is None:
                continue
            if not (_is_finite_number(xv) and _is_finite_number(yv)):
                continue
            xs.append(float(xv))
            ys.append(float(yv))
        return xs, ys

    def _topk(mode: str) -> List[Dict[str, Any]]:
        rs = [r for r in best_rows if r.get("mode") == mode and _is_finite_number(r.get("det_ap_iou"))]
        rs.sort(key=lambda r: float(r.get("det_ap_iou") or -1.0), reverse=True)
        return rs[: int(args.topk)]

    # Protocol groups (strict_fair only). This prevents mixing posed vs deployment-like runs.
    strict_rows = [r for r in best_rows if r.get("fairness_tier") == "strict_fair"]
    group_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for r in strict_rows:
        g = str(r.get("protocol_group") or "")
        if not g:
            continue
        group_to_rows.setdefault(g, []).append(r)

    lines: List[str] = []
    lines.append("# Geometry Completion vs det/e2e AP (Test50)\n")
    lines.append(f"- generated_at_utc: `{_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- canonical_frames_json: `{_rel_to_eval(args.canonical_frames_json)}`")
    lines.append(f"- canonical_frames_hash_md5: `{can_hash}`")
    lines.append(f"- scanned summaries: `{scanned}` (bad_json={bad_json})")
    lines.append(f"- matched run_dirs (same frames + has det_ap_iou): `{matched}`")
    lines.append(f"- exported rows (summary/model/mode): `{len(rows)}`")
    lines.append(f"- best-by-AP rows (run_dir/mode): `{len(best_rows)}`\n")

    # Fairness tier counts.
    from collections import Counter

    tier_counts = Counter([str(r.get('fairness_tier')) for r in best_rows])
    lines.append("## Fairness Tier Counts (best-by-AP rows)\n")
    for k, n in sorted(tier_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `strict_fair`: full meta available (det_head_cfg/det_decode_cfg/model_task/keep_camera_poses/eval_script_md5).")
    lines.append("- `partial_fair`: det_decode_cfg partly available (e.g., backfilled score/iou), but other protocol fields missing.")
    lines.append("- `legacy_unknown`: missing most protocol fields; use for exploratory plots only.\n")

    # Protocol group summary (strict_fair only).
    from collections import Counter

    def _parse_protocol_group(g: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in str(g).split("|"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            out[str(k)] = str(v)
        return out

    lines.append("## Strict-Fair Protocol Groups (Counts)\n")
    if not group_to_rows:
        lines.append("- (none)\n")
    else:
        counts = Counter({k: len(v) for k, v in group_to_rows.items()})
        # List all strict protocol groups (typically small N), so new protocols don't get hidden
        # just because they have fewer historical runs.
        for g, n in counts.most_common():
            info = _parse_protocol_group(str(g))
            head = info.get("head", "")
            task = info.get("task", "")
            keep_cam = info.get("keep_cam", "")
            keep_main = info.get("keep_main", "")
            eval_md5 = info.get("eval", "")
            eval_short = str(eval_md5)[:8] if eval_md5 else ""
            head_ckpt = info.get("head_ckpt", "")
            if head_ckpt:
                p = Path(head_ckpt)
                head_ckpt_short = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else str(p)
            else:
                head_ckpt_short = ""
            lines.append(
                f"- n={n} | task={task} keep_camera_poses={keep_cam} keep_main_agent_poses={keep_main} "
                f"det_head_cfg={head} det_head_ckpt={head_ckpt_short} eval={eval_short}"
            )
        lines.append("")

    def _collect_from(rows_in: List[Dict[str, Any]], mode: str, x_key: str) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for r in rows_in:
            if r.get("mode") != mode:
                continue
            xv = r.get(x_key)
            yv = r.get("det_ap_iou")
            if xv is None or yv is None:
                continue
            if not (_is_finite_number(xv) and _is_finite_number(yv)):
                continue
            xs.append(float(xv))
            ys.append(float(yv))
        return xs, ys

    # Correlations per strict protocol group (top groups only).
    lines.append("## Correlations (Pearson r; strict_fair; grouped by protocol)\n")
    if not group_to_rows:
        lines.append("- (none)\n")
    else:
        top_groups = sorted(group_to_rows.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]
        for g, rs in top_groups:
            info = _parse_protocol_group(str(g))
            head = info.get("head", "")
            task = info.get("task", "")
            keep_cam = info.get("keep_cam", "")
            head_ckpt = info.get("head_ckpt", "")
            if head_ckpt:
                p = Path(head_ckpt)
                head_ckpt_short = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else str(p)
            else:
                head_ckpt_short = ""
            lines.append(f"### protocol: task={task} keep_camera_poses={keep_cam} det_head_cfg={head} det_head_ckpt={head_ckpt_short}\n")
            for mode in ("coop", "single"):
                for x_key in ("scale_to_gt_mult_err_mean", "depth_rel_mean", "pose_abs_mean", "cross_agent_pose_trans_mean"):
                    xs, ys = _collect_from(rs, mode, x_key)
                    r = _pearson(xs, ys)
                    lines.append(f"- {mode}: r(det_ap_iou, {x_key}) = {('NA' if r is None else f'{r:.3f}')} (n={len(xs)})")
            lines.append("")

    # Scale-stage bucket stats (by scale stage, coop only) per strict protocol group.
    lines.append("## Scale-Stage Bucket AP Summary (coop; strict_fair; grouped by protocol)\n")
    if not group_to_rows:
        lines.append("- (none)\n")
    else:
        top_groups = sorted(group_to_rows.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]
        for g, rs in top_groups:
            info = _parse_protocol_group(str(g))
            head = info.get("head", "")
            task = info.get("task", "")
            keep_cam = info.get("keep_cam", "")
            head_ckpt = info.get("head_ckpt", "")
            if head_ckpt:
                p = Path(head_ckpt)
                head_ckpt_short = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else str(p)
            else:
                head_ckpt_short = ""
            lines.append(f"### protocol: task={task} keep_camera_poses={keep_cam} det_head_cfg={head} det_head_ckpt={head_ckpt_short}\n")
            stage_rows: Dict[str, List[float]] = {}
            for r in rs:
                if r.get("mode") != "coop":
                    continue
                st = str(r.get("stage_scale") or "S_unknown")
                ap_iou = r.get("det_ap_iou")
                if not _is_finite_number(ap_iou):
                    continue
                stage_rows.setdefault(st, []).append(float(ap_iou))
            if not stage_rows:
                lines.append("- (no coop rows)\n")
                continue
            lines.append("| stage_scale | n | ap_mean | ap_median | ap_min | ap_max |")
            lines.append("| --- | ---:| ---:| ---:| ---:| ---:|")
            for st in sorted(stage_rows.keys()):
                xs = np.asarray(stage_rows[st], dtype=np.float64)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            st,
                            str(xs.size),
                            f"{float(xs.mean()):.6f}",
                            f"{float(np.median(xs)):.6f}",
                            f"{float(xs.min()):.6f}",
                            f"{float(xs.max()):.6f}",
                        ]
                    )
                    + " |"
                )
            lines.append("")

    def _fmt_num(x: Any) -> str:
        if x is None:
            return "NA"
        if not _is_finite_number(x):
            return "NA"
        return f"{float(x):.6f}"

    lines.append(f"## Top-{args.topk} by det_ap_iou (coop; strict_fair; grouped by protocol)\n")
    if not group_to_rows:
        lines.append("- (none)\n")
    else:
        # Prefer common protocols (many points) but always include the best posed and best predpose
        # protocol groups so the report surfaces the current frontier even for new protocols.
        top_by_count = sorted(group_to_rows.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]

        def _best_group_by_coop_ap(predicate) -> tuple[str, List[Dict[str, Any]]] | None:
            best_g = None
            best_ap = -1.0
            for g, rs in group_to_rows.items():
                info = _parse_protocol_group(str(g))
                if not predicate(info):
                    continue
                aps = [
                    float(r.get("det_ap_iou"))
                    for r in rs
                    if r.get("mode") == "coop" and _is_finite_number(r.get("det_ap_iou"))
                ]
                if not aps:
                    continue
                ap = max(aps)
                if ap > best_ap:
                    best_ap = ap
                    best_g = (g, rs)
            return best_g

        best_predpose = _best_group_by_coop_ap(lambda info: info.get("task") == "calibrated_sfm" and info.get("keep_cam") == "False")
        pred_eval = None
        if best_predpose is not None:
            pred_eval = _parse_protocol_group(str(best_predpose[0])).get("eval")

        # For posed-vs-predpose comparisons, prefer a posed protocol group that shares the same
        # eval_script_md5 with the best predpose group (otherwise the AP numbers are not strictly
        # comparable across eval versions).
        best_posed = None
        if pred_eval is not None:
            best_posed = _best_group_by_coop_ap(
                lambda info: (
                    info.get("task") == "posed_sfm"
                    and info.get("keep_cam") == "True"
                    and info.get("eval") == pred_eval
                )
            )
        if best_posed is None:
            best_posed = _best_group_by_coop_ap(lambda info: info.get("task") == "posed_sfm" and info.get("keep_cam") == "True")

        top_groups = list(top_by_count)
        for cand in (best_posed, best_predpose):
            if cand is None:
                continue
            if not any(str(cand[0]) == str(g) for g, _ in top_groups):
                top_groups.append(cand)

        for g, rs in top_groups:
            info = _parse_protocol_group(str(g))
            head = info.get("head", "")
            task = info.get("task", "")
            keep_cam = info.get("keep_cam", "")
            keep_main = info.get("keep_main", "")
            eval_md5 = info.get("eval", "")
            eval_short = str(eval_md5)[:8] if eval_md5 else ""
            head_ckpt = info.get("head_ckpt", "")
            if head_ckpt:
                p = Path(head_ckpt)
                head_ckpt_short = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else str(p)
            else:
                head_ckpt_short = ""
            lines.append(
                f"### protocol: task={task} keep_camera_poses={keep_cam} keep_main_agent_poses={keep_main} "
                f"det_head_cfg={head} det_head_ckpt={head_ckpt_short} eval={eval_short}\n"
            )
            coop_rs = [r for r in rs if r.get("mode") == "coop" and _is_finite_number(r.get("det_ap_iou"))]
            coop_rs.sort(key=lambda r: float(r.get("det_ap_iou") or -1.0), reverse=True)
            coop_rs = coop_rs[: int(args.topk)]
            if not coop_rs:
                lines.append("- (no coop rows)\n")
                continue
            lines.append("| det_ap_iou | run_dir | model_key | scale_mult | scale_ratio | depth_rel | pose_abs | cross_trans | evidence |")
            lines.append("| ---:| --- | --- | ---:| ---:| ---:| ---:| ---:| --- |")
            for r in coop_rs:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _fmt_num(r.get("det_ap_iou")),
                            str(r.get("run_dir")),
                            str(r.get("model_key")),
                            _fmt_num(r.get("scale_to_gt_mult_err_mean")),
                            _fmt_num(r.get("scale_to_gt_ratio_mean")),
                            _fmt_num(r.get("depth_rel_mean")),
                            _fmt_num(r.get("pose_abs_mean")),
                            _fmt_num(r.get("cross_agent_pose_trans_mean")),
                            f"`{r.get('summary_path')}`",
                        ]
                    )
                    + " |"
                )
            lines.append("")

    lines.append("## Outputs\n")
    lines.append(f"- out_dir: `{_rel_to_eval(out_dir)}`")
    lines.append(f"- all rows: `{_rel_to_eval(all_csv)}`")
    lines.append(f"- best-by-AP: `{_rel_to_eval(best_csv)}`")
    lines.append(f"- plots: `{_rel_to_eval(plots_dir)}`\n")

    # Brief interpretation hint.
    lines.append("## How To Read (practical)\n")
    lines.append("1) Use `det_geom_best_by_ap.csv` as the main table (one point per run_dir/mode).")
    lines.append("2) Check `fairness_tier`: treat `legacy_unknown` as exploratory (protocol fields missing).")
    lines.append("3) For geometry completion, `scale_to_gt_mult_err_mean` and `pose_abs_mean` exist for most historical runs;")
    lines.append("   `cross_agent_pose_*` exists only for newer evals, so its scatter may have few points.\n")

    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
