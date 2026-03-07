#!/usr/bin/env python3
"""Plot per-frame metric distributions (and baseline-distance correlations) for a fixed OPV2V contract.

This is a readability / interpretation tool: it turns a per-frame eval CSV
(e.g. `coop_metrics.csv`) into a small set of PNG plots + a short markdown report.

Typical usage (Test500, coop):
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/plot_contract_metrics_report.py \
    --frames_json eval_runs/frames_test500_seed42.json \
    --images_root data/opv2v \
    --metrics_csv eval_runs/geom_fairlock_test500_20260226_promoted_v2/geom_model/coop_metrics.csv \
    --out_dir eval_runs/geom_fairlock_test500_20260226_promoted_v2/plots \
    --title promoted_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _stats(xs: np.ndarray) -> Dict[str, float]:
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {}
    qs = np.quantile(xs, [0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "n": float(xs.size),
        "mean": float(np.mean(xs)),
        "p10": float(qs[1]),
        "p50": float(qs[2]),
        "p90": float(qs[3]),
        "p95": float(qs[4]),
        "p99": float(qs[5]),
        "min": float(qs[0]),
        "max": float(qs[6]),
    }


def _write_png(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


@dataclass(frozen=True)
class FrameKey:
    sequence: str
    frame: str
    main_agent: str
    other_agent: str


def _parse_frames_json(frames_json: Path) -> Tuple[str, List[FrameKey]]:
    doc = json.loads(frames_json.read_text(encoding="utf-8"))
    split = str(doc.get("split") or "")
    frames = doc.get("frames") or []
    if not split or not isinstance(frames, list):
        raise ValueError(f"Bad frames_json: {frames_json}")
    out: List[FrameKey] = []
    for fr in frames:
        if not isinstance(fr, Mapping):
            continue
        seq = fr.get("sequence")
        frame = fr.get("frame")
        main = fr.get("main_agent")
        coop = fr.get("coop_agents") or []
        if not (isinstance(seq, str) and isinstance(frame, str) and main):
            continue
        if not (isinstance(coop, list) and len(coop) == 2):
            continue
        main_s = str(main)
        a0, a1 = str(coop[0]), str(coop[1])
        other = a0 if a0 != main_s else a1
        out.append(FrameKey(sequence=seq, frame=frame, main_agent=main_s, other_agent=other))
    return split, out


def _baseline_distance_m(*, images_root: Path, split: str, key: FrameKey) -> float:
    # Local import keeps this script runnable in limited environments.
    from data_processing.opv2v_pose_utils import load_frame_metadata  # noqa: WPS433

    def lidar_xyz(agent: str) -> Tuple[float, float, float]:
        meta = load_frame_metadata(images_root / split / key.sequence / agent / f"{key.frame}.yaml")
        x, y, z = meta["lidar_pose"][:3]
        return float(x), float(y), float(z)

    x0, y0, z0 = lidar_xyz(key.main_agent)
    x1, y1, z1 = lidar_xyz(key.other_agent)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _load_metrics_csv(metrics_csv: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            seq = row.get("sequence")
            frame = row.get("frame")
            if not seq or not frame:
                continue
            out[(str(seq), str(frame))] = dict(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames_json", type=Path, required=True)
    ap.add_argument("--images_root", type=Path, required=True)
    ap.add_argument("--split", type=str, default=None, help="Override split (default: read from frames_json)")
    ap.add_argument("--metrics_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--cache_baseline_json", type=Path, default=None, help="Optional cache for baseline distances")
    args = ap.parse_args()

    frames_json = args.frames_json.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    metrics_csv = args.metrics_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_in, keys = _parse_frames_json(frames_json)
    split = str(args.split) if args.split else split_in
    if not split:
        raise ValueError("Empty split")

    rows = _load_metrics_csv(metrics_csv)

    cache_path = args.cache_baseline_json.expanduser().resolve() if args.cache_baseline_json else (out_dir / "baseline_distance_cache.json")
    cache: Dict[str, float] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    xs_dist: List[float] = []
    ys_cross_t: List[float] = []
    ys_cross_r: List[float] = []
    ys_depth_rel: List[float] = []
    ys_ratio: List[float] = []
    ys_mult: List[float] = []

    used = 0
    for k in keys:
        row = rows.get((k.sequence, k.frame))
        if row is None:
            continue
        cache_key = f"{k.sequence}/{k.frame}/{k.main_agent}/{k.other_agent}"
        dist = cache.get(cache_key)
        if dist is None:
            dist = _baseline_distance_m(images_root=images_root, split=split, key=k)
            cache[cache_key] = float(dist)

        cross_t = _safe_float(row.get("cross_agent_pose_trans_m"))
        cross_r = _safe_float(row.get("cross_agent_pose_rot_deg"))
        depth_rel = _safe_float(row.get("depth_rel"))
        ratio = _safe_float(row.get("scale_to_gt_ratio_mean"))
        eq_rel = _safe_float(row.get("scale_to_gt_eq_rel_err"))
        log_err = _safe_float(row.get("scale_to_gt_log_err"))
        mult = (eq_rel + 1.0) if eq_rel is not None else (math.exp(log_err) if log_err is not None else None)

        if cross_t is None or cross_r is None or depth_rel is None or ratio is None or mult is None:
            continue

        used += 1
        xs_dist.append(float(dist))
        ys_cross_t.append(float(cross_t))
        ys_cross_r.append(float(cross_r))
        ys_depth_rel.append(float(depth_rel))
        ys_ratio.append(float(ratio))
        ys_mult.append(float(mult))

    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    if used <= 0:
        raise SystemExit("No usable rows (missing required columns?)")

    dist = np.asarray(xs_dist, dtype=np.float64)
    cross_t = np.asarray(ys_cross_t, dtype=np.float64)
    cross_r = np.asarray(ys_cross_r, dtype=np.float64)
    depth_rel = np.asarray(ys_depth_rel, dtype=np.float64)
    ratio = np.asarray(ys_ratio, dtype=np.float64)
    mult = np.asarray(ys_mult, dtype=np.float64)

    title_prefix = (args.title.strip() + " | ") if args.title.strip() else ""

    # Histograms
    def hist(x: np.ndarray, *, name: str, xlabel: str, bins: int = 50) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
        ax.hist(x[np.isfinite(x)], bins=int(bins), color="tab:blue", alpha=0.85)
        ax.set_title(f"{title_prefix}{name}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        _write_png(fig, out_dir / f"hist_{name}.png")

    hist(dist, name="baseline_distance_m", xlabel="meters", bins=40)
    hist(cross_t, name="cross_trans_m", xlabel="meters", bins=60)
    hist(cross_r, name="cross_rot_deg", xlabel="degrees", bins=60)
    hist(depth_rel, name="depth_rel", xlabel="|dz|/z", bins=60)
    hist(ratio, name="scale_to_gt_ratio_mean", xlabel="s/g", bins=60)
    hist(mult, name="scale_to_gt_mult_err", xlabel="exp(|log(s/g)|)", bins=60)

    # Scatter plots vs baseline distance
    def scatter(x: np.ndarray, y: np.ndarray, *, name: str, xlabel: str, ylabel: str, ylog: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
        ax.scatter(x, y, s=8.0, alpha=0.35, c="tab:orange", edgecolors="none", rasterized=True)
        ax.set_title(f"{title_prefix}{name}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        if ylog:
            ax.set_yscale("log")
        _write_png(fig, out_dir / f"scatter_{name}.png")

    scatter(dist, cross_t, name="baseline_vs_cross_trans", xlabel="baseline distance (m)", ylabel="cross_trans (m)", ylog=True)
    scatter(dist, mult, name="baseline_vs_scale_mult_err", xlabel="baseline distance (m)", ylabel="scale mult err", ylog=True)
    scatter(dist, ratio, name="baseline_vs_scale_ratio", xlabel="baseline distance (m)", ylabel="scale ratio (s/g)", ylog=False)
    scatter(dist, depth_rel, name="baseline_vs_depth_rel", xlabel="baseline distance (m)", ylabel="depth_rel", ylog=True)

    def fmt_stats(name: str, x: np.ndarray) -> str:
        st = _stats(x)
        if not st:
            return f"- {name}: n=0"
        return (
            f"- {name}: n={int(st['n'])} mean={st['mean']:.4g} p50={st['p50']:.4g} "
            f"p90={st['p90']:.4g} p95={st['p95']:.4g} p99={st['p99']:.4g} max={st['max']:.4g}"
        )

    md: List[str] = []
    md.append("# Contract Metrics Report")
    md.append("")
    md.append(f"- frames_json: `{frames_json}`")
    md.append(f"- metrics_csv: `{metrics_csv}`")
    md.append(f"- images_root: `{images_root}`")
    md.append(f"- split: `{split}`")
    md.append(f"- used_rows: `{used}`")
    md.append("")
    md.append("## Key Stats")
    md.append("")
    md.append(fmt_stats("baseline_distance_m", dist))
    md.append(fmt_stats("cross_trans_m", cross_t))
    md.append(fmt_stats("cross_rot_deg", cross_r))
    md.append(fmt_stats("depth_rel", depth_rel))
    md.append(fmt_stats("scale_to_gt_ratio_mean", ratio))
    md.append(fmt_stats("scale_to_gt_mult_err", mult))
    md.append("")
    md.append("## Plots (PNG files)")
    md.append("")
    for p in sorted(out_dir.glob("*.png")):
        md.append(f"- `{p.name}`")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] wrote report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

