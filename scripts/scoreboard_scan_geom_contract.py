#!/usr/bin/env python3
"""
Scan `eval_runs/**/summary_test.json` and report geometry metrics on one fixed frame contract.

Use this to avoid cross-sample comparison mistakes:
  - Only keep runs with `split=test`
  - Only keep runs whose `(sequence, frame)` set matches the canonical contract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_pairs(frames: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in frames:
        if not isinstance(it, dict):
            continue
        seq = it.get("sequence")
        frame = it.get("frame")
        if seq is None or frame is None:
            continue
        out.append((str(seq), str(frame)))
    return out


def _frames_hash_md5_v1(frames: Iterable[Dict[str, Any]]) -> str:
    items = sorted({f"{seq}/{frame}" for seq, frame in _frame_pairs(frames)})
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _frames_hash_md5_v2(frames: Iterable[Dict[str, Any]]) -> str:
    """Fair hash: include main_agent + coop_agents order (prevents silent pair drift)."""
    items = []
    for it in frames:
        if not isinstance(it, dict):
            continue
        seq = it.get("sequence")
        frame = it.get("frame")
        main = it.get("main_agent")
        coop = it.get("coop_agents")
        if seq is None or frame is None:
            continue
        if main is None or coop is None or not isinstance(coop, (list, tuple)):
            # If pairing is missing, v2 hash cannot be trusted.
            continue
        coop_s = ",".join(str(a) for a in coop)
        items.append(f"{str(seq)}/{str(frame)}/main={str(main)}/coop={coop_s}")
    items = sorted(set(items))
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()

def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if x != x:
            return "nan"
        if not math.isfinite(x):
            return "inf"
        return f"{x:.{nd}f}"
    return str(x)


@dataclass(frozen=True)
class Row:
    path: Path
    run_dir: str
    model: str
    mode: str
    pose_abs_mean: Optional[float]
    pose_rot_mean: Optional[float]
    cross_agent_pose_trans_mean: Optional[float]
    cross_agent_pose_rot_mean: Optional[float]
    intra_agent_rig_pose_trans_mean: Optional[float]
    intra_agent_rig_pose_rot_mean: Optional[float]
    pose_ate_mean: Optional[float]
    depth_rel_mean: Optional[float]
    depth_rmse_mean: Optional[float]
    scale_err_mean: Optional[float]
    scale_log_err_mean: Optional[float]
    scale_eq_rel_err_mean: Optional[float]
    scale_mult_err_mean: Optional[float]
    scale_ratio_mean: Optional[float]
    scale_gt_factor_mean: Optional[float]
    scale_to_gt_err_mean: Optional[float]
    scale_to_gt_log_err_mean: Optional[float]
    scale_to_gt_eq_rel_err_mean: Optional[float]
    scale_to_gt_mult_err_mean: Optional[float]
    scale_to_gt_ratio_mean: Optional[float]
    chamfer_pred_to_gt_mean: Optional[float]
    chamfer_gt_to_pred_mean: Optional[float]
    bev_iou_raw_mean: Optional[float]
    bev_iou_filtered_mean: Optional[float]


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _extract_rows(summary_path: Path, payload: Dict[str, Any]) -> List[Row]:
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        return []
    out: List[Row] = []
    for model_key, per_mode in metrics.items():
        if not isinstance(per_mode, dict):
            continue
        for mode, m in per_mode.items():
            if not isinstance(m, dict):
                continue
            out.append(
                Row(
                    path=summary_path,
                    run_dir=summary_path.parent.name,
                    model=str(model_key),
                    mode=str(mode),
                    pose_abs_mean=m.get("pose_abs_mean"),
                    pose_rot_mean=m.get("pose_rot_mean"),
                    cross_agent_pose_trans_mean=m.get("cross_agent_pose_trans_mean"),
                    cross_agent_pose_rot_mean=m.get("cross_agent_pose_rot_mean"),
                    intra_agent_rig_pose_trans_mean=m.get("intra_agent_rig_pose_trans_mean"),
                    intra_agent_rig_pose_rot_mean=m.get("intra_agent_rig_pose_rot_mean"),
                    pose_ate_mean=m.get("pose_ate_mean"),
                    depth_rel_mean=m.get("depth_rel_mean"),
                    depth_rmse_mean=m.get("depth_rmse_mean"),
                    scale_err_mean=m.get("scale_err_mean"),
                    scale_log_err_mean=m.get("scale_log_err_mean"),
                    scale_eq_rel_err_mean=m.get("scale_eq_rel_err_mean"),
                    scale_mult_err_mean=m.get("scale_mult_err_mean"),
                    scale_ratio_mean=m.get("scale_ratio_mean"),
                    scale_gt_factor_mean=m.get("scale_gt_factor_mean"),
                    scale_to_gt_err_mean=m.get("scale_to_gt_err_mean"),
                    scale_to_gt_log_err_mean=m.get("scale_to_gt_log_err_mean"),
                    scale_to_gt_eq_rel_err_mean=m.get("scale_to_gt_eq_rel_err_mean"),
                    scale_to_gt_mult_err_mean=m.get("scale_to_gt_mult_err_mean"),
                    scale_to_gt_ratio_mean=m.get("scale_to_gt_ratio_mean"),
                    chamfer_pred_to_gt_mean=m.get("chamfer_pred_to_gt_mean"),
                    chamfer_gt_to_pred_mean=m.get("chamfer_gt_to_pred_mean"),
                    bev_iou_raw_mean=m.get("bev_iou_raw_mean"),
                    bev_iou_filtered_mean=m.get("bev_iou_filtered_mean"),
                )
            )
    return out


def _canonical_frames(canonical_summary: Path | None, canonical_frames_json: Path | None) -> List[Dict[str, Any]]:
    if canonical_summary is not None:
        payload = _load_json(canonical_summary)
        frames = payload.get("frames") or []
        if not isinstance(frames, list) or not frames:
            raise SystemExit(f"canonical summary has no frames: {canonical_summary}")
        return frames
    if canonical_frames_json is not None:
        payload = _load_json(canonical_frames_json)
        frames = payload.get("frames") or []
        if not isinstance(frames, list) or not frames:
            raise SystemExit(f"canonical frames json has no frames: {canonical_frames_json}")
        return frames
    raise SystemExit("must provide canonical source")


def _sortable(x: Any, *, nan_to: float = 1e18) -> float:
    if _is_finite_number(x):
        return float(x)
    return nan_to


def _scale_sort_key(row: Row) -> float:
    # Prefer ratio-to-GT metric if available, fallback to legacy ratio-to-1 metric.
    if _is_finite_number(row.scale_to_gt_err_mean):
        return float(row.scale_to_gt_err_mean)
    return _sortable(row.scale_err_mean)


def _maybe_hold_until_external_queue_done(run_prefix: str) -> None:
    """Optional "hold" hook to keep the caller process alive.

    Some platform environments reclaim the instance immediately after the user entry command exits.
    If you have follow-up jobs running in a separate process tree, the instance may be destroyed
    before those jobs finish. This hook lets the caller keep running until a configured condition
    is met.

    Trigger: create a JSON file:
      REPO_ROOT/experiments/long_runs/<run_prefix>__pipeline/hold_until.json

    Supported config keys:
      - status_tsv: path to a TSV that is appended per job (absolute or relative to REPO_ROOT)
      - expected_jobs: int, wait until status_tsv has >= expected_jobs + 1 lines (header + jobs)
      - poll_sec: int, polling interval (default 30)
      - timeout_sec: int, max wait (default 6 hours). On timeout, continue anyway.
      - delete_on_exit: bool, delete hold file after condition is met/timeout (default true)
    """
    if not run_prefix:
        return

    hold_dir = REPO_ROOT / "experiments" / "long_runs" / f"{run_prefix}__pipeline"
    hold_path = hold_dir / "hold_until.json"
    if not hold_path.is_file():
        return

    try:
        cfg = _load_json(hold_path)
    except Exception as exc:
        print(f"[hold] ignore invalid {hold_path}: {exc}", file=sys.stderr)
        return

    status_tsv = str(cfg.get("status_tsv", "")).strip()
    expected_jobs = int(cfg.get("expected_jobs", 0) or 0)
    poll_sec = int(cfg.get("poll_sec", 30) or 30)
    timeout_sec = int(cfg.get("timeout_sec", 6 * 3600) or 6 * 3600)
    delete_on_exit = bool(cfg.get("delete_on_exit", True))

    if not status_tsv or expected_jobs <= 0:
        print(f"[hold] ignore {hold_path}: require status_tsv + expected_jobs>0", file=sys.stderr)
        return

    status_path = Path(status_tsv)
    if not status_path.is_absolute():
        status_path = (REPO_ROOT / status_path).resolve()

    min_rows = expected_jobs + 1  # header + jobs
    start = time.time()
    last_heartbeat = 0.0

    print(
        f"[hold] enabled by {hold_path} (waiting for {status_path} rows>={min_rows}, timeout={timeout_sec}s)",
        file=sys.stderr,
    )

    while True:
        try:
            n_rows = len(status_path.read_text(encoding="utf-8").splitlines())
        except Exception:
            n_rows = 0

        if n_rows >= min_rows:
            print(f"[hold] satisfied: {status_path} rows={n_rows} >= {min_rows}", file=sys.stderr)
            break

        elapsed = time.time() - start
        if elapsed >= timeout_sec:
            print(
                f"[hold] timeout after {int(elapsed)}s (rows={n_rows} < {min_rows}); continue anyway",
                file=sys.stderr,
            )
            break

        if elapsed - last_heartbeat >= max(60, poll_sec):
            last_heartbeat = elapsed
            print(f"[hold] waiting... elapsed={int(elapsed)}s rows={n_rows}/{min_rows}", file=sys.stderr)

        time.sleep(max(1, poll_sec))

    if delete_on_exit:
        try:
            hold_path.unlink()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "eval_runs")
    ap.add_argument("--canonical_summary", type=Path, default=None)
    ap.add_argument("--canonical_frames_json", type=Path, default=REPO_ROOT / "eval_runs" / "frames_test500_seed42.json")
    ap.add_argument("--run_prefix", type=str, default="")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    # Normalize to absolute paths to avoid `Path.relative_to` surprises when users pass relative paths.
    args.root = args.root.resolve()
    if args.canonical_summary is not None:
        args.canonical_summary = args.canonical_summary.resolve()
    if args.canonical_frames_json is not None:
        args.canonical_frames_json = args.canonical_frames_json.resolve()

    _maybe_hold_until_external_queue_done(args.run_prefix)

    can_frames = _canonical_frames(args.canonical_summary, args.canonical_frames_json)
    can_hash_v1 = _frames_hash_md5_v1(can_frames)
    can_hash_v2 = _frames_hash_md5_v2(can_frames)
    can_len = len(can_frames)

    rows: List[Row] = []
    for summary_path in args.root.rglob("summary_test.json"):
        if args.run_prefix and args.run_prefix not in str(summary_path.parent.name):
            continue
        try:
            payload = _load_json(summary_path)
        except Exception:
            continue
        if payload.get("split") != "test":
            continue
        frames = payload.get("frames") or []
        if not isinstance(frames, list) or len(frames) != can_len:
            continue
        # Prefer v2 (pair-aware) matching; fall back to v1 when v2 is unavailable.
        summ_hash_v2 = _frames_hash_md5_v2(frames)
        if summ_hash_v2 and summ_hash_v2 == can_hash_v2:
            pass
        elif _frames_hash_md5_v1(frames) != can_hash_v1:
            continue
        rows.extend(_extract_rows(summary_path, payload))

    if args.model:
        rows = [r for r in rows if r.model == args.model]

    print("## Geometry Contract")
    if args.canonical_summary is not None:
        print(f"- canonical_summary: `{args.canonical_summary.relative_to(REPO_ROOT)}`")
    else:
        assert args.canonical_frames_json is not None
        print(f"- canonical_frames_json: `{args.canonical_frames_json.relative_to(REPO_ROOT)}`")
    print(f"- frames: {can_len}")
    print(f"- frames_hash_md5 (v1 seq/frame): `{can_hash_v1}`")
    print(f"- frames_hash_md5_v2 (pair-aware): `{can_hash_v2}`")
    print(f"- matched_rows: {len(rows)}")
    print("")

    def show(mode: str) -> None:
        rs = [r for r in rows if r.mode == mode]
        if mode == "coop":
            rs.sort(
                key=lambda r: (
                    _scale_sort_key(r),
                    _sortable(r.cross_agent_pose_trans_mean),
                    _sortable(r.depth_rel_mean),
                )
            )
            print(
                f"## Mode={mode} (sorted by scale_to_gt_err_mean, cross_agent_pose_trans_mean, depth_rel_mean)"
            )
        else:
            rs.sort(
                key=lambda r: (
                    _scale_sort_key(r),
                    _sortable(r.pose_abs_mean),
                    _sortable(r.depth_rel_mean),
                )
            )
            print(f"## Mode={mode} (sorted by scale_to_gt_err_mean, pose_abs_mean, depth_rel_mean)")
        print(
            "| run_dir | model | pose_abs | pose_rot | cross_trans | cross_rot | rig_trans | rig_rot | pose_ate | depth_rel | depth_rmse | scale_to_gt_err | scale_to_gt_log_err | scale_to_gt_mult_err | scale_to_gt_ratio | scale_gt_factor | legacy_scale_err | legacy_scale_mult_err | legacy_scale_ratio | chamfer_p2g | chamfer_g2p | bev_iou_raw | bev_iou_filtered | evidence |"
        )
        print(
            "| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| --- |"
        )
        for r in rs[: args.topk]:
            print(
                "| "
                + " | ".join(
                    [
                        r.run_dir,
                        r.model,
                        _fmt(r.pose_abs_mean),
                        _fmt(r.pose_rot_mean),
                        _fmt(r.cross_agent_pose_trans_mean),
                        _fmt(r.cross_agent_pose_rot_mean),
                        _fmt(r.intra_agent_rig_pose_trans_mean),
                        _fmt(r.intra_agent_rig_pose_rot_mean),
                        _fmt(r.pose_ate_mean),
                        _fmt(r.depth_rel_mean),
                        _fmt(r.depth_rmse_mean),
                        _fmt(r.scale_to_gt_err_mean),
                        _fmt(r.scale_to_gt_log_err_mean),
                        _fmt(r.scale_to_gt_mult_err_mean),
                        _fmt(r.scale_to_gt_ratio_mean),
                        _fmt(r.scale_gt_factor_mean),
                        _fmt(r.scale_err_mean),
                        _fmt(r.scale_mult_err_mean),
                        _fmt(r.scale_ratio_mean),
                        _fmt(r.chamfer_pred_to_gt_mean),
                        _fmt(r.chamfer_gt_to_pred_mean),
                        _fmt(r.bev_iou_raw_mean),
                        _fmt(r.bev_iou_filtered_mean),
                        f"`{r.path.relative_to(REPO_ROOT)}`",
                    ]
                )
                + " |"
            )
        print("")

    show("single")
    show("coop")


if __name__ == "__main__":
    main()
