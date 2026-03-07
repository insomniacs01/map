#!/usr/bin/env python3
"""Compute OPV2V ground-truth scale factors for a fixed frame contract.

Background
----------
MapAnything scale supervision/regression in this repo uses the *normalization factor*
from `normalize_multiple_pointclouds(..., norm_mode="avg_dis", ret_factor=True)`.
That factor is the mean distance-to-origin (in the reference view0 camera frame)
over valid depth pixels, aggregated across all views.

For evaluation, the model emits `pred["metric_scaling_factor"]` (a positive scalar).
To judge scale correctness, you must compare it to the GT scale factor for the
same view set (single/coop), not to the constant 1.

This script computes those per-frame GT scale factors under a *fixed frames contract*
(`frames_json`), so downstream tools can report ratio-to-GT scale metrics fairly.

Output
------
Writes a JSON with per-frame:
  - gt_scale_single: using main-agent 4 cameras
  - gt_scale_coop: using all cameras from both agents (2 agents x 4 cameras)

Notes
-----
- The contract is assumed to use 2 agents per frame (`coop_agents` length == 2).
- The reference view0 is defined by the ordering in `build_coop_raw`: camera0 of
  the first agent in `coop_agents`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data_processing.opv2v_pose_utils import CARLA_TO_CAMERA_CV, cords_to_pose, load_frame_metadata


def _frames_hash_md5_v1(frames: List[Dict[str, Any]]) -> str:
    """Legacy hash: md5 over newline-joined, sorted unique `sequence/frame` identifiers."""
    items = sorted({f"{f['sequence']}/{f['frame']}" for f in frames if isinstance(f, dict)})
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _frames_hash_md5_v2(frames: List[Dict[str, Any]]) -> str:
    """Fair hash: include pairing (main + coop_agents) to prevent silent contract drift."""
    items = sorted(
        {
            f"{f['sequence']}/{f['frame']}"
            f"/main={f.get('main_agent')}"
            f"/coop={','.join([str(a) for a in (f.get('coop_agents') or [])])}"
            for f in frames
            if isinstance(f, dict)
        }
    )
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _gt_norm_factor_avg_dis_sparse(raw_views: List[Dict[str, Any]]) -> Tuple[float, int]:
    """Compute avg distance to origin in view0 camera frame over valid depth pixels.

    This is mathematically equivalent to `normalize_multiple_pointclouds(..., norm_mode="avg_dis")[-1]`
    but implemented with sparse valid pixels (depth>0) for speed.
    """

    if not raw_views:
        return float("nan"), 0

    pose0 = np.asarray(raw_views[0]["camera_poses"], dtype=np.float64)
    inv_pose0 = np.linalg.inv(pose0)

    total_sum = 0.0
    total_cnt = 0
    for rv in raw_views:
        depth = np.asarray(rv["depth_z"], dtype=np.float64)
        intr = np.asarray(rv["intrinsics"], dtype=np.float64)
        pose = np.asarray(rv["camera_poses"], dtype=np.float64)

        T = inv_pose0 @ pose  # cam_i -> cam0
        R = T[:3, :3]
        t = T[:3, 3]

        mask = depth > 0.0
        v, u = np.nonzero(mask)
        if v.size == 0:
            continue

        z = depth[v, u]
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts = np.stack([x, y, z], axis=0)  # (3, N)
        pts0 = (R @ pts) + t.reshape(3, 1)
        dis = np.linalg.norm(pts0, axis=0)

        total_sum += float(dis.sum())
        total_cnt += int(dis.size)

    if total_cnt <= 0:
        return float("nan"), 0
    return total_sum / float(total_cnt), total_cnt


def _convert_pose_to_opencv(pose_carla: np.ndarray) -> np.ndarray:
    pose_cv = np.asarray(pose_carla, dtype=np.float32).copy()
    basis = CARLA_TO_CAMERA_CV[:3, :3]
    pose_cv[:3, :3] = basis @ pose_cv[:3, :3] @ basis.T
    pose_cv[:3, 3] = basis @ pose_cv[:3, 3]
    return pose_cv


def _build_coop_raw_fast(
    images_root: Path,
    depth_root: Path,
    split: str,
    *,
    sequence: str,
    frame: str,
    main_agent: str,
    coop_agents: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """Fast reimplementation of `batch_eval.build_coop_raw` without loading images."""

    sequence_dir = images_root / split / sequence
    depth_dir = depth_root / split / sequence

    metadata: Dict[str, Dict[str, Any]] = {}
    for agent in coop_agents:
        yaml_path = sequence_dir / agent / f"{frame}.yaml"
        metadata[agent] = load_frame_metadata(yaml_path)

    main_meta = metadata[main_agent]
    T_world_main = cords_to_pose(main_meta["lidar_pose"])
    T_main_world = np.linalg.inv(T_world_main)

    raw_views: List[Dict[str, Any]] = []
    for agent in coop_agents:
        meta = metadata[agent]
        for cam_key in sorted(k for k in meta.keys() if k.startswith("camera")):
            depth_path = depth_dir / agent / f"{frame}_{cam_key}_depth.npy"
            depth = np.load(depth_path).astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            intr = np.asarray(meta[cam_key]["intrinsic"], dtype=np.float32)
            cam_pose_world = cords_to_pose(meta[cam_key]["cords"])
            cam_pose_main = T_main_world @ cam_pose_world

            raw_views.append(
                {
                    "depth_z": depth,
                    "intrinsics": intr,
                    "camera_poses": _convert_pose_to_opencv(cam_pose_main),
                }
            )

    return raw_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_json", type=Path, required=True)
    parser.add_argument(
        "--images_root",
        type=Path,
        default=Path("map-anything/data/opv2v_images"),
        help="Path to opv2v_images root (contains split/sequence/agent/*).",
    )
    parser.add_argument(
        "--depth_root",
        type=Path,
        default=Path("map-anything/data/opv2v_depth"),
        help="Path to opv2v_depth root (contains split/sequence/agent/*).",
    )
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args()

    # Reduce spam from intrinsics logging in `color_compare.rescale_intrinsics_to_image`.
    logging.getLogger("color_compare").setLevel(logging.ERROR)

    frames_doc = json.loads(args.frames_json.read_text())
    split = str(frames_doc.get("split"))
    frames: List[Dict[str, Any]] = list(frames_doc.get("frames") or [])
    if not frames:
        raise ValueError(f"No frames found in {args.frames_json}")
    if args.limit is not None:
        frames = frames[: int(args.limit)]

    out_frames = []
    t0 = time.time()

    for i, fr in enumerate(frames):
        sequence = str(fr["sequence"])
        frame = str(fr["frame"])
        main_agent = str(fr["main_agent"])
        coop_agents = tuple(str(a) for a in (fr.get("coop_agents") or []))
        if len(coop_agents) != 2:
            raise ValueError(f"Expected 2 coop_agents per frame; got {coop_agents} for {fr}")

        raw_views = _build_coop_raw_fast(
            args.images_root,
            args.depth_root,
            split,
            sequence=sequence,
            frame=frame,
            main_agent=main_agent,
            coop_agents=coop_agents,
        )
        # build_coop_raw order: agent0 cameras (4) then agent1 cameras (4)
        gt_single, cnt_single = _gt_norm_factor_avg_dis_sparse(raw_views[:4])
        gt_coop, cnt_coop = _gt_norm_factor_avg_dis_sparse(raw_views)

        out_frames.append(
            {
                "sequence": sequence,
                "frame": frame,
                "main_agent": main_agent,
                "coop_agents": list(coop_agents),
                "gt_scale_single": float(gt_single),
                "gt_scale_coop": float(gt_coop),
                "gt_valid_pixels_single": int(cnt_single),
                "gt_valid_pixels_coop": int(cnt_coop),
            }
        )

        if args.progress_every and (i + 1) % int(args.progress_every) == 0:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            eta = (len(frames) - (i + 1)) / max(rate, 1e-6)
            print(
                f"[{i+1}/{len(frames)}] gt scales computed | {rate:.2f} frames/s | ETA {eta/60:.1f} min",
                flush=True,
            )

    out = {
        "frames_json": str(args.frames_json),
        "split": split,
        "norm_mode": "avg_dis",
        # Keep v1 for backward compatibility; prefer v2 for fairness across pair policies.
        "frames_hash_md5": _frames_hash_md5_v1(out_frames),
        "frames_hash_md5_v2": _frames_hash_md5_v2(out_frames),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames": out_frames,
    }

    out_path = args.out_json
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
