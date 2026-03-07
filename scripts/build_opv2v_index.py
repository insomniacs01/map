#!/usr/bin/env python
"""Build a global OPV2V index cache for fast filtering.

This script scans the OPV2V dataset once, extracts key per-frame metadata,
and writes a split-wise index under opv2v_index/ inside the dataset root.

The index is intended to be reusable across runs and filtering thresholds
without re-parsing YAML files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from data_processing.opv2v_pose_utils import load_frame_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OPV2V global index")
    parser.add_argument(
        "--root",
        required=True,
        help="OPV2V root (contains train/validate/test)",
    )
    parser.add_argument(
        "--depth-root",
        required=True,
        help="Depth root (contains train/validate/test depth npy)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for opv2v_index",
    )
    parser.add_argument(
        "--splits",
        default="train,validate,test",
        help="Comma-separated splits to scan (default: train,validate,test)",
    )
    parser.add_argument(
        "--camera-ids",
        default="0,1,2,3",
        help="Comma-separated camera ids (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N sequences",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, 1=disable).",
    )
    return parser.parse_args()


def _iter_sequence_dirs(split_root: Path) -> Iterable[Path]:
    for item in sorted(split_root.iterdir()):
        if item.is_dir():
            yield item


def _build_frame_to_agents(sequence_dir: Path) -> Dict[str, List[str]]:
    frame_to_agents: Dict[str, List[str]] = {}
    for agent_dir in sorted(d for d in sequence_dir.iterdir() if d.is_dir()):
        for yaml_path in agent_dir.glob("*.yaml"):
            frame_id = yaml_path.stem
            if not frame_id.isdigit():
                continue
            frame_to_agents.setdefault(frame_id, []).append(agent_dir.name)
    return frame_to_agents


def _has_assets(
    *,
    root: Path,
    depth_root: Path,
    split: str,
    sequence: str,
    agent_id: str,
    frame_id: str,
    camera_keys: List[str],
) -> bool:
    agent_dir = root / split / sequence / agent_id
    depth_dir = depth_root / split / sequence / agent_id
    for cam_key in camera_keys:
        img_path = agent_dir / f"{frame_id}_{cam_key}.png"
        depth_path = depth_dir / f"{frame_id}_{cam_key}_depth.npy"
        if not img_path.is_file() or not depth_path.is_file():
            return False
    return True


def _scan_sequence(
    *,
    sequence_dir: Path,
    root: Path,
    depth_root: Path,
    split: str,
    camera_keys: List[str],
) -> List[Dict]:
    rows: List[Dict] = []
    frame_to_agents = _build_frame_to_agents(sequence_dir)
    sequence = sequence_dir.name

    for frame_id, agents in sorted(frame_to_agents.items()):
        agents_sorted = sorted(agents)
        agent_pose = {}
        agent_xy = {}
        agent_has_assets = {}
        agent_num_boxes = {}
        agent_has_boxes = {}

        for agent_id in agents_sorted:
            yaml_path = sequence_dir / agent_id / f"{frame_id}.yaml"
            if not yaml_path.is_file():
                continue
            meta = load_frame_metadata(yaml_path)
            lidar_pose = meta.get("lidar_pose")
            if lidar_pose is None:
                continue
            lidar_pose = [float(x) for x in lidar_pose]
            agent_pose[agent_id] = lidar_pose
            agent_xy[agent_id] = [float(lidar_pose[0]), float(lidar_pose[1])]
            agent_has_assets[agent_id] = _has_assets(
                root=root,
                depth_root=depth_root,
                split=split,
                sequence=sequence,
                agent_id=agent_id,
                frame_id=frame_id,
                camera_keys=camera_keys,
            )
            vehicles = meta.get("vehicles") or {}
            num_boxes = len(vehicles) if isinstance(vehicles, dict) else 0
            agent_num_boxes[agent_id] = int(num_boxes)
            agent_has_boxes[agent_id] = bool(num_boxes)

        if not agent_pose:
            continue

        main_agent = sorted(agent_pose.keys())[0]
        pair_agent = None
        if len(agent_pose) > 1:
            mx, my = agent_xy[main_agent]
            best_dist = None
            for agent_id, (ax, ay) in agent_xy.items():
                if agent_id == main_agent:
                    continue
                if not agent_has_assets.get(agent_id, False):
                    continue
                dist = ((ax - mx) ** 2 + (ay - my) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    pair_agent = agent_id

        rows.append(
            dict(
                split=split,
                sequence=sequence,
                frame=str(frame_id),
                agents=agents_sorted,
                main_agent=main_agent,
                pair_agent=pair_agent,
                num_agents=len(agent_pose),
                num_agents_with_assets=sum(1 for v in agent_has_assets.values() if v),
                agent_pose=agent_pose,
                agent_xy=agent_xy,
                agent_has_assets=agent_has_assets,
                agent_num_boxes=agent_num_boxes,
                agent_has_boxes=agent_has_boxes,
            )
        )

    return rows


def _scan_split(
    *,
    root: Path,
    depth_root: Path,
    split: str,
    camera_keys: List[str],
    print_every: int,
    workers: int,
) -> pd.DataFrame:
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    seq_dirs = list(_iter_sequence_dirs(split_root))
    rows: List[Dict] = []

    if workers <= 1:
        for idx, sequence_dir in enumerate(seq_dirs, start=1):
            rows.extend(
                _scan_sequence(
                    sequence_dir=sequence_dir,
                    root=root,
                    depth_root=depth_root,
                    split=split,
                    camera_keys=camera_keys,
                )
            )
            if print_every > 0 and idx % print_every == 0:
                print(f"[{split}] scanned {idx}/{len(seq_dirs)} sequences")
        return pd.DataFrame(rows)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _scan_sequence,
                sequence_dir=sequence_dir,
                root=root,
                depth_root=depth_root,
                split=split,
                camera_keys=camera_keys,
            )
            for sequence_dir in seq_dirs
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if print_every > 0 and idx % print_every == 0:
                print(f"[{split}] scanned {idx}/{len(seq_dirs)} sequences")

    return pd.DataFrame(rows)


def _write_schema(output_dir: Path) -> None:
    schema = {
        "version": "opv2v_index_v1",
        "description": "Global OPV2V index cache (frame-level metadata)",
        "fields": {
            "split": "string",
            "sequence": "string",
            "frame": "string",
            "agents": "list[string] (sorted)",
            "main_agent": "string (default=sorted(agents)[0])",
            "pair_agent": "string or null (nearest agent with assets)",
            "num_agents": "int",
            "num_agents_with_assets": "int",
            "agent_pose": "dict[agent_id -> [x,y,z,roll,yaw,pitch]]",
            "agent_xy": "dict[agent_id -> [x,y]]",
            "agent_has_assets": "dict[agent_id -> bool]",
            "agent_num_boxes": "dict[agent_id -> int]",
            "agent_has_boxes": "dict[agent_id -> bool]",
        },
    }
    with (output_dir / "schema.json").open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)


def main() -> None:
    args = _parse_args()
    root = Path(args.root).resolve()
    depth_root = Path(args.depth_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    camera_ids = [int(x.strip()) for x in args.camera_ids.split(",") if x.strip()]
    camera_keys = [f"camera{idx}" for idx in camera_ids]

    summary = {
        "version": "opv2v_index_v1",
        "root": str(root),
        "depth_root": str(depth_root),
        "splits": splits,
        "camera_keys": camera_keys,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {},
    }

    for split in splits:
        df = _scan_split(
            root=root,
            depth_root=depth_root,
            split=split,
            camera_keys=camera_keys,
            print_every=args.print_every,
            workers=args.workers,
        )
        out_path = output_dir / f"index_{split}.parquet"
        df.to_parquet(out_path, index=False)
        summary["counts"][split] = len(df)
        print(f"[OK] {split}: {len(df)} frames -> {out_path}")

    _write_schema(output_dir)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("[DONE] OPV2V index built successfully")


if __name__ == "__main__":
    main()
