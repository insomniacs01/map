#!/usr/bin/env python3
"""
Visualize OPV2V ground-truth LiDAR point cloud + ground-truth vehicle boxes as offline HTML.

This is intended as a coordinate-alignment baseline: GT points and GT boxes should match.

Examples:
  # Directly specify a frame
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_gt_pcd_boxes_html.py \
    --split validate --sequence 2021_08_20_21_48_35 --agent 00000 --frame 000000

  # Pick a cooperative scene by index (reproduces OPV2VCoopCylindricalDataset scene ordering)
  PYTHONPATH=$(pwd) python scripts/viz_opv2v_gt_pcd_boxes_html.py \
    --split validate --index 0 --min_agents 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_processing.opv2v_pose_utils import load_frame_metadata
from mapanything.datasets.opv2v_cyl import _extract_vehicle_boxes_in_ego


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points


def _filter_points(
    points: np.ndarray,
    *,
    z_min: float | None,
    z_max: float | None,
    radius_max: float | None,
) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= points[:, 2] <= float(z_max)
    if radius_max is not None:
        mask &= np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) <= float(radius_max)
    return points[mask]


def _downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    corners_all = []
    for box in boxes:
        x, y, z, l, w, h, yaw = [float(v) for v in box]
        dx = l / 2.0
        dy = w / 2.0
        dz = h / 2.0
        local = np.array(
            [
                [dx, dy, dz],
                [dx, -dy, dz],
                [-dx, -dy, dz],
                [-dx, dy, dz],
                [dx, dy, -dz],
                [dx, -dy, -dz],
                [-dx, -dy, -dz],
                [-dx, dy, -dz],
            ],
            dtype=np.float32,
        )
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        corners = local @ rot.T + np.array([x, y, z], dtype=np.float32)
        corners_all.append(corners)
    if not corners_all:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return np.stack(corners_all, axis=0)


def _iter_box_edges() -> Iterable[Tuple[int, int]]:
    return [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def _add_boxes(fig: go.Figure, corners: np.ndarray, *, name: str, color: str) -> None:
    if corners.size == 0:
        return

    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    for corner in corners:
        for a, b in _iter_box_edges():
            xs.extend([float(corner[a, 0]), float(corner[b, 0]), None])
            ys.extend([float(corner[a, 1]), float(corner[b, 1]), None])
            zs.extend([float(corner[a, 2]), float(corner[b, 2]), None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name=name,
            line=dict(color=color, width=4),
        )
    )


def _add_bev_boxes(fig: go.Figure, boxes: np.ndarray, *, name: str, color: str) -> None:
    if boxes.size == 0:
        return

    xs: List[float | None] = []
    ys: List[float | None] = []
    for box in boxes:
        x, y, _z, l, w, _h, yaw = [float(v) for v in box]
        dx = l / 2.0
        dy = w / 2.0
        local = np.array(
            [
                [dx, dy],
                [dx, -dy],
                [-dx, -dy],
                [-dx, dy],
                [dx, dy],
            ],
            dtype=np.float32,
        )
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        poly = local @ rot.T + np.array([x, y], dtype=np.float32)
        xs.extend([float(v) for v in poly[:, 0]] + [None])
        ys.extend([float(v) for v in poly[:, 1]] + [None])

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        )
    )


def _count_points_in_box(points: np.ndarray, box: np.ndarray) -> int:
    x, y, z, l, w, h, yaw = [float(v) for v in box]
    rel = points.astype(np.float64) - np.array([x, y, z], dtype=np.float64)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    xloc = c * rel[:, 0] - s * rel[:, 1]
    yloc = s * rel[:, 0] + c * rel[:, 1]
    zloc = rel[:, 2]
    inside = (np.abs(xloc) <= l / 2.0) & (np.abs(yloc) <= w / 2.0) & (np.abs(zloc) <= h / 2.0)
    return int(inside.sum())


def _discover_coop_scenes(
    *,
    opv2v_root: Path,
    split: str,
    min_agents: int,
    max_scenes: Optional[int],
) -> List[dict]:
    split_root = opv2v_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    scenes: List[dict] = []
    for sequence_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
        agent_dirs = [d for d in sequence_dir.iterdir() if d.is_dir()]
        if len(agent_dirs) < min_agents:
            continue

        frame_to_agents: dict[str, List[str]] = {}
        for agent_dir in agent_dirs:
            for yaml_path in agent_dir.glob("*.yaml"):
                frame_id = yaml_path.stem
                if not frame_id.isdigit():
                    continue
                frame_to_agents.setdefault(frame_id, []).append(agent_dir.name)

        for frame_id, agents in sorted(frame_to_agents.items()):
            agents_sorted = sorted(agents)
            if len(agents_sorted) < min_agents:
                continue
            scenes.append(
                {
                    "sequence": sequence_dir.name,
                    "frame": frame_id,
                    "agents": agents_sorted,
                }
            )
            if max_scenes is not None and len(scenes) >= int(max_scenes):
                return scenes
    if not scenes:
        raise RuntimeError(f"No cooperative OPV2V scenes found for split={split}")
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opv2v-root", type=Path, default=REPO_ROOT / "data" / "opv2v")
    parser.add_argument("--split", type=str, default="validate")

    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None, help="Ego agent for GT visualization (main agent).")
    parser.add_argument("--frame", type=str, default=None)

    parser.add_argument("--index", type=int, default=None, help="Cooperative scene index (optional shortcut).")
    parser.add_argument("--min_agents", type=int, default=2)
    parser.add_argument("--max_scenes", type=int, default=None)

    parser.add_argument("--bbox_range", type=float, default=120.0)
    parser.add_argument("--max_num_boxes", type=int, default=256)
    parser.add_argument("--max_points", type=int, default=160_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--radius_max", type=float, default=None)
    parser.add_argument("--out_html", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    opv2v_root = args.opv2v_root.expanduser().resolve()

    if args.index is not None:
        scenes = _discover_coop_scenes(
            opv2v_root=opv2v_root,
            split=args.split,
            min_agents=int(args.min_agents),
            max_scenes=args.max_scenes,
        )
        if args.index < 0 or args.index >= len(scenes):
            raise IndexError(f"index out of range: {args.index} (num_scenes={len(scenes)})")
        scene = scenes[int(args.index)]
        sequence = str(scene["sequence"])
        frame = str(scene["frame"])
        agent = str(args.agent) if args.agent else str(scene["agents"][0])
        if agent not in scene["agents"]:
            raise ValueError(f"--agent {agent} not in available agents: {scene['agents']}")
    else:
        if not (args.sequence and args.agent and args.frame):
            raise ValueError("Provide either --index, or all of --sequence/--agent/--frame.")
        sequence = str(args.sequence)
        agent = str(args.agent)
        frame = str(args.frame)

    yaml_path = opv2v_root / args.split / sequence / agent / f"{frame}.yaml"
    pcd_path = opv2v_root / args.split / sequence / agent / f"{frame}.pcd"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    if not pcd_path.is_file():
        raise FileNotFoundError(f"PCD not found: {pcd_path}")

    meta = load_frame_metadata(yaml_path)
    boxes_padded, boxes_mask = _extract_vehicle_boxes_in_ego(
        meta, max_range=float(args.bbox_range), max_num_boxes=int(args.max_num_boxes)
    )
    boxes = boxes_padded[boxes_mask.astype(bool)]
    corners = _boxes_to_corners(boxes)

    points = _load_ascii_pcd_xyz(pcd_path)
    points = _filter_points(points, z_min=args.z_min, z_max=args.z_max, radius_max=args.radius_max)
    points = _downsample(points, int(args.max_points), rng)

    counts = [_count_points_in_box(points, box) for box in boxes]
    count_summary = "n/a"
    if counts:
        count_summary = f"min/med/max={min(counts)}/{int(np.median(counts))}/{max(counts)}"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("BEV (X forward, Y right)", "3D (CARLA ego)"),
    )
    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            name="GT LiDAR",
            marker=dict(size=2, color=points[:, 2], colorscale="Turbo", opacity=0.35, showscale=False),
        ),
        row=1,
        col=1,
    )
    _add_bev_boxes(fig, boxes, name="GT boxes", color="rgba(0,0,0,0.9)")
    fig.add_trace(
        go.Scatter(
            x=[0.0, 10.0, None, 0.0, 0.0, None],
            y=[0.0, 0.0, None, 0.0, 10.0, None],
            mode="lines",
            name="Axes (+X forward, +Y right)",
            line=dict(color="rgba(0,0,0,0.5)", width=3),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            name="GT LiDAR (3D)",
            marker=dict(size=1, color="rgba(140,140,140,0.35)"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _add_boxes(fig, corners, name="GT boxes (3D)", color="rgba(0,0,0,0.9)")

    fig.update_layout(
        title=(
            f"OPV2V GT baseline: split={args.split} seq={sequence} agent={agent} frame={frame} "
            f"(points={points.shape[0]}, boxes={boxes.shape[0]}, pts_in_box={count_summary})"
        ),
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    out_path = args.out_html
    if out_path is None:
        tag = f"{sequence}_{agent}_{frame}"
        if args.index is not None:
            tag = f"idx{int(args.index)}_{tag}"
        out_path = Path("eval_runs/gt_baseline") / args.split / f"{tag}.html"
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="directory", full_html=True)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
