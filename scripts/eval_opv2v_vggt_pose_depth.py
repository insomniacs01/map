#!/usr/bin/env python3
"""
Evaluate VGGT pose/depth errors on OPV2V (single-agent or cooperative).

This script computes metric errors (meters / degrees) on a subset of samples:
  - Pose translation L2 error (m), relative to view0
  - Pose rotation angle error (deg), relative to view0
  - Depth-z MAE/RMSE (m) in camera frame

Example (coop, 8 views):
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/eval_opv2v_vggt_pose_depth.py \
    --checkpoint experiments/vggt/training/opv2v_coop_vggt_pose_metric/XXXX/checkpoint-best.pth \
    --mode coop --split validate --num_views 8 --max_samples 200

Example (single, 4 views):
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/eval_opv2v_vggt_pose_depth.py \
    --checkpoint experiments/vggt/training/opv2v_single_vggt_pose_metric/XXXX/checkpoint-best.pth \
    --mode single --split validate --num_views 4 --max_samples 500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mapanything.datasets.opv2v import OPV2VCoopDataset, OPV2VDataset
from mapanything.utils.geometry import transform_pose_using_quats_and_trans_2_to_1
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local


def _rotation_angle_deg_from_quats_xyzw(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dot)
    return ang * (180.0 / math.pi)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", type=str, choices=("single", "coop"), required=True)
    p.add_argument("--split", type=str, default="validate")
    p.add_argument("--num_views", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--principal_point_centered", action="store_true", default=True)
    p.add_argument("--no_principal_point_centered", dest="principal_point_centered", action="store_false")
    p.add_argument("--enable_metric_scale_head", action="store_true", help="Enable VGGT metric scale head (if checkpoint was trained with it).")
    p.add_argument("--images_root", type=Path, default=Path("data/opv2v_images"))
    p.add_argument("--depth_root", type=Path, default=Path("data/opv2v_depth"))
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--out_json", type=Path, default=None)
    return p


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = (repo_root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    local_cfg = {
        "path": str((repo_root / "configs" / "train.yaml").resolve()),
        "checkpoint_path": str(ckpt_path),
        "config_overrides": [
            "machine=local_a800",
            "model=vggt",
            "model.model_config.load_pretrained_weights=false",
            f"model.model_config.enable_metric_scale_head={'true' if args.enable_metric_scale_head else 'false'}",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)
    model.eval()

    images_root = (repo_root / args.images_root).resolve() if not args.images_root.is_absolute() else args.images_root
    depth_root = (repo_root / args.depth_root).resolve() if not args.depth_root.is_absolute() else args.depth_root

    if args.resolution is None:
        if args.mode == "coop":
            resolution = (448, 252)
        else:
            resolution = (518, 392)
    else:
        resolution = (int(args.resolution[0]), int(args.resolution[1]))

    if args.mode == "single":
        dataset = OPV2VDataset(
            split=args.split,
            ROOT=str(images_root),
            depth_root=str(depth_root),
            camera_ids=(0, 1, 2, 3),
            deterministic_camera_order=True,
            num_views=int(args.num_views),
            variable_num_views=False,
            resolution=resolution,
            principal_point_centered=bool(args.principal_point_centered),
            transform="imgnorm",
            data_norm_type="identity",
            seed=int(args.seed),
            max_num_retries=0,
        )
    else:
        dataset = OPV2VCoopDataset(
            split=args.split,
            ROOT=str(images_root),
            depth_root=str(depth_root),
            camera_ids=(0, 1, 2, 3),
            pair_agents=True,
            pair_agent_policy="nearest",
            num_views=int(args.num_views),
            variable_num_views=False,
            resolution=resolution,
            principal_point_centered=bool(args.principal_point_centered),
            transform="imgnorm",
            data_norm_type="identity",
            seed=int(args.seed),
            include_vehicle_boxes=False,
            min_agents=2,
            min_num_views=int(args.num_views),
            max_num_views=int(args.num_views),
            main_agent=None,
            main_agent_policy="first",
            max_num_retries=0,
        )

    max_samples = min(int(args.max_samples), len(dataset))
    indices = list(range(max_samples))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        drop_last=False,
    )

    depth_abs_sum = 0.0
    depth_sq_sum = 0.0
    depth_count = 0
    depth_mae_per_sample: list[float] = []

    pose_trans_sum = 0.0
    pose_rot_sum = 0.0
    pose_count = 0
    pose_trans_per_sample: list[float] = []
    pose_rot_per_sample: list[float] = []

    for batch_idx, views in enumerate(loader):
        for view in views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device, non_blocking=True)

        preds = model(views)

        # Depth metrics (camera-frame z depth)
        for view, pred in zip(views, preds):
            gt_depth = view["depthmap"][..., 0]  # (B,H,W)
            valid = view["valid_mask"]  # (B,H,W)
            pred_depth = pred["pts3d_cam"][..., 2]  # (B,H,W)
            mask = valid & (gt_depth > 1e-6) & torch.isfinite(pred_depth)
            if not bool(mask.any()):
                continue
            diff = (pred_depth[mask] - gt_depth[mask]).float()
            depth_abs_sum += float(diff.abs().sum().detach().cpu())
            depth_sq_sum += float((diff * diff).sum().detach().cpu())
            depth_count += int(mask.sum().detach().cpu())

        # Per-sample depth MAE (view0 only, for stable reporting)
        gt0 = views[0]["depthmap"][..., 0]
        valid0 = views[0]["valid_mask"]
        pr0 = preds[0]["pts3d_cam"][..., 2]
        mask0 = valid0 & (gt0 > 1e-6) & torch.isfinite(pr0)
        if bool(mask0.any()):
            depth_mae_per_sample.extend(
                ((pr0[mask0] - gt0[mask0]).abs().view(-1).float().mean().detach().cpu().repeat(int(args.batch_size))).tolist()
            )

        # Pose metrics (relative to view0)
        q0 = preds[0]["cam_quats"]
        t0 = preds[0]["cam_trans"]
        gt_q0 = views[0]["camera_pose_quats"]
        gt_t0 = views[0]["camera_pose_trans"]

        trans_err_this_sample = []
        rot_err_this_sample = []
        for view_idx in range(1, len(views)):
            q_rel, t_rel = transform_pose_using_quats_and_trans_2_to_1(
                q0, t0, preds[view_idx]["cam_quats"], preds[view_idx]["cam_trans"]
            )
            gt_q_rel, gt_t_rel = transform_pose_using_quats_and_trans_2_to_1(
                gt_q0,
                gt_t0,
                views[view_idx]["camera_pose_quats"],
                views[view_idx]["camera_pose_trans"],
            )
            trans_err = torch.linalg.norm((t_rel - gt_t_rel).float(), dim=-1)  # (B,)
            rot_err = _rotation_angle_deg_from_quats_xyzw(q_rel.float(), gt_q_rel.float())  # (B,)

            pose_trans_sum += float(trans_err.sum().detach().cpu())
            pose_rot_sum += float(rot_err.sum().detach().cpu())
            pose_count += int(trans_err.numel())
            trans_err_this_sample.append(trans_err.detach().cpu())
            rot_err_this_sample.append(rot_err.detach().cpu())

        if trans_err_this_sample:
            trans_err_b = torch.stack(trans_err_this_sample, dim=0).mean(dim=0)  # (B,)
            rot_err_b = torch.stack(rot_err_this_sample, dim=0).mean(dim=0)  # (B,)
            pose_trans_per_sample.extend(trans_err_b.tolist())
            pose_rot_per_sample.extend(rot_err_b.tolist())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"[{batch_idx+1:4d}/{len(loader):4d}] processed")

    depth_mae_mean = float(depth_abs_sum / max(1, depth_count))
    depth_rmse = float(math.sqrt(depth_sq_sum / max(1, depth_count)))
    depth_mae_med = float(np.median(depth_mae_per_sample)) if depth_mae_per_sample else float("nan")

    pose_trans_mean = float(pose_trans_sum / max(1, pose_count))
    pose_rot_mean = float(pose_rot_sum / max(1, pose_count))
    pose_trans_med = float(np.median(pose_trans_per_sample)) if pose_trans_per_sample else float("nan")
    pose_rot_med = float(np.median(pose_rot_per_sample)) if pose_rot_per_sample else float("nan")

    result = {
        "checkpoint": str(ckpt_path),
        "mode": str(args.mode),
        "split": str(args.split),
        "num_views": int(args.num_views),
        "resolution": [int(resolution[0]), int(resolution[1])],
        "max_samples": int(max_samples),
        "batch_size": int(args.batch_size),
        "depth_z_mae_mean_m": depth_mae_mean,
        "depth_z_mae_median_m": depth_mae_med,
        "depth_z_rmse_m": depth_rmse,
        "pose_trans_l2_mean_m": pose_trans_mean,
        "pose_trans_l2_median_m": pose_trans_med,
        "pose_rot_mean_deg": pose_rot_mean,
        "pose_rot_median_deg": pose_rot_med,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out_json is not None:
        out_path = args.out_json
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
