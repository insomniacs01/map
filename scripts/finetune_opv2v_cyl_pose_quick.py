#!/usr/bin/env python3
"""
Quick pose finetune for OPV2V cylindrical cooperative scenes.

This is a pragmatic tool for debugging pose convergence: it finetunes ONLY the pose
prediction modules (pose head + pose adaptor) on a small set of frames so that
MapAnything predicted poses align with OPV2V GT ego poses.

Typical workflow:
  1) Run the GT baseline viz to confirm coordinate alignment.
  2) Run `viz_opv2v_pred_vs_gt_boxes_html.py` and observe `pose_err_rel` is large.
  3) Run this script for a few hundred steps.
  4) Re-run the viz and confirm `pose_err_rel` and pred-pose overlay improves.

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/finetune_opv2v_cyl_pose_quick.py \
    --init_checkpoint experiments/opv2v_coop_det_e2e/checkpoint-best.pth \
    --out_checkpoint experiments/opv2v_cyl_pose_quick/checkpoint-last.pth \
    --split validate --sequence 2021_08_21_17_30_41 --main_agent 2488 \
    --frames 000107,000109,000111 --num_views 3 --steps 400 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.utils.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _pose_matrix_from_quat_trans(quat_xyzw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    # Be defensive: at the beginning of finetuning, pose quats can be near-zero.
    # `quaternion_to_rotation_matrix` normalizes without eps, which can yield NaNs.
    quat_xyzw = torch.nan_to_num(quat_xyzw, nan=0.0, posinf=0.0, neginf=0.0)
    trans = torch.nan_to_num(trans, nan=0.0, posinf=0.0, neginf=0.0)
    eps = 1e-6
    norms = quat_xyzw.norm(dim=1, keepdim=True)
    if torch.any(norms < eps):
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat_xyzw.device, dtype=quat_xyzw.dtype).unsqueeze(0)
        quat_xyzw = torch.where(norms < eps, ident, quat_xyzw)
    rot = quaternion_to_rotation_matrix(quat_xyzw)
    mat = torch.eye(4, device=rot.device, dtype=rot.dtype).unsqueeze(0).repeat(rot.shape[0], 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


def _geodesic_angle_rad(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Return rotation angle between two rotation matrices, in radians."""
    r_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace_val = (r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2] - 1.0) / 2.0
    trace_val = torch.clamp(trace_val, -1.0, 1.0)
    return torch.acos(trace_val)


def _se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse for batched SE(3) matrices (B,4,4)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t)
    out = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
    out[:, :3, :3] = R_inv
    out[:, :3, 3:4] = t_inv
    return out


def _pose_loss_rel_to_view0(preds: Sequence[dict], views: Sequence[dict]) -> torch.Tensor:
    """Pose loss on relative SE(3): inv(T0)@Ti vs inv(GT0)@GTi."""
    if not preds or not views:
        raise ValueError("Empty preds/views")
    if len(preds) != len(views):
        raise ValueError(f"Expected len(preds)==len(views); got {len(preds)} vs {len(views)}")

    pred_quats = torch.cat([p["cam_quats"] for p in preds], dim=0)  # (V,4)
    pred_trans = torch.cat([p["cam_trans"] for p in preds], dim=0)  # (V,3)
    T_pred = _pose_matrix_from_quat_trans(pred_quats, pred_trans)  # (V,4,4)

    T_gt = torch.cat([v["camera_pose"] for v in views], dim=0)  # (V,4,4)

    T_pred0_inv = _se3_inverse(T_pred[0:1])
    T_gt0_inv = _se3_inverse(T_gt[0:1])

    pred_rel = torch.matmul(T_pred0_inv, T_pred)  # (V,4,4)
    gt_rel = torch.matmul(T_gt0_inv, T_gt)  # (V,4,4)

    t_err = torch.nn.functional.smooth_l1_loss(
        pred_rel[:, :3, 3],
        gt_rel[:, :3, 3],
        reduction="mean",
        beta=1.0,
    )
    # Rotation loss: avoid acos-based geodesic loss (unstable near identity, can create Inf grads).
    q_pred = rotation_matrix_to_quaternion(pred_rel[:, :3, :3])
    q_gt = rotation_matrix_to_quaternion(gt_rel[:, :3, :3])
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    dot = torch.sum(q_pred * q_gt, dim=-1)
    r_err = torch.mean(1.0 - dot.square())
    return t_err + 10.0 * r_err


def _has_nonfinite_pose(preds: Sequence[dict]) -> bool:
    for pred in preds:
        quat = pred.get("cam_quats")
        trans = pred.get("cam_trans")
        if quat is None or trans is None:
            continue
        if not torch.isfinite(quat).all():
            return True
        if not torch.isfinite(trans).all():
            return True
    return False


def _set_requires_grad(model: torch.nn.Module, enabled_modules: Iterable[str]) -> None:
    enabled = tuple(enabled_modules)
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name for key in enabled)


def _set_train_mode_for_modules(model: torch.nn.Module, *, train_module_names: Iterable[str]) -> None:
    """Keep the full model in eval mode, but enable train-mode for a subset of modules.

    This avoids BatchNorm running-stat drift (and dropout noise) in frozen parts of the network,
    which can otherwise destroy the dense mask/depth heads during quick finetuning.
    """

    model.eval()
    wanted = tuple(train_module_names)
    for name, module in model.named_modules():
        if any(key in name for key in wanted):
            module.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint", type=Path, required=True)
    parser.add_argument("--out_checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated frame ids (e.g. 000107,000109)")
    parser.add_argument("--main_agent", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=10.0, help="0 or negative disables clipping")
    parser.add_argument("--abort_on_nonfinite", action="store_true", help="Stop immediately if pose outputs become NaN/Inf")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0, overwrite --out_checkpoint every N steps (helps when long runs get interrupted).",
    )
    parser.add_argument(
        "--max_agent_distance",
        type=float,
        default=None,
        help="Optional: only sample cooperative agents whose GT ego distance to main is <= this XY radius (meters).",
    )
    parser.add_argument(
        "--agent_selection_policy",
        type=str,
        default="random",
        choices=("random", "nearest"),
        help="How to select agents when more than num_views are available.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_torch_hub",
        action="store_true",
        help="Allow loading encoder weights via torch hub if missing from checkpoint.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ckpt = (REPO_ROOT / args.init_checkpoint).resolve() if not args.init_checkpoint.is_absolute() else args.init_checkpoint
    local_cfg = {
        "path": str((REPO_ROOT / "configs" / "train.yaml").resolve()),
        "checkpoint_path": str(init_ckpt),
        "config_overrides": [
            "machine=local_a800",
            "model=mapanything",
            f"model.encoder.uses_torch_hub={'true' if args.use_torch_hub else 'false'}",
            "model/task=images_only",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)
    # Important: do NOT put the whole model in `train()` here. We only finetune a tiny subset
    # of pose modules, and enabling train-mode globally would update BatchNorm running stats
    # in frozen heads, often collapsing the predicted non-ambiguous mask.
    _set_train_mode_for_modules(model, train_module_names=("pose_head", "pose_adaptor"))

    # Freeze everything except pose prediction modules.
    _set_requires_grad(model, enabled_modules=("pose_head", "pose_adaptor"))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected; check module name filters.")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    frame_list = _parse_csv_list(args.frames)
    if not frame_list:
        raise ValueError("--frames must provide at least 1 frame id")

    dataset = OPV2VCoopCylindricalDataset(
        split=args.split,
        ROOT=str((REPO_ROOT / "data" / "opv2v_images").resolve()),
        depth_root=str((REPO_ROOT / "data" / "opv2v_depth").resolve()),
        camera_ids=(0, 1, 2, 3),
        num_views=int(args.num_views),
        variable_num_views=False,
        resolution=(1008, 252),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=777,
        min_agents=int(args.num_views),
        min_num_views=int(args.num_views),
        max_num_views=int(args.num_views),
        panorama_resolution=(1008, 252),
        panorama_vertical_fov_deg=90.0,
        panorama_elevation_center_deg=0.0,
        view_selection_margin=0.05,
        max_num_retries=0,
        main_agent=str(args.main_agent),
        ensure_main_agent_first=True,
        max_agent_distance=float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        agent_selection_policy=str(args.agent_selection_policy),
    )

    wanted = []
    frames_set = set(frame_list)
    for idx, scene in enumerate(dataset.scenes):
        if str(scene.get("sequence")) != str(args.sequence):
            continue
        if str(scene.get("frame")) not in frames_set:
            continue
        if str(args.main_agent) not in list(scene.get("agents") or []):
            continue
        wanted.append(int(idx))
    if not wanted:
        raise RuntimeError(
            f"No scenes found for split={args.split} seq={args.sequence} main_agent={args.main_agent} frames={frame_list}"
        )

    subset = torch.utils.data.Subset(dataset, wanted)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=True, num_workers=0)
    it = iter(loader)

    out_path = args.out_checkpoint
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(step: int) -> None:
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save({"model": model.state_dict(), "step": int(step)}, str(tmp_path))
        tmp_path.replace(out_path)
        print(f"[CKPT] step={step:04d} wrote {out_path}")

    for step in range(int(args.steps)):
        try:
            batch_views = next(it)
        except StopIteration:
            it = iter(loader)
            batch_views = next(it)

        # Move tensors to device.
        for view in batch_views:
            for key, value in view.items():
                if isinstance(value, torch.Tensor):
                    view[key] = value.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(batch_views)
        if _has_nonfinite_pose(preds):
            msg = f"[step {step:04d}] detected non-finite cam_quats/cam_trans"
            if args.abort_on_nonfinite:
                raise RuntimeError(msg)
            print(msg + "; skipping step")
            continue
        loss = _pose_loss_rel_to_view0(preds, batch_views)
        if not torch.isfinite(loss):
            print(f"[step {step:04d}] pose_loss is not finite ({loss}); skipping step")
            continue
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip))
        optimizer.step()

        if step % 20 == 0 or step == int(args.steps) - 1:
            print(f"[step {step:04d}] pose_loss={loss.item():.6f}")
        if int(args.save_every) > 0 and (step + 1) % int(args.save_every) == 0:
            _save_checkpoint(step + 1)

    _save_checkpoint(int(args.steps))
    print(f"[OK] final checkpoint at {out_path}")


if __name__ == "__main__":
    main()
