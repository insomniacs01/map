#!/usr/bin/env python3
"""
Quick geometry finetune for OPV2V cylindrical cooperative scenes.

Goal: make MapAnything predicted point cloud align with OPV2V geometry so that:
  - predicted-vs-GT LiDAR overlays look reasonable (pred points close to GT points)
  - cooperative pose stays consistent (relative SE(3) between agents is correct)

This script finetunes a small subset of modules (by name substring) on a small set
of frames using:
  - camera-frame pointmap supervision: pred['pts3d_cam'] vs GT view['pts3d_cam']
    (computed from OPV2V fused depth panoramas)
  - optional relative pose supervision: predicted relative pose vs GT relative pose

Example:
  cd map-anything
  PYTHONPATH=$(pwd) python scripts/finetune_opv2v_cyl_geom_quick.py \
    --init_checkpoint experiments/opv2v_cyl_pose_quick/checkpoint-test_2021_08_18_19_48_05_1045_v2.pth \
    --out_checkpoint experiments/opv2v_cyl_geom_quick/checkpoint-last.pth \
    --split test --sequence 2021_08_18_19_48_05 --main_agent 1045 \
    --frames 000280,000282,000284,000286,000288 --num_views 2 \
    --steps 400 --lr 5e-5 --w_pts_cam 1.0 --w_pose 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.utils.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local


REPO_ROOT = Path(__file__).resolve().parents[1]

_CARLA_TO_CV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _load_ascii_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("DATA"):
                break
        points = np.loadtxt(fh, dtype=np.float32, usecols=(0, 1, 2))
    if points.ndim == 1:
        points = points.reshape(1, 3)
    return points


def _lidar_pcd_to_cyl_depthmap(
    points_carla: np.ndarray,
    *,
    width: int,
    height: int,
    vertical_fov_deg: float,
    elevation_center_deg: float,
    azimuth_offset_rad: float,
) -> np.ndarray:
    """Project LiDAR point cloud into a cylindrical depthmap (range along ray) in OpenCV convention."""

    if points_carla.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    # CARLA (X forward, Y right, Z up) -> OpenCV (X right, Y down, Z forward)
    pts_cv = points_carla.astype(np.float32) @ _CARLA_TO_CV.T
    x = pts_cv[:, 0].astype(np.float64)
    y = pts_cv[:, 1].astype(np.float64)
    z = pts_cv[:, 2].astype(np.float64)
    r = np.sqrt(x * x + y * y + z * z)
    keep = r > 1e-6
    if not np.any(keep):
        return np.zeros((height, width), dtype=np.float32)
    x = x[keep]
    y = y[keep]
    z = z[keep]
    r = r[keep]

    az = np.arctan2(x, z)  # [-pi, pi]
    elev = np.arcsin(np.clip(-y / r, -1.0, 1.0))

    v_fov = np.deg2rad(float(vertical_fov_deg))
    elev_center = np.deg2rad(float(elevation_center_deg))
    elev_min = elev_center - v_fov / 2.0
    elev_max = elev_center + v_fov / 2.0
    keep = (elev >= elev_min) & (elev <= elev_max)
    if not np.any(keep):
        return np.zeros((height, width), dtype=np.float32)
    az = az[keep]
    elev = elev[keep]
    r = r[keep].astype(np.float32)

    u = (az - float(azimuth_offset_rad)) / (2.0 * np.pi) + 0.5
    u = np.mod(u, 1.0)
    v = (elev_max - elev) / (elev_max - elev_min)

    u_idx = np.floor(u * float(width)).astype(np.int32)
    v_idx = np.floor(v * float(height)).astype(np.int32)
    u_idx = np.clip(u_idx, 0, width - 1)
    v_idx = np.clip(v_idx, 0, height - 1)
    lin = v_idx * int(width) + u_idx

    depth = np.full((height * width,), np.inf, dtype=np.float32)
    np.minimum.at(depth, lin, r)
    depth = depth.reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _pose_matrix_from_quat_trans(quat_xyzw: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
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


def _se3_inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t)
    out = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
    out[:, :3, :3] = R_inv
    out[:, :3, 3:4] = t_inv
    return out


def _pose_loss_rel_to_view0(preds: Sequence[dict], views: Sequence[dict]) -> torch.Tensor:
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

    q_pred = rotation_matrix_to_quaternion(pred_rel[:, :3, :3])
    q_gt = rotation_matrix_to_quaternion(gt_rel[:, :3, :3])
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    dot = torch.sum(q_pred * q_gt, dim=-1)
    r_err = torch.mean(1.0 - dot.square())
    return t_err + 10.0 * r_err


def _first_scalar(value) -> str:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return ""
        return str(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def _pts3d_cam_loss(
    preds: Sequence[dict],
    views: Sequence[dict],
    *,
    supervision: str,
    lidar_root: Path,
    split: str,
    sequence: str,
    vertical_fov_deg: float,
    elevation_center_deg: float,
    azimuth_offset_rad: float,
    lidar_cache: dict[tuple[str, str], np.ndarray],
    lidar_depth_cache: dict[tuple[str, str, int, int, float, float, float], np.ndarray],
) -> torch.Tensor:
    if not preds or not views:
        raise ValueError("Empty preds/views")
    if len(preds) != len(views):
        raise ValueError(f"Expected len(preds)==len(views); got {len(preds)} vs {len(views)}")

    pred_pts = []
    gt_pts = []
    for pred, view in zip(preds, views):
        pts3d_cam_pred = pred.get("pts3d_cam")
        if pts3d_cam_pred is None or pts3d_cam_pred.ndim != 4 or pts3d_cam_pred.shape[-1] != 3:
            continue

        if str(supervision) == "dataset_depth":
            pts3d_cam_gt = view.get("pts3d_cam")
            valid_mask = view.get("valid_mask")
            if pts3d_cam_gt is None or valid_mask is None:
                continue
            if pts3d_cam_gt.ndim != 4 or pts3d_cam_gt.shape[-1] != 3:
                continue
            if valid_mask.ndim != 3:
                continue
        elif str(supervision) == "lidar":
            # Build a sparse cylindrical depthmap by projecting GT LiDAR points.
            agent_id = _first_scalar(view.get("agent_id"))
            instance = _first_scalar(view.get("instance"))
            frame_id = str(instance).split("/")[0] if instance else ""
            if not agent_id or not frame_id:
                continue

            key = (frame_id, agent_id)
            if key not in lidar_cache:
                pcd_path = lidar_root / split / sequence / agent_id / f"{frame_id}.pcd"
                if not pcd_path.is_file():
                    continue
                lidar_cache[key] = _load_ascii_pcd_xyz(pcd_path)

            pts_carla = lidar_cache[key]
            height, width = int(pts3d_cam_pred.shape[1]), int(pts3d_cam_pred.shape[2])
            depth_key = (
                frame_id,
                agent_id,
                int(height),
                int(width),
                float(vertical_fov_deg),
                float(elevation_center_deg),
                float(azimuth_offset_rad),
            )
            if depth_key not in lidar_depth_cache:
                lidar_depth_cache[depth_key] = _lidar_pcd_to_cyl_depthmap(
                    pts_carla,
                    width=width,
                    height=height,
                    vertical_fov_deg=float(vertical_fov_deg),
                    elevation_center_deg=float(elevation_center_deg),
                    azimuth_offset_rad=float(azimuth_offset_rad),
                )
            lidar_depth = lidar_depth_cache[depth_key]
            # Deterministic ray directions for the cylindrical panorama are already in the dataset.
            ray_dirs_cam = view.get("ray_directions_cam")
            if ray_dirs_cam is None or ray_dirs_cam.ndim != 4 or ray_dirs_cam.shape[-1] != 3:
                continue
            depth_t = torch.from_numpy(lidar_depth).to(device=pts3d_cam_pred.device, dtype=pts3d_cam_pred.dtype)
            pts3d_cam_gt = depth_t[None, ..., None] * ray_dirs_cam.to(dtype=pts3d_cam_pred.dtype)
            valid_mask = depth_t[None] > 0
        else:
            raise ValueError(f"Unknown supervision '{supervision}' (expected: dataset_depth|lidar)")

        mask = valid_mask & torch.isfinite(pts3d_cam_gt).all(dim=-1) & torch.isfinite(pts3d_cam_pred).all(dim=-1)
        mask = mask.reshape(-1)
        if int(mask.sum().item()) == 0:
            continue

        pred_pts.append(pts3d_cam_pred.reshape(-1, 3)[mask])
        gt_pts.append(pts3d_cam_gt.reshape(-1, 3)[mask])

    if not pred_pts:
        raise RuntimeError("No valid pts3d_cam supervision found in batch.")
    pred_all = torch.cat(pred_pts, dim=0)
    gt_all = torch.cat(gt_pts, dim=0)
    return torch.nn.functional.smooth_l1_loss(pred_all, gt_all, reduction="mean", beta=0.5)


def _mask_logits_loss_from_lidar(
    preds: Sequence[dict],
    views: Sequence[dict],
    *,
    lidar_root: Path,
    split: str,
    sequence: str,
    vertical_fov_deg: float,
    elevation_center_deg: float,
    azimuth_offset_rad: float,
    lidar_cache: dict[tuple[str, str], np.ndarray],
    lidar_depth_cache: dict[tuple[str, str, int, int, float, float, float], np.ndarray],
    pos_weight_max: float,
) -> torch.Tensor:
    """Supervise `non_ambiguous_mask_logits` using LiDAR occupancy in panorama pixels."""

    losses = []
    for pred, view in zip(preds, views):
        logits = pred.get("non_ambiguous_mask_logits")
        if logits is None or not torch.is_tensor(logits) or logits.ndim != 3:
            continue

        agent_id = _first_scalar(view.get("agent_id"))
        instance = _first_scalar(view.get("instance"))
        frame_id = str(instance).split("/")[0] if instance else ""
        if not agent_id or not frame_id:
            continue

        key = (frame_id, agent_id)
        if key not in lidar_cache:
            pcd_path = lidar_root / split / sequence / agent_id / f"{frame_id}.pcd"
            if not pcd_path.is_file():
                continue
            lidar_cache[key] = _load_ascii_pcd_xyz(pcd_path)

        pts_carla = lidar_cache[key]
        height, width = int(logits.shape[1]), int(logits.shape[2])
        depth_key = (
            frame_id,
            agent_id,
            int(height),
            int(width),
            float(vertical_fov_deg),
            float(elevation_center_deg),
            float(azimuth_offset_rad),
        )
        if depth_key not in lidar_depth_cache:
            lidar_depth_cache[depth_key] = _lidar_pcd_to_cyl_depthmap(
                pts_carla,
                width=width,
                height=height,
                vertical_fov_deg=float(vertical_fov_deg),
                elevation_center_deg=float(elevation_center_deg),
                azimuth_offset_rad=float(azimuth_offset_rad),
            )
        lidar_depth = lidar_depth_cache[depth_key]
        gt_mask = torch.from_numpy(lidar_depth > 0).to(device=logits.device)  # (H,W) bool
        gt_mask = gt_mask[None].to(dtype=logits.dtype)  # (1,H,W) float

        pos = float(gt_mask.sum().detach().cpu().item())
        neg = float(gt_mask.numel()) - pos
        if pos <= 0:
            continue
        pos_weight = min(max(neg / max(pos, 1e-6), 1.0), float(pos_weight_max))
        pos_weight_t = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)

        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                gt_mask,
                pos_weight=pos_weight_t,
                reduction="mean",
            )
        )

    if not losses:
        raise RuntimeError("No valid mask supervision found in batch.")
    return torch.mean(torch.stack(losses, dim=0))


def _has_nonfinite(preds: Sequence[dict]) -> bool:
    for pred in preds:
        for key in ("cam_quats", "cam_trans", "pts3d_cam", "pts3d"):
            val = pred.get(key)
            if val is None:
                continue
            if not torch.is_tensor(val):
                continue
            if not torch.isfinite(val).all():
                return True
    return False


def _set_requires_grad(model: torch.nn.Module, enabled_modules: Iterable[str]) -> None:
    enabled = tuple(enabled_modules)
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name for key in enabled)


def _set_train_mode_for_modules(model: torch.nn.Module, *, train_module_names: Iterable[str]) -> None:
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
    parser.add_argument("--frames", type=str, required=True, help="Comma-separated frame ids (e.g. 000107,000109)")
    parser.add_argument("--main_agent", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=10.0, help="0 or negative disables clipping")
    parser.add_argument("--abort_on_nonfinite", action="store_true")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If >0, overwrite --out_checkpoint every N steps (helps when long runs get interrupted).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w_pts_cam", type=float, default=1.0)
    parser.add_argument("--w_pose", type=float, default=0.2)
    parser.add_argument("--w_mask", type=float, default=0.2, help="LiDAR-only: mask BCE weight")
    parser.add_argument("--mask_pos_weight_max", type=float, default=50.0)
    parser.add_argument(
        "--pts_supervision",
        type=str,
        default="lidar",
        choices=("dataset_depth", "lidar"),
        help="Use GT cylindrical depth panoramas (dataset_depth) or sparse LiDAR-projected supervision (lidar).",
    )
    parser.add_argument("--lidar_root", type=Path, default=Path("data/opv2v"))
    parser.add_argument("--panorama_vertical_fov_deg", type=float, default=90.0)
    parser.add_argument("--panorama_elevation_center_deg", type=float, default=0.0)
    parser.add_argument("--panorama_azimuth_offset_deg", type=float, default=0.0)
    parser.add_argument(
        "--use_torch_hub",
        action="store_true",
        help="Allow loading encoder weights via torch hub if missing from checkpoint.",
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
    parser.add_argument(
        "--train_modules",
        type=str,
        default="dpt_feature_head,dpt_regressor_head,ray_dirs_encoder,depth_encoder,depth_scale_encoder,cam_trans_scale_encoder,pose_head,pose_adaptor,scale_head,scale_adaptor",
        help="Comma-separated substrings of parameter/module names to finetune.",
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

    train_keys = _parse_csv_list(args.train_modules)
    _set_train_mode_for_modules(model, train_module_names=train_keys)
    _set_requires_grad(model, enabled_modules=train_keys)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected; check --train_modules filters.")

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
        panorama_vertical_fov_deg=float(args.panorama_vertical_fov_deg),
        panorama_elevation_center_deg=float(args.panorama_elevation_center_deg),
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

    lidar_cache: dict[tuple[str, str], np.ndarray] = {}
    lidar_depth_cache: dict[tuple[str, str, int, int, float, float, float], np.ndarray] = {}
    lidar_root = args.lidar_root
    if not lidar_root.is_absolute():
        lidar_root = (REPO_ROOT / lidar_root).resolve()

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

        for view in batch_views:
            for key, value in view.items():
                if isinstance(value, torch.Tensor):
                    view[key] = value.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(batch_views)
        if _has_nonfinite(preds):
            msg = f"[step {step:04d}] detected non-finite outputs"
            if args.abort_on_nonfinite:
                raise RuntimeError(msg)
            print(msg + "; skipping step")
            continue

        loss = 0.0
        loss_terms = {}

        if float(args.w_pts_cam) > 0:
            l_pts = _pts3d_cam_loss(
                preds,
                batch_views,
                supervision=str(args.pts_supervision),
                lidar_root=lidar_root,
                split=str(args.split),
                sequence=str(args.sequence),
                vertical_fov_deg=float(args.panorama_vertical_fov_deg),
                elevation_center_deg=float(args.panorama_elevation_center_deg),
                azimuth_offset_rad=float(np.deg2rad(float(args.panorama_azimuth_offset_deg))),
                lidar_cache=lidar_cache,
                lidar_depth_cache=lidar_depth_cache,
            )
            loss_terms["pts3d_cam"] = float(l_pts.detach().cpu().item())
            loss = loss + float(args.w_pts_cam) * l_pts

        if float(args.w_mask) > 0 and str(args.pts_supervision) == "lidar":
            l_mask = _mask_logits_loss_from_lidar(
                preds,
                batch_views,
                lidar_root=lidar_root,
                split=str(args.split),
                sequence=str(args.sequence),
                vertical_fov_deg=float(args.panorama_vertical_fov_deg),
                elevation_center_deg=float(args.panorama_elevation_center_deg),
                azimuth_offset_rad=float(np.deg2rad(float(args.panorama_azimuth_offset_deg))),
                lidar_cache=lidar_cache,
                lidar_depth_cache=lidar_depth_cache,
                pos_weight_max=float(args.mask_pos_weight_max),
            )
            loss_terms["mask_bce"] = float(l_mask.detach().cpu().item())
            loss = loss + float(args.w_mask) * l_mask

        if float(args.w_pose) > 0:
            l_pose = _pose_loss_rel_to_view0(preds, batch_views)
            loss_terms["pose_rel"] = float(l_pose.detach().cpu().item())
            loss = loss + float(args.w_pose) * l_pose

        if not torch.is_tensor(loss) or not torch.isfinite(loss):
            print(f"[step {step:04d}] loss is not finite ({loss}); skipping step")
            continue

        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip))
        optimizer.step()

        if step % 20 == 0 or step == int(args.steps) - 1:
            terms = ", ".join(f"{k}={v:.6f}" for k, v in loss_terms.items())
            print(f"[step {step:04d}] loss={loss.item():.6f} ({terms})")
        if int(args.save_every) > 0 and (step + 1) % int(args.save_every) == 0:
            _save_checkpoint(step + 1)

    _save_checkpoint(int(args.steps))
    print(f"[OK] final checkpoint at {out_path}")


if __name__ == "__main__":
    main()
