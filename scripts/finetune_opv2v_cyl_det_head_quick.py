#!/usr/bin/env python3
"""
Quick detection head finetune for OPV2V cylindrical cooperative scenes.

Trains ONLY the BEV detection head on a small set of frames so that predicted
boxes align with OPV2V GT vehicle boxes. Intended for fast debugging on a
specific sequence (not full-scale training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import yaml

from mapanything.datasets.opv2v_cyl import OPV2VCoopCylindricalDataset
from mapanything.train.losses import BEVCenternetDetLoss
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _load_frames_from_summary(path: Path) -> List[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("frames", payload)
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a list under 'frames'")
    specs: List[tuple[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        seq = item.get("sequence")
        frame = item.get("frame")
        if seq is None or frame is None:
            continue
        specs.append((str(seq), str(frame)))
    if not specs:
        raise ValueError(f"No valid frame specs found in {path}")
    return specs


def _build_det_loss(loss_cfg_name: str) -> BEVCenternetDetLoss:
    cfg_path = REPO_ROOT / "configs" / "loss" / f"{loss_cfg_name}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    loss_expr = data.get("train_criterion")
    if not loss_expr:
        raise ValueError(f"No train_criterion found in {cfg_path}")
    loss_fn = eval(loss_expr, {"BEVCenternetDetLoss": BEVCenternetDetLoss})
    if not isinstance(loss_fn, BEVCenternetDetLoss):
        raise TypeError(f"Expected BEVCenternetDetLoss from {loss_cfg_name}, got {type(loss_fn)}")
    return loss_fn


def _move_views_to_device(views, device: torch.device) -> None:
    for view in views:
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                view[key] = value.to(device)


def _set_det_head_trainable(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "det_head") or model.det_head is None:
        raise RuntimeError("Model was initialized without det_head enabled.")
    for param in model.det_head.parameters():
        param.requires_grad = True
    model.eval()
    model.det_head.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint", type=Path, required=True)
    parser.add_argument("--out_checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated frame ids (e.g. 000107,000109)")
    parser.add_argument("--frames_json", type=Path, default=None, help="JSON with frame list (same format as batch_eval summary)")
    parser.add_argument("--main_agent", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=10.0, help="0 or negative disables clipping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--det_head_cfg", type=str, default="bev_centernet_wide_v7_gate_noconf")
    parser.add_argument("--loss_cfg", type=str, default="opv2v_det_only_wide_v5")
    parser.add_argument("--task", type=str, default="images_only")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument(
        "--max_agent_distance",
        type=float,
        default=None,
        help="Only sample cooperative agents whose GT ego distance to main is <= this XY radius (meters).",
    )
    parser.add_argument(
        "--agent_selection_policy",
        type=str,
        default="nearest",
        choices=("random", "nearest"),
    )
    parser.add_argument("--panorama_vertical_fov_deg", type=float, default=90.0)
    parser.add_argument("--panorama_elevation_center_deg", type=float, default=0.0)
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
            "model=mapanything_det",
            f"model/det_head={args.det_head_cfg}",
            f"model.encoder.uses_torch_hub={'true' if args.use_torch_hub else 'false'}",
            f"model/task={args.task}",
        ],
        "strict": False,
    }
    model = initialize_mapanything_local(local_cfg, device)

    _set_det_head_trainable(model)

    det_loss = _build_det_loss(args.loss_cfg).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.det_head.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    frame_list = _parse_csv_list(args.frames)
    frame_specs = None
    if args.frames_json is not None:
        summary_path = (REPO_ROOT / args.frames_json).resolve() if not args.frames_json.is_absolute() else args.frames_json
        frame_specs = _load_frames_from_summary(summary_path)
    if not frame_list and not frame_specs:
        raise ValueError("Provide --frames or --frames_json")

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
        include_vehicle_boxes=True,
        max_num_boxes=128,
        bbox_range=120.0,
        min_agents=int(args.num_views),
        min_num_views=int(args.num_views),
        max_num_views=int(args.num_views),
        panorama_resolution=(1008, 252),
        panorama_vertical_fov_deg=float(args.panorama_vertical_fov_deg),
        panorama_elevation_center_deg=float(args.panorama_elevation_center_deg),
        view_selection_margin=0.05,
        max_num_retries=0,
        main_agent=str(args.main_agent) if args.main_agent else None,
        ensure_main_agent_first=True,
        max_agent_distance=float(args.max_agent_distance) if args.max_agent_distance is not None else None,
        agent_selection_policy=str(args.agent_selection_policy),
    )

    wanted: List[int] = []
    frames_set = set(frame_list)
    allowed = {(seq, frame) for seq, frame in frame_specs} if frame_specs else None
    for idx, scene in enumerate(dataset.scenes):
        scene_seq = str(scene.get("sequence"))
        scene_frame = str(scene.get("frame"))
        if allowed is not None:
            if (scene_seq, scene_frame) not in allowed:
                continue
        else:
            if args.sequence is None:
                continue
            if scene_seq != str(args.sequence):
                continue
            if frames_set and scene_frame not in frames_set:
                continue
        if args.main_agent:
            if str(args.main_agent) not in list(scene.get("agents") or []):
                continue
        wanted.append(int(idx))
    if not wanted:
        raise RuntimeError("No scenes found for the provided frame selection")

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

    log_every = max(1, int(args.log_every))
    for step in range(int(args.steps)):
        try:
            batch_views = next(it)
        except StopIteration:
            it = iter(loader)
            batch_views = next(it)

        _move_views_to_device(batch_views, device)
        preds = model(batch_views)

        loss, _ = det_loss(batch_views, preds)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite det loss at step {step}: {loss}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.det_head.parameters(), float(args.grad_clip))
        optimizer.step()

        if step % log_every == 0 or step == int(args.steps) - 1:
            print(f"[train] step={step:04d} loss={float(loss.detach()):.4f}")

        if args.save_every and int(args.save_every) > 0:
            if (step + 1) % int(args.save_every) == 0:
                _save_checkpoint(step + 1)

    _save_checkpoint(int(args.steps))


if __name__ == "__main__":
    main()
