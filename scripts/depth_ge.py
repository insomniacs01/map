#!/usr/bin/env python3
"""Run MapAnything on OPV2V configs and export per-view depth/point-cloud assets."""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.opv2v_pointclouds import predictions_to_pointcloud, save_point_cloud
from mapanything.utils.opv2v_viz import load_views_from_config

logger = logging.getLogger("depth_ge")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory storing <stem>_cameraX.png")
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Single YAML config to process",
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--depth_dir",
        type=Path,
        help="Directory containing *_depth.png files for reference",
    )
    parser.add_argument(
        "--model",
        default="facebook/map-anything",
        help="Model identifier or path",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to store per-view depth/mask numpy dumps",
    )
    parser.add_argument(
        "--save_pred_dir",
        type=Path,
        help="Directory to store predicted clouds (.pcd)",
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory efficient inference",
    )
    return parser


def _iter_configs(args):
    if args.config_path:
        yield args.config_path
    if args.config_dir:
        for cfg in sorted(args.config_dir.glob("*.yaml")):
            yield cfg


def _save_depth_assets(predictions, out_dir: Path, stem: str, depth_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, pred in enumerate(predictions):
        depth = pred["depth_z"][0].detach().cpu().numpy()
        mask = pred["mask"][0].detach().cpu().numpy()
        np.save(out_dir / f"{stem}_view{idx}_depth.npy", depth)
        np.save(out_dir / f"{stem}_view{idx}_mask.npy", mask)
        if depth_dir is not None:
            src = depth_dir / f"{stem}_camera{idx}_depth.png"
            if src.exists():
                shutil.copy(src, out_dir / src.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _arg_parser()
    args = parser.parse_args()

    configs = list(_iter_configs(args))
    if not configs:
        raise RuntimeError("No configs supplied")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)
    if args.depth_dir and not args.depth_dir.is_dir():
        raise FileNotFoundError(args.depth_dir)

    device = torch.device(args.device)
    model = MapAnything.from_pretrained(args.model).to(device)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pred_dir:
        args.save_pred_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        views, _ = load_views_from_config(cfg, args.image_dir, device)
        with torch.inference_mode():
            predictions = model.infer(views, memory_efficient_inference=args.memory_efficient)
        if args.output_dir:
            _save_depth_assets(predictions, args.output_dir, cfg.stem, args.depth_dir)
        if args.save_pred_dir:
            pts, _ = predictions_to_pointcloud(predictions)
            out_path = args.save_pred_dir / f"{cfg.stem}.pcd"
            save_point_cloud(out_path, pts)
            logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
