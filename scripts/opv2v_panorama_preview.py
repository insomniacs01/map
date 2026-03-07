#!/usr/bin/env python3
"""
Utility script to preview cylindrical panoramas produced by OPV2VCylindricalDataset.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.datasets.opv2v_cyl import OPV2VCylindricalDataset


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255.0).round().astype(np.uint8)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Preview OPV2V cylindrical panoramas")
    parser.add_argument("--root", type=str, required=True, help="OPV2V RGB root")
    parser.add_argument("--depth_root", type=str, required=True, help="OPV2V depth root")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize")
    parser.add_argument(
        "--output_dir", type=str, default="panorama_preview", help="Directory to save outputs"
    )
    parser.add_argument("--width", type=int, default=1008)
    parser.add_argument("--height", type=int, default=252)
    args = parser.parse_args()

    dataset = OPV2VCylindricalDataset(
        num_views=1,
        split=args.split,
        resolution=(args.width, args.height),
        transform="imgnorm",
        data_norm_type="identity",
        ROOT=args.root,
        depth_root=args.depth_root,
        panorama_resolution=(args.width, args.height),
    )

    sample = dataset[args.index]
    view = sample[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = tensor_to_uint8_image(view["img"])

    depth_np = view["depthmap"]
    if isinstance(depth_np, torch.Tensor):
        depth_np = depth_np.detach().cpu().numpy()
    depth = depth_np[..., 0]

    mask_np = view["non_ambiguous_mask"]
    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.detach().cpu().numpy()
    mask = mask_np.astype(np.uint8)

    Image.fromarray(rgb).save(output_dir / "panorama.png")
    np.save(output_dir / "panorama_depth.npy", depth)
    np.save(output_dir / "panorama_mask.npy", mask)

    print(f"Saved panorama preview to {output_dir}")


if __name__ == "__main__":
    main()
