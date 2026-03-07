#!/usr/bin/env python3
"""Batch wrapper over depth/depth.py for entire directories."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd_dir", type=Path)
    parser.add_argument("yaml_dir", type=Path)
    parser.add_argument("--output_root", type=Path, default=Path("depth_outputs"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 540))
    args = parser.parse_args()

    if not args.pcd_dir.is_dir() or not args.yaml_dir.is_dir():
        raise FileNotFoundError("Input directories must exist")

    for yaml_file in sorted(args.yaml_dir.glob("*.yaml")):
        stem = yaml_file.stem
        pcd_path = args.pcd_dir / f"{stem}.pcd"
        if not pcd_path.exists():
            continue
        out_dir = args.output_root / stem
        cmd = [
            "python",
            str(Path(__file__).parent / "depth.py"),
            str(pcd_path),
            str(yaml_file),
            "--output_dir",
            str(out_dir),
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
