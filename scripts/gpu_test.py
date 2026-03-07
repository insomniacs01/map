#!/usr/bin/env python3
"""Quick CUDA smoke test for MapAnything environments."""

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print("Torch version:", torch.__version__)
    print("CUDA device count:", torch.cuda.device_count())
    x = torch.randn((1024, 1024), device=device)
    y = torch.randn((1024, 1024), device=device)
    torch.cuda.synchronize()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    print("Computation succeeded with norm", z.norm().item())


if __name__ == "__main__":
    main()
