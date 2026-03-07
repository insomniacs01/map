#!/usr/bin/env python3
"""Lightweight directory tree utility for sanity-checking dataset copies."""

import argparse
from pathlib import Path


def _tree(path: Path, max_depth: int, prefix: str = "") -> None:
    if max_depth < 0:
        return
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for idx, entry in enumerate(entries):
        connector = "└── " if idx == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir() and max_depth > 0:
            extension = "    " if idx == len(entries) - 1 else "│   "
            _tree(entry, max_depth - 1, prefix + extension)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--max_depth", type=int, default=2)
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(args.path)
    print(args.path.resolve())
    _tree(args.path, args.max_depth)


if __name__ == "__main__":
    main()
