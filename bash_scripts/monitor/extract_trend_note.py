#!/usr/bin/env python3
"""
Extract a short trend note from summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()

    if not args.summary.is_file():
        print("")
        return
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    trend = data.get("trend") or {}
    parts = []
    for key in ("train_loss", "eval_validate_loss", "eval_test_loss"):
        v = trend.get(key)
        if isinstance(v, dict):
            direction = v.get("direction", "unknown")
            cnt = v.get("count", 0)
            parts.append(f"{key}:{direction}(n={cnt})")
    if not parts:
        print("")
        return
    print("trend=" + ",".join(parts))


if __name__ == "__main__":
    main()
