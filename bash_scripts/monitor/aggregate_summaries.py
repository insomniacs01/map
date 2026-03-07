#!/usr/bin/env python3
"""
Aggregate per-run summary.json files into a single Markdown progress table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_root", type=Path, required=True)
    p.add_argument("--out_md", type=Path, required=True)
    args = p.parse_args()

    summaries = list(args.summary_root.rglob("summary.json"))
    rows = []
    for path in summaries:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        train = data.get("train_last", {})
        eval_last = data.get("eval_last", {})
        val = eval_last.get("validate") or eval_last.get("val") or {}
        test = eval_last.get("test") or {}
        rows.append(
            {
                "path": str(path),
                "mtime": mtime,
                "exit": data.get("exit_status"),
                "train_epoch": train.get("epoch"),
                "train_scale": (train.get("metrics", {}) or {}).get("FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale"),
                "val_pose": (val.get("metrics", {}) or {}).get("pose_trans_l2_m"),
                "val_depth": (val.get("metrics", {}) or {}).get("depth_z_mae_m"),
                "test_pose": (test.get("metrics", {}) or {}).get("pose_trans_l2_m"),
                "test_depth": (test.get("metrics", {}) or {}).get("depth_z_mae_m"),
            }
        )

    rows.sort(key=lambda x: x["mtime"])

    lines = [
        "# Auto Queue Progress",
        "",
        "| time | exit | train_ep | train_scale | val_pose_m | val_depth_mae | test_pose_m | test_depth_mae | summary |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows[-50:]:
        lines.append(
            "| {time} | {exit} | {train_epoch} | {train_scale} | {val_pose} | {val_depth} | {test_pose} | {test_depth} | {path} |".format(
                time=r["mtime"],
                exit=_fmt(r["exit"]),
                train_epoch=_fmt(r["train_epoch"]),
                train_scale=_fmt(r["train_scale"]),
                val_pose=_fmt(r["val_pose"]),
                val_depth=_fmt(r["val_depth"]),
                test_pose=_fmt(r["test_pose"]),
                test_depth=_fmt(r["test_depth"]),
                path=r["path"],
            )
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
