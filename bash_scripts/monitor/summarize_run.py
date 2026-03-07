#!/usr/bin/env python3
"""
Summarize a training log into JSON/Markdown for quick progress tracking.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone


METRIC_KEYS = [
    "loss",
    "depth_z_mae_m",
    "depth_z_rmse_m",
    "pose_trans_l2_m",
    "pose_rot_deg",
    "FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale",
    "scale_err_mean",
    "scale_log_err_mean",
    "scale_eq_rel_err_mean",
    "scale_ratio_mean",
]

# NOTE: `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` is loss-space.
# Ratio-space scale diagnostics (`scale_err_mean`, etc.) are more human-friendly.


def _parse_metrics(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        m = re.search(rf"{re.escape(key)}:\s*([-0-9.eE]+)", line)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                continue
    return out


def _epoch_from_line(line: str) -> int | None:
    m = re.search(r"Epoch:\s*\[(\d+)\]", line)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def summarize(log_path: Path) -> dict:
    data = {
        "log": str(log_path),
        "utc_summary_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "train_last": {},
        "eval_last": {},
        "last_checkpoint": None,
        "trend": {},
    }

    eval_tag = None
    if not log_path.is_file():
        return data

    train_losses: list[float] = []
    eval_losses: dict[str, list[float]] = {}

    for line in log_path.read_text(errors="ignore").splitlines():
        if "Testing on" in line:
            m = re.search(r"@ .*:([a-zA-Z_]+)", line)
            if m:
                eval_tag = m.group(1)
        if ">> Saving model to" in line:
            m = re.search(r">> Saving model to\s+(.*)", line)
            if m:
                data["last_checkpoint"] = m.group(1).strip()
        if "Test Epoch:" in line and eval_tag:
            metrics = _parse_metrics(line)
            if metrics:
                data["eval_last"][eval_tag] = {
                    "epoch": _epoch_from_line(line),
                    "metrics": metrics,
                }
                if "loss" in metrics:
                    eval_losses.setdefault(eval_tag, []).append(metrics["loss"])
        elif "Epoch:" in line and "Test Epoch" not in line:
            metrics = _parse_metrics(line)
            if metrics:
                data["train_last"] = {
                    "epoch": _epoch_from_line(line),
                    "metrics": metrics,
                }
                if "loss" in metrics:
                    train_losses.append(metrics["loss"])
    return data


def _trend_from_series(values: list[float], tol: float = 0.02) -> dict:
    if len(values) < 2:
        return {"direction": "unknown", "count": len(values)}
    first = float(values[0])
    last = float(values[-1])
    if first == 0:
        pct = 0.0
    else:
        pct = (last - first) / abs(first)
    if last <= first * (1 - tol):
        direction = "improving"
    elif last >= first * (1 + tol):
        direction = "worsening"
    else:
        direction = "flat"
    return {
        "direction": direction,
        "count": len(values),
        "first": first,
        "last": last,
        "delta": last - first,
        "pct": pct,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--exit_status", type=int, default=0)
    p.add_argument("--cmd", type=str, default="")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = summarize(args.log)
    # Populate trend info by rescanning minimal info from the same log.
    try:
        lines = args.log.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        lines = []
    train_losses: list[float] = []
    eval_losses: dict[str, list[float]] = {}
    eval_tag = None
    for line in lines:
        if "Testing on" in line:
            m = re.search(r"@ .*:([a-zA-Z_]+)", line)
            if m:
                eval_tag = m.group(1)
        if "Test Epoch:" in line and eval_tag:
            metrics = _parse_metrics(line)
            if "loss" in metrics:
                eval_losses.setdefault(eval_tag, []).append(metrics["loss"])
        elif "Epoch:" in line and "Test Epoch" not in line:
            metrics = _parse_metrics(line)
            if "loss" in metrics:
                train_losses.append(metrics["loss"])
    data["trend"]["train_loss"] = _trend_from_series(train_losses)
    for split, vals in eval_losses.items():
        data["trend"][f"eval_{split}_loss"] = _trend_from_series(vals)
    data["exit_status"] = int(args.exit_status)
    data["cmd"] = args.cmd

    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def fmt_metrics(metrics: dict[str, float]) -> str:
        if not metrics:
            return "n/a"
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        return ", ".join(parts)

    lines = [
        "# Run Summary",
        f"- Log: {data.get('log')}",
        f"- Exit: {data.get('exit_status')}",
        f"- Cmd: {data.get('cmd')}",
        f"- Last checkpoint: {data.get('last_checkpoint')}",
    ]
    if data.get("train_last"):
        tl = data["train_last"]
        lines.append(
            f"- Train last: epoch={tl.get('epoch')} metrics: {fmt_metrics(tl.get('metrics', {}))}"
        )
    if data.get("eval_last"):
        for k, v in data["eval_last"].items():
            lines.append(
                f"- Eval {k}: epoch={v.get('epoch')} metrics: {fmt_metrics(v.get('metrics', {}))}"
            )
    if data.get("trend"):
        trend_parts = []
        for k, v in data["trend"].items():
            if isinstance(v, dict):
                direction = v.get("direction", "unknown")
                cnt = v.get("count", 0)
                trend_parts.append(f"{k}={direction} (n={cnt})")
        if trend_parts:
            lines.append("- Trend: " + ", ".join(trend_parts))

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
