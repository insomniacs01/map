#!/usr/bin/env python3
"""
Auto-tune GATE_MAX thresholds based on recent summary metrics.

Behavior:
- If metrics pass current GATE_MAX, reset fail_count.
- If metrics fail for N runs, relax thresholds by a factor (bounded by caps).

Env:
  GATE_TUNE_FAILS_BEFORE_RELAX (default: 2)
  GATE_TUNE_RELAX_FACTOR       (default: 1.1)
  GATE_TUNE_MAX_CAPS           (e.g., "depth_z_mae_m=3,pose_trans_l2_m=1.0")
  GATE_SPLIT                   (default: validate,test,train)
  GATE_TUNE_STATE              (default: <gate>.state.json)

Scale metric convention:
  - If `GATE_MAX` includes `scale_err_mean`, that key is ratio-space (smaller toward 0).
  - `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` is loss-space (optimization only).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_kv(s: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            continue
    return out


def _select_metrics(summary: dict, split_order: list[str]) -> tuple[str, dict]:
    eval_last = summary.get("eval_last") or {}
    for key in split_order:
        if key in eval_last and "metrics" in eval_last[key]:
            return key, eval_last[key]["metrics"]
    train_last = summary.get("train_last") or {}
    if "metrics" in train_last:
        return "train", train_last["metrics"]
    return "none", {}


def _load_gate_max(gate_text: str) -> tuple[dict[str, float], str]:
    for line in gate_text.splitlines():
        if "GATE_MAX" in line and "=" in line:
            raw = line.split("=", 1)[1].strip()
            raw = raw.strip('"').strip("'")
            return _parse_kv(raw), line
    return {}, ""


def _write_gate_max(gate_text: str, new_max: dict[str, float]) -> str:
    parts = [f"{k}={new_max[k]:.4f}".rstrip("0").rstrip(".") for k in sorted(new_max.keys())]
    new_line = f'export GATE_MAX="{",".join(parts)}"'
    lines = gate_text.splitlines()
    out_lines = []
    replaced = False
    for line in lines:
        if "GATE_MAX" in line and "=" in line:
            out_lines.append(new_line)
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append(new_line)
    return "\n".join(out_lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    args = p.parse_args()

    if not args.summary.is_file() or not args.gate.is_file():
        return 0

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    gate_text = args.gate.read_text(encoding="utf-8")

    gate_max, _ = _load_gate_max(gate_text)
    if not gate_max:
        return 0

    split_order = [s.strip() for s in os.environ.get("GATE_SPLIT", "validate,test,train").split(",") if s.strip()]
    _, metrics = _select_metrics(data, split_order)
    if not metrics:
        return 0

    state_path = Path(os.environ.get("GATE_TUNE_STATE", str(args.gate) + ".state.json"))
    state = {"fail_count": 0}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"fail_count": 0}

    fail = False
    for k, thr in gate_max.items():
        if k not in metrics:
            continue
        if metrics[k] > thr:
            fail = True
            break

    if not fail:
        state["fail_count"] = 0
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return 0

    fail_count = int(state.get("fail_count", 0)) + 1
    state["fail_count"] = fail_count
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    relax_after = int(os.environ.get("GATE_TUNE_FAILS_BEFORE_RELAX", "2"))
    if fail_count < relax_after:
        return 0

    relax_factor = float(os.environ.get("GATE_TUNE_RELAX_FACTOR", "1.1"))
    caps = _parse_kv(os.environ.get("GATE_TUNE_MAX_CAPS"))
    new_max = dict(gate_max)
    updated = False
    for k, thr in gate_max.items():
        if k not in metrics:
            continue
        if metrics[k] <= thr:
            continue
        proposal = max(thr * relax_factor, metrics[k] * relax_factor)
        cap = caps.get(k, None)
        if cap is not None:
            proposal = min(proposal, cap)
        if proposal > thr:
            new_max[k] = proposal
            updated = True

    if updated:
        args.gate.write_text(_write_gate_max(gate_text, new_max), encoding="utf-8")
        # reset fail_count after relaxing
        state["fail_count"] = 0
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
