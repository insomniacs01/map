#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[2]
    / "eval_runs"
    / "geom_fairlock_test500_20260226_promoted_v2"
    / "summary_test.json"
)


def _pick_model(metrics: dict, requested: str | None) -> str:
    if requested and requested in metrics:
        return requested
    if len(metrics) == 1:
        return next(iter(metrics))
    if requested:
        raise KeyError(f"model `{requested}` not found; available={sorted(metrics.keys())}")
    raise KeyError(f"multiple models present; specify one of {sorted(metrics.keys())}")


def _load_mode_metrics(path: Path, mode: str, model_name: str | None) -> tuple[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    if not metrics:
        raise KeyError(f"`metrics` missing in {path}")
    resolved_model = _pick_model(metrics, model_name)
    model_metrics = metrics.get(resolved_model) or {}
    if mode not in model_metrics:
        raise KeyError(f"mode `{mode}` missing in {path} for model `{resolved_model}`")
    return resolved_model, model_metrics[mode]


def _fmt(value) -> str:
    if value is None:
        return "missing"
    return f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate geometry readiness against a locked Test500 baseline.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(os.environ.get("GEOM_GATE_BASELINE", str(DEFAULT_BASELINE))))
    parser.add_argument("--model-name", type=str, default=os.environ.get("GEOM_GATE_MODEL_NAME", "geom_model"))
    parser.add_argument("--mode", type=str, default=os.environ.get("GEOM_GATE_MODE", "coop"))
    parser.add_argument("--scale-mult-max", type=float, default=float(os.environ.get("GEOM_GATE_SCALE_MULT_MAX", "1.30")))
    parser.add_argument("--ratio-target", type=float, default=float(os.environ.get("GEOM_GATE_RATIO_TARGET", "1.0")))
    parser.add_argument("--ratio-tol", type=float, default=float(os.environ.get("GEOM_GATE_RATIO_TOL", "0.10")))
    parser.add_argument("--guardrail-mult", type=float, default=float(os.environ.get("GEOM_GATE_GUARDRAIL_MULT", "1.10")))
    parser.add_argument("--require-ratio", action="store_true", default=os.environ.get("GEOM_GATE_REQUIRE_RATIO", "0") == "1")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    candidate_model, candidate = _load_mode_metrics(args.summary, args.mode, args.model_name)
    baseline_model, baseline = _load_mode_metrics(args.baseline, args.mode, args.model_name)

    failures: list[str] = []
    warnings: list[str] = []
    checks: list[dict] = []

    def record(name: str, candidate_value, limit, passed: bool, kind: str, note: str = "") -> None:
        checks.append(
            {
                "name": name,
                "candidate": candidate_value,
                "limit": limit,
                "passed": passed,
                "kind": kind,
                "note": note,
            }
        )

    scale_mult = candidate.get("scale_to_gt_mult_err_mean")
    scale_mult_ok = scale_mult is not None and scale_mult <= args.scale_mult_max
    if not scale_mult_ok:
        failures.append(f"scale_to_gt_mult_err_mean={_fmt(scale_mult)} > {args.scale_mult_max:.3f}")
    record("scale_to_gt_mult_err_mean", scale_mult, args.scale_mult_max, scale_mult_ok, "hard_max")

    ratio = candidate.get("scale_to_gt_ratio_mean")
    ratio_low = args.ratio_target - args.ratio_tol
    ratio_high = args.ratio_target + args.ratio_tol
    ratio_ok = ratio is not None and ratio_low <= ratio <= ratio_high
    ratio_note = f"target={args.ratio_target:.3f} tol=±{args.ratio_tol:.3f}"
    if not ratio_ok:
        message = f"scale_to_gt_ratio_mean={_fmt(ratio)} outside [{ratio_low:.3f}, {ratio_high:.3f}]"
        if args.require_ratio:
            failures.append(message)
        else:
            warnings.append(message)
    record("scale_to_gt_ratio_mean", ratio, [ratio_low, ratio_high], ratio_ok or not args.require_ratio, "report_bias", ratio_note)

    for key in ("depth_rel_mean", "cross_agent_pose_trans_mean", "cross_agent_pose_rot_mean"):
        base_value = baseline.get(key)
        limit = None if base_value is None else base_value * args.guardrail_mult
        candidate_value = candidate.get(key)
        passed = candidate_value is not None and limit is not None and candidate_value <= limit
        note = f"baseline={_fmt(base_value)} x {args.guardrail_mult:.3f}" if limit is not None else "baseline missing"
        if not passed:
            failures.append(f"{key}={_fmt(candidate_value)} > {_fmt(limit)} ({note})")
        record(key, candidate_value, limit, passed, "baseline_guardrail", note)

    overall_pass = not failures
    report = {
        "summary": str(args.summary),
        "baseline": str(args.baseline),
        "mode": args.mode,
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "overall_pass": overall_pass,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "baseline_anchor": {
            "scale_to_gt_mult_err_mean": baseline.get("scale_to_gt_mult_err_mean"),
            "scale_to_gt_ratio_mean": baseline.get("scale_to_gt_ratio_mean"),
            "depth_rel_mean": baseline.get("depth_rel_mean"),
            "cross_agent_pose_trans_mean": baseline.get("cross_agent_pose_trans_mean"),
            "cross_agent_pose_rot_mean": baseline.get("cross_agent_pose_rot_mean"),
        },
    }

    print(f"[geom-gate] summary={args.summary}")
    print(f"[geom-gate] baseline={args.baseline}")
    print(f"[geom-gate] model={candidate_model} baseline_model={baseline_model} mode={args.mode}")
    for item in checks:
        limit = item["limit"]
        if isinstance(limit, list):
            limit_text = f"[{limit[0]:.3f}, {limit[1]:.3f}]"
        elif limit is None:
            limit_text = "n/a"
        else:
            limit_text = f"{limit:.6f}"
        print(
            f"[geom-gate] {'PASS' if item['passed'] else 'FAIL'} {item['name']}: candidate={_fmt(item['candidate'])} limit={limit_text}"
            + (f" ({item['note']})" if item.get("note") else "")
        )
    for warning in warnings:
        print(f"[geom-gate] WARN {warning}")
    if failures:
        print("[geom-gate] FAIL " + "; ".join(failures))
    else:
        print("[geom-gate] PASS geometry-ready-for-det gate satisfied")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if overall_pass else 10


if __name__ == "__main__":
    sys.exit(main())
