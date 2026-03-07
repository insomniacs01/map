#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ValueError(f"unsupported boolean text: {text}")


def _maybe_resolve_contract(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    if p.is_file():
        return p
    if not p.is_absolute():
        p2 = (REPO_ROOT / p).resolve()
        if p2.is_file():
            return p2
    return p


def _contract_hash_v1(frames: Sequence[Mapping[str, Any]]) -> str:
    items = sorted(
        {
            f"{str(frame.get('sequence'))}/{str(frame.get('frame'))}"
            for frame in frames
            if frame.get("sequence") is not None and frame.get("frame") is not None
        }
    )
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def _contract_hash_v2(frames: Sequence[Mapping[str, Any]]) -> str:
    items = sorted(
        {
            f"{str(frame.get('sequence'))}/{str(frame.get('frame'))}/main={str(frame.get('main_agent'))}/coop={','.join(str(v) for v in (frame.get('coop_agents') or []))}"
            for frame in frames
            if frame.get("sequence") is not None and frame.get("frame") is not None
        }
    )
    return hashlib.md5("\n".join(items).encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path)
    ap.add_argument("--expected-contract", type=Path)
    ap.add_argument("--expected-model-task")
    ap.add_argument("--expected-keep-camera-poses")
    ap.add_argument("--expected-keep-main-agent-poses")
    ap.add_argument("--require-det", action="store_true")
    ap.add_argument("--require-rerun-file", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args(argv)

    failures: list[str] = []
    summary = _load_json(args.summary)
    meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}

    required_meta = [
        "frames_json",
        "frames_hash_md5",
        "frames_hash_md5_v2",
        "eval_script_md5",
        "model_task",
        "keep_camera_poses",
        "keep_main_agent_poses",
    ]
    for key in required_meta:
        if meta.get(key) in {None, ""}:
            failures.append(f"missing meta.{key}")

    if len(str(meta.get("eval_script_md5", ""))) != 32:
        failures.append("meta.eval_script_md5 is not a 32-char md5")

    expected_contract = args.expected_contract
    if expected_contract is None:
        expected_contract = _maybe_resolve_contract(meta.get("frames_json"))
    contract_doc = None
    contract_hash_v1 = None
    contract_hash_v2 = None
    if expected_contract is not None and expected_contract.is_file():
        contract_doc = _load_json(expected_contract)
        contract_frames = contract_doc.get("frames") if isinstance(contract_doc.get("frames"), list) else []
        contract_hash_v1 = contract_doc.get("frames_hash_md5_v1") or contract_doc.get("frames_hash_md5") or _contract_hash_v1(contract_frames)
        contract_hash_v2 = contract_doc.get("frames_hash_md5_v2") or _contract_hash_v2(contract_frames)
        if meta.get("frames_hash_md5") != contract_hash_v1:
            failures.append("meta.frames_hash_md5 does not match contract frames_hash_md5_v1")
        if meta.get("frames_hash_md5_v2") != contract_hash_v2:
            failures.append("meta.frames_hash_md5_v2 does not match contract frames_hash_md5_v2")
        if int(meta.get("sample_size", -1)) != len(contract_frames):
            failures.append("meta.sample_size does not match contract frame count")
        if len(summary.get("frames", [])) != len(contract_frames):
            failures.append("summary.frames length does not match contract frame count")
    elif expected_contract is not None:
        failures.append(f"expected contract not found: {expected_contract}")

    if args.expected_model_task and meta.get("model_task") != args.expected_model_task:
        failures.append(f"model_task mismatch: expected {args.expected_model_task} got {meta.get('model_task')}")

    expected_keep_camera_poses = _resolve_bool(args.expected_keep_camera_poses)
    if expected_keep_camera_poses is not None and bool(meta.get("keep_camera_poses")) != expected_keep_camera_poses:
        failures.append(
            f"keep_camera_poses mismatch: expected {expected_keep_camera_poses} got {meta.get('keep_camera_poses')}"
        )

    expected_keep_main_agent_poses = _resolve_bool(args.expected_keep_main_agent_poses)
    if expected_keep_main_agent_poses is not None and bool(meta.get("keep_main_agent_poses")) != expected_keep_main_agent_poses:
        failures.append(
            f"keep_main_agent_poses mismatch: expected {expected_keep_main_agent_poses} got {meta.get('keep_main_agent_poses')}"
        )

    if args.require_det:
        if not bool(meta.get("det_metrics")):
            failures.append("meta.det_metrics is not true")
        det_decode_cfg = meta.get("det_decode_cfg")
        if not isinstance(det_decode_cfg, dict):
            failures.append("meta.det_decode_cfg missing for det summary")
        if not metrics:
            failures.append("metrics block missing")

    if args.require_rerun_file and not args.require_rerun_file.is_file():
        failures.append(f"missing rerun file: {args.require_rerun_file}")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Batch Eval Summary Protocol Validation",
        "",
        f"- Summary: `{args.summary}`",
        f"- Expected contract: `{expected_contract}`" if expected_contract else "- Expected contract: auto/none",
        f"- Status: **{status}**",
        "",
        "## Checks",
        "",
        f"- meta.frames_json: `{meta.get('frames_json')}`",
        f"- meta.frames_hash_md5_v2: `{meta.get('frames_hash_md5_v2')}`",
        f"- contract.frames_hash_md5_v2: `{contract_hash_v2}`",
        f"- meta.eval_script_md5: `{meta.get('eval_script_md5')}`",
        f"- model_task: `{meta.get('model_task')}`",
        f"- keep_camera_poses: `{meta.get('keep_camera_poses')}`",
        f"- keep_main_agent_poses: `{meta.get('keep_main_agent_poses')}`",
        f"- det_metrics: `{meta.get('det_metrics')}`",
        "",
    ]
    if failures:
        lines.extend(["## Failures", ""] + [f"- {item}" for item in failures] + [""])
    else:
        lines.extend(["## Result", "", "- All requested protocol checks passed.", ""])

    report = "\n".join(lines)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(report, encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "status": status,
                    "failures": failures,
                    "summary": str(args.summary),
                    "expected_contract": str(expected_contract) if expected_contract else None,
                    "contract_hash_v1": contract_hash_v1,
                    "contract_hash_v2": contract_hash_v2,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(report)
    print(f"VALIDATE_BATCH_EVAL_SUMMARY_PROTOCOL: {status}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
