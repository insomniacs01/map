#!/usr/bin/env python3
"""Audit batch_eval summary JSONs for strict fairness / protocol consistency.

This script consumes one or more `summary_<split>.json` files produced by
`map-anything/scripts/batch_eval.py` and prints:
  1) A markdown table of key meta fields.
  2) PASS/FAIL checks indicating whether the supplied summaries are comparable.

It is intentionally tolerant of older summaries that may lack some meta keys
("backward compatibility"): missing fields are reported explicitly and treated
as FAIL for strict fairness.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


MISSING = object()


def _as_str(v: object) -> str:
    if v is MISSING:
        return "<missing>"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_as_str(x) for x in v) + "]"
    return str(v)


def _canonical_modes(modes: object) -> object:
    if modes is MISSING:
        return MISSING
    if modes is None:
        return None
    if isinstance(modes, (list, tuple)):
        items = []
        for m in modes:
            if isinstance(m, str):
                items.append(m)
        return tuple(sorted(dict.fromkeys(items)))
    if isinstance(modes, str):
        return tuple(sorted(dict.fromkeys([modes])))
    return str(modes)


def _get_meta(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = doc.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _meta_get(meta: Mapping[str, Any], key: str) -> object:
    return meta[key] if key in meta else MISSING


@dataclass(frozen=True)
class SummaryInfo:
    path: Path
    run_name: str
    split: object
    frames_hash_md5: object
    frames_hash_md5_v2: object
    sample_size: object
    seed: object
    modes: object
    model_task: object
    keep_camera_poses: object
    keep_main_agent_poses: object
    eval_script_md5: object
    gt_scale_contract_hash_md5: object
    gt_scale_contract_hash_md5_v2: object
    gt_scale_contract_json: object
    resolved_data_norm_type: object
    model_arch: object
    vggt_enable_metric_scale_head: object


def load_summary(path: Path) -> SummaryInfo:
    doc = json.loads(path.read_text(encoding="utf-8"))
    meta = _get_meta(doc)

    split = doc.get("split", MISSING)
    run_name = path.parent.name
    frames_hash_md5 = _meta_get(meta, "frames_hash_md5")
    frames_hash_md5_v2 = _meta_get(meta, "frames_hash_md5_v2")
    sample_size = _meta_get(meta, "sample_size")
    seed = _meta_get(meta, "seed")
    modes = _canonical_modes(_meta_get(meta, "modes"))
    model_task = _meta_get(meta, "model_task")
    keep_camera_poses = _meta_get(meta, "keep_camera_poses")
    keep_main_agent_poses = _meta_get(meta, "keep_main_agent_poses")
    eval_script_md5 = _meta_get(meta, "eval_script_md5")
    gt_scale_contract_hash_md5 = _meta_get(meta, "gt_scale_contract_hash_md5")
    gt_scale_contract_hash_md5_v2 = _meta_get(meta, "gt_scale_contract_hash_md5_v2")
    gt_scale_contract_json = _meta_get(meta, "gt_scale_contract_json")

    # Prefer the resolved value (added for fairness audits); fall back to the raw CLI override.
    resolved_data_norm_type = _meta_get(meta, "resolved_data_norm_type")
    if resolved_data_norm_type is MISSING:
        resolved_data_norm_type = _meta_get(meta, "data_norm_type")

    model_arch = _meta_get(meta, "model_arch")
    vggt_enable_metric_scale_head = _meta_get(meta, "vggt_enable_metric_scale_head")

    return SummaryInfo(
        path=path,
        run_name=run_name,
        split=split,
        frames_hash_md5=frames_hash_md5,
        frames_hash_md5_v2=frames_hash_md5_v2,
        sample_size=sample_size,
        seed=seed,
        modes=modes,
        model_task=model_task,
        keep_camera_poses=keep_camera_poses,
        keep_main_agent_poses=keep_main_agent_poses,
        eval_script_md5=eval_script_md5,
        gt_scale_contract_hash_md5=gt_scale_contract_hash_md5,
        gt_scale_contract_hash_md5_v2=gt_scale_contract_hash_md5_v2,
        gt_scale_contract_json=gt_scale_contract_json,
        resolved_data_norm_type=resolved_data_norm_type,
        model_arch=model_arch,
        vggt_enable_metric_scale_head=vggt_enable_metric_scale_head,
    )


def _md_table(rows: Sequence[Mapping[str, str]], headers: Sequence[str]) -> str:
    def esc(cell: str) -> str:
        # Minimal escaping for markdown tables.
        return cell.replace("|", "\\|")

    out = []
    out.append("| " + " | ".join(esc(h) for h in headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(esc(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def _check_equal(label: str, values: Sequence[object]) -> Tuple[bool, str]:
    uniq = []
    for v in values:
        if not any(v == u for u in uniq):
            uniq.append(v)
    ok = len(uniq) == 1 and uniq[0] is not MISSING
    return ok, f"{label}: {'PASS' if ok else 'FAIL'} ({', '.join(_as_str(v) for v in uniq)})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="Paths to batch_eval summary_*.json files")
    args = parser.parse_args(argv)

    infos: List[SummaryInfo] = []
    for p in args.summaries:
        p = p.expanduser()
        if not p.is_file():
            print(f"[ERROR] Missing summary file: {p}", file=sys.stderr)
            return 2
        infos.append(load_summary(p))

    # Markdown table (one row per summary).
    rows: List[Dict[str, str]] = []
    for info in infos:
        rows.append(
            {
                "run": info.run_name,
                "summary": str(info.path),
                "split": _as_str(info.split),
                "sample_size": _as_str(info.sample_size),
                "seed": _as_str(info.seed),
                "modes": _as_str(info.modes),
                "frames_hash_md5": _as_str(info.frames_hash_md5),
                "frames_hash_md5_v2": _as_str(info.frames_hash_md5_v2),
                "model_task": _as_str(info.model_task),
                "keep_camera_poses": _as_str(info.keep_camera_poses),
                "keep_main_agent_poses": _as_str(info.keep_main_agent_poses),
                "model_arch": _as_str(info.model_arch),
                "data_norm_type": _as_str(info.resolved_data_norm_type),
                "vggt_scale_head": _as_str(info.vggt_enable_metric_scale_head),
                "gt_scale_contract_hash": _as_str(info.gt_scale_contract_hash_md5),
                "gt_scale_contract_hash_v2": _as_str(info.gt_scale_contract_hash_md5_v2),
                "eval_script_md5": _as_str(info.eval_script_md5),
            }
        )

    headers = [
        "run",
        "summary",
        "split",
        "sample_size",
        "seed",
        "modes",
        "frames_hash_md5",
        "frames_hash_md5_v2",
        "model_task",
        "keep_camera_poses",
        "keep_main_agent_poses",
        "model_arch",
        "data_norm_type",
        "vggt_scale_head",
        "gt_scale_contract_hash",
        "gt_scale_contract_hash_v2",
        "eval_script_md5",
    ]

    print("# Batch Eval Fairness Audit\n")
    print(_md_table(rows, headers))
    print("\n## Fairness Checks\n")

    # NOTE: Comparing different model families (MapAnything vs VGGT) is expected to
    # differ in `model_arch`, `data_norm_type` (defaults), and possibly VGGT flags.
    # We therefore split the checks into:
    #   - protocol_fair: shared contract/protocol invariants must match.
    #   - strict_same_pipeline: additionally requires same arch/norm/flags.

    protocol_checks: List[Tuple[bool, str]] = []
    protocol_checks.append(_check_equal("frames_hash_md5", [i.frames_hash_md5 for i in infos]))
    if any(i.frames_hash_md5_v2 is not MISSING for i in infos):
        protocol_checks.append(_check_equal("frames_hash_md5_v2", [i.frames_hash_md5_v2 for i in infos]))
    protocol_checks.append(_check_equal("sample_size", [i.sample_size for i in infos]))
    protocol_checks.append(_check_equal("seed", [i.seed for i in infos]))
    protocol_checks.append(_check_equal("modes", [i.modes for i in infos]))
    protocol_checks.append(_check_equal("model_task", [i.model_task for i in infos]))
    protocol_checks.append(_check_equal("keep_camera_poses", [i.keep_camera_poses for i in infos]))
    protocol_checks.append(_check_equal("keep_main_agent_poses", [i.keep_main_agent_poses for i in infos]))
    protocol_checks.append(_check_equal("eval_script_md5", [i.eval_script_md5 for i in infos]))
    protocol_checks.append(_check_equal("gt_scale_contract_hash_md5", [i.gt_scale_contract_hash_md5 for i in infos]))
    if any(i.gt_scale_contract_hash_md5_v2 is not MISSING for i in infos):
        protocol_checks.append(
            _check_equal(
                "gt_scale_contract_hash_md5_v2",
                [i.gt_scale_contract_hash_md5_v2 for i in infos],
            )
        )

    strict_pipeline_checks: List[Tuple[bool, str]] = list(protocol_checks)
    strict_pipeline_checks.append(_check_equal("model_arch", [i.model_arch for i in infos]))
    strict_pipeline_checks.append(_check_equal("data_norm_type", [i.resolved_data_norm_type for i in infos]))
    strict_pipeline_checks.append(
        _check_equal(
            "vggt_enable_metric_scale_head",
            [i.vggt_enable_metric_scale_head for i in infos],
        )
    )

    print("### protocol_fair\n")
    protocol_ok = True
    for ok, msg in protocol_checks:
        protocol_ok = protocol_ok and ok
        print(f"- {msg}")

    print("\n### strict_same_pipeline (additional)\n")
    # Only print the extra checks; the protocol checks are shared.
    extra_checks = strict_pipeline_checks[len(protocol_checks) :]
    for _, msg in extra_checks:
        print(f"- {msg}")

    strict_pipeline_ok = True
    for ok, _ in strict_pipeline_checks:
        strict_pipeline_ok = strict_pipeline_ok and ok

    print("\n## Overall\n")
    print(f"- protocol_fair: {'PASS' if protocol_ok else 'FAIL'}")
    print(f"- strict_same_pipeline: {'PASS' if strict_pipeline_ok else 'FAIL'}")
    return 0 if protocol_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
