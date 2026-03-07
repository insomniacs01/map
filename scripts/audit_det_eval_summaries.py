#!/usr/bin/env python3
"""Audit batch_eval det/e2e summaries for strict fairness / protocol consistency.

This is a det-specific companion to `scripts/audit_batch_eval_summaries.py`.
It focuses on the additional invariants required to compare detection AP fairly:
  - same frames contract (frames_hash_md5)
  - same det decoding config (det_decode_cfg)
  - same det head config (det_head_cfg)
  - same model_task / keep_camera_poses

It assumes inputs are `summary_<split>.json` produced by `scripts/batch_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


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
    if isinstance(v, dict):
        # Stable-ish repr for quick eyeballing in markdown tables.
        items = ", ".join(f"{k}={_as_str(v[k])}" for k in sorted(v.keys()))
        return "{" + items + "}"
    return str(v)


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
    model_arch: object
    det_head_cfg: object
    det_head_ckpt: object
    det_model_ckpt: object
    det_metrics: object
    det_decode_cfg: object


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


def load_summary(path: Path) -> SummaryInfo:
    doc = json.loads(path.read_text(encoding="utf-8"))
    meta = _get_meta(doc)
    model_paths = doc.get("model_paths") if isinstance(doc, Mapping) else None
    if not isinstance(model_paths, Mapping):
        model_paths = {}

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
    model_arch = _meta_get(meta, "model_arch")
    det_head_cfg = _meta_get(meta, "det_head_cfg")
    det_head_ckpt = _meta_get(meta, "det_head_ckpt")
    det_model_ckpt = model_paths.get("det_model", MISSING)
    det_metrics = _meta_get(meta, "det_metrics")
    det_decode_cfg = _meta_get(meta, "det_decode_cfg")

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
        model_arch=model_arch,
        det_head_cfg=det_head_cfg,
        det_head_ckpt=det_head_ckpt,
        det_model_ckpt=det_model_ckpt,
        det_metrics=det_metrics,
        det_decode_cfg=det_decode_cfg,
    )


def _md_table(rows: Sequence[Mapping[str, str]], headers: Sequence[str]) -> str:
    def esc(cell: str) -> str:
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", nargs="+", type=Path, help="Paths to batch_eval summary_*.json files")
    args = ap.parse_args(argv)

    infos = []
    for p in args.summaries:
        p = p.expanduser()
        if not p.is_file():
            print(f"[ERROR] Missing summary file: {p}", file=sys.stderr)
            return 2
        infos.append(load_summary(p))

    rows = []
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
                "det_head_cfg": _as_str(info.det_head_cfg),
                "det_head_ckpt": _as_str(info.det_head_ckpt),
                "det_model_ckpt": _as_str(info.det_model_ckpt),
                "det_metrics": _as_str(info.det_metrics),
                "det_decode_cfg": _as_str(info.det_decode_cfg),
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
        "det_head_cfg",
        "det_head_ckpt",
        "det_model_ckpt",
        "det_metrics",
        "det_decode_cfg",
        "eval_script_md5",
    ]

    print("# Det Eval Fairness Audit\n")
    print(_md_table(rows, headers))
    print("\n## Fairness Checks\n")

    core_checks = []
    core_checks.append(_check_equal("frames_hash_md5", [i.frames_hash_md5 for i in infos]))
    if any(i.frames_hash_md5_v2 is not MISSING for i in infos):
        core_checks.append(_check_equal("frames_hash_md5_v2", [i.frames_hash_md5_v2 for i in infos]))
    core_checks.append(_check_equal("sample_size", [i.sample_size for i in infos]))
    # As of 2026-02-27, batch_eval.py seeds RNG from --seed for deterministic det AP
    # (BEV rasterization may subsample points). Treat seed as part of the det protocol.
    core_checks.append(_check_equal("seed", [i.seed for i in infos]))
    core_checks.append(_check_equal("modes", [i.modes for i in infos]))
    core_checks.append(_check_equal("model_task", [i.model_task for i in infos]))
    core_checks.append(_check_equal("keep_camera_poses", [i.keep_camera_poses for i in infos]))
    core_checks.append(_check_equal("keep_main_agent_poses", [i.keep_main_agent_poses for i in infos]))
    core_checks.append(_check_equal("det_head_cfg", [i.det_head_cfg for i in infos]))
    core_checks.append(_check_equal("det_metrics", [i.det_metrics for i in infos]))
    core_checks.append(_check_equal("det_decode_cfg", [i.det_decode_cfg for i in infos]))
    core_checks.append(_check_equal("eval_script_md5", [i.eval_script_md5 for i in infos]))

    compose_checks = list(core_checks)
    compose_checks.append(_check_equal("det_head_ckpt", [i.det_head_ckpt for i in infos]))

    # Cross-model comparisons (MapAnything vs VGGT) are expected to differ in `model_arch`.
    # Treat it as a strict-tier check (same pipeline), not a protocol-tier check.
    strict_core_checks = list(core_checks)
    strict_core_checks.append(_check_equal("model_arch", [i.model_arch for i in infos]))
    strict_compose_checks = list(strict_core_checks)
    strict_compose_checks.append(_check_equal("det_head_ckpt", [i.det_head_ckpt for i in infos]))

    # Useful when comparing det-head sweeps (same det_model ckpt, different det_head_ckpt).
    det_model_check = _check_equal("det_model_ckpt", [i.det_model_ckpt for i in infos])

    print("### protocol_fair (cross-model comparable)\n")
    ok_core = True
    for ok, msg in core_checks:
        ok_core = ok_core and ok
        print(f"- {msg}")
    ok_compose = True
    for ok, msg in compose_checks:
        ok_compose = ok_compose and ok
        if msg.startswith("det_head_ckpt:"):
            print(f"- {msg}")
    print(f"- {det_model_check[1]}")

    print("\n### strict_same_pipeline (requires same model_arch)\n")
    ok_strict_core = True
    for ok, msg in strict_core_checks:
        ok_strict_core = ok_strict_core and ok
        if msg.startswith("model_arch:"):
            print(f"- {msg}")
    ok_strict_compose = True
    for ok, msg in strict_compose_checks:
        ok_strict_compose = ok_strict_compose and ok
        if msg.startswith("model_arch:") or msg.startswith("det_head_ckpt:"):
            print(f"- {msg}")

    print("\n## Overall\n")
    print(f"- protocol_fair_core (ignore det_head_ckpt): {'PASS' if ok_core else 'FAIL'}")
    print(f"- protocol_fair_compose (require det_head_ckpt): {'PASS' if ok_compose else 'FAIL'}")
    print(f"- strict_same_pipeline_core: {'PASS' if ok_strict_core else 'FAIL'}")
    print(f"- strict_same_pipeline_compose: {'PASS' if ok_strict_compose else 'FAIL'}")
    return 0 if ok_core else 1


if __name__ == "__main__":
    raise SystemExit(main())
