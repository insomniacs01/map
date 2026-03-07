#!/usr/bin/env python3
"""Compare det/e2e `summary_test.json` across multiple eval runs (no inference).

This is the det counterpart of `compare_geom_bucket_suite.py`:
- reads existing `summary_test.json` files
- checks protocol fairness invariants (frames hash / decode cfg / seed / etc.)
- prints a compact metrics table (AP/precision/recall for single+coop)

Typical usage (mainlines, predpose composed det head):
  python map-anything/scripts/compare_det_suite.py \
    --eval_roots \
      map-anything/eval_runs/<runA> \
      map-anything/eval_runs/<runB> \
    --out_md map-anything/eval_runs/<suite>/compare.md
"""

from __future__ import annotations

import argparse
import json
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
    if isinstance(v, dict):
        # Stable-ish repr for markdown tables / diffs.
        items = ", ".join(f"{k}={_as_str(v[k])}" for k in sorted(v.keys()))
        return "{" + items + "}"
    return str(v)


def _get_meta(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = doc.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _meta_get(meta: Mapping[str, Any], key: str) -> object:
    return meta[key] if key in meta else MISSING


def _canonical_modes(modes: object) -> object:
    if modes is MISSING:
        return MISSING
    if modes is None:
        return None
    if isinstance(modes, (list, tuple)):
        ms = [m for m in modes if isinstance(m, str)]
        return tuple(sorted(dict.fromkeys(ms)))
    if isinstance(modes, str):
        return tuple(sorted(dict.fromkeys([modes])))
    return str(modes)


def _md_table(rows: Sequence[Mapping[str, str]], headers: Sequence[str]) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    out = []
    out.append("| " + " | ".join(esc(h) for h in headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(esc(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def _uniq(values: Iterable[object]) -> List[object]:
    uniq: List[object] = []
    for v in values:
        if not any(v == u for u in uniq):
            uniq.append(v)
    return uniq


def _check_equal(label: str, values: Sequence[object]) -> Tuple[bool, str]:
    uniq = _uniq(values)
    ok = len(uniq) == 1 and uniq[0] is not MISSING
    return ok, f"{label}: {'PASS' if ok else 'FAIL'} ({', '.join(_as_str(v) for v in uniq)})"

def _is_missing(v: object) -> bool:
    return v is MISSING or v is None


def _check_optional_equal(label: str, values: Sequence[object]) -> Tuple[bool, str]:
    """Optional fairness check.

    If all values are missing (MISSING/null), skip; otherwise require all present + equal.
    """
    if not any(not _is_missing(v) for v in values):
        return True, f"{label}: SKIP (all missing)"
    uniq = _uniq(values)
    ok = len({v for v in values if not _is_missing(v)}) == 1 and all(not _is_missing(v) for v in values)
    return ok, f"{label}: {'PASS' if ok else 'FAIL'} ({', '.join(_as_str(v) for v in uniq)})"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_summary(eval_root_or_summary: Path) -> Path:
    p = eval_root_or_summary.expanduser()
    if p.is_dir():
        # Standard outputs use summary_test.json.
        cand = p / "summary_test.json"
        if cand.is_file():
            return cand
        # Fallback: any summary_*.json under the directory.
        matches = sorted(p.glob("summary_*.json"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"no summary json found under: {p}")
    if p.is_file():
        return p
    raise FileNotFoundError(f"missing eval_root/summary path: {p}")


@dataclass(frozen=True)
class DetRow:
    run: str
    eval_root: Path
    summary_path: Path
    # protocol/meta
    split: object
    sample_size: object
    seed: object
    modes: object
    frames_json: object
    frames_hash_md5: object
    frames_hash_md5_v2: object
    model_task: object
    keep_camera_poses: object
    keep_main_agent_poses: object
    model_arch: object
    det_head_cfg: object
    det_head_ckpt: object
    det_decode_cfg: object
    eval_script_md5: object
    # metrics
    ap_single: object
    ap_coop: object
    prec_single: object
    prec_coop: object
    rec_single: object
    rec_coop: object
    # artifacts
    bev_png_index: str


def _pick_det_model_key(metrics: object) -> str | None:
    if not isinstance(metrics, Mapping):
        return None
    if "det_model" in metrics:
        return "det_model"
    # fallback: first key (common when renaming models)
    for k in metrics.keys():
        if isinstance(k, str):
            return k
    return None


def load_det_row(eval_root_or_summary: Path) -> DetRow:
    summary_path = _resolve_summary(eval_root_or_summary).resolve()
    eval_root = summary_path.parent
    doc = _read_json(summary_path)
    meta = _get_meta(doc)
    metrics = doc.get("metrics")

    det_key = _pick_det_model_key(metrics)
    per_mode = (metrics or {}).get(det_key) if det_key and isinstance(metrics, Mapping) else {}
    if not isinstance(per_mode, Mapping):
        per_mode = {}

    def get_metric(mode: str, name: str) -> object:
        md = per_mode.get(mode)
        if not isinstance(md, Mapping):
            return MISSING
        return md.get(name, MISSING)

    bev_png_index = ""
    cand = eval_root / "bev_png" / "index.html"
    if cand.is_file():
        bev_png_index = str(cand)

    run = eval_root.name
    return DetRow(
        run=run,
        eval_root=eval_root,
        summary_path=summary_path,
        split=doc.get("split", MISSING),
        sample_size=_meta_get(meta, "sample_size"),
        seed=_meta_get(meta, "seed"),
        modes=_canonical_modes(_meta_get(meta, "modes")),
        frames_json=_meta_get(meta, "frames_json"),
        frames_hash_md5=_meta_get(meta, "frames_hash_md5"),
        frames_hash_md5_v2=_meta_get(meta, "frames_hash_md5_v2"),
        model_task=_meta_get(meta, "model_task"),
        keep_camera_poses=_meta_get(meta, "keep_camera_poses"),
        keep_main_agent_poses=_meta_get(meta, "keep_main_agent_poses"),
        model_arch=_meta_get(meta, "model_arch"),
        det_head_cfg=_meta_get(meta, "det_head_cfg"),
        det_head_ckpt=_meta_get(meta, "det_head_ckpt"),
        det_decode_cfg=_meta_get(meta, "det_decode_cfg"),
        eval_script_md5=_meta_get(meta, "eval_script_md5"),
        ap_single=get_metric("single", "det_ap_iou"),
        ap_coop=get_metric("coop", "det_ap_iou"),
        prec_single=get_metric("single", "det_precision_iou"),
        prec_coop=get_metric("coop", "det_precision_iou"),
        rec_single=get_metric("single", "det_recall_iou"),
        rec_coop=get_metric("coop", "det_recall_iou"),
        bev_png_index=bev_png_index,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_roots", nargs="+", type=Path, required=True, help="Eval dirs (contain summary_test.json) or summary json files")
    ap.add_argument("--out_md", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = [load_det_row(p) for p in args.eval_roots]

    out_md = args.out_md.expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Det Suite Comparison\n")
    lines.append(f"- runs: {len(rows)}")
    lines.append("")

    # Protocol checks (cross-model comparable).
    checks = [
        _check_equal("frames_hash_md5", [r.frames_hash_md5 for r in rows]),
        _check_optional_equal("frames_hash_md5_v2", [r.frames_hash_md5_v2 for r in rows]),
        _check_equal("sample_size", [r.sample_size for r in rows]),
        _check_equal("seed", [r.seed for r in rows]),
        _check_equal("modes", [r.modes for r in rows]),
        _check_equal("model_task", [r.model_task for r in rows]),
        _check_equal("keep_camera_poses", [r.keep_camera_poses for r in rows]),
        _check_equal("keep_main_agent_poses", [r.keep_main_agent_poses for r in rows]),
        _check_equal("det_head_cfg", [r.det_head_cfg for r in rows]),
        _check_equal("det_head_ckpt", [r.det_head_ckpt for r in rows]),
        _check_equal("det_decode_cfg", [r.det_decode_cfg for r in rows]),
        _check_equal("eval_script_md5", [r.eval_script_md5 for r in rows]),
    ]

    lines.append("## Fairness Checks (protocol_fair)\n")
    for _ok, msg in checks:
        lines.append(f"- {msg}")
    lines.append("")

    # Meta table.
    meta_rows: List[Dict[str, str]] = []
    for r in rows:
        meta_rows.append(
            {
                "run": r.run,
                "summary": str(r.summary_path),
                "model_arch": _as_str(r.model_arch),
                "frames_hash_md5": _as_str(r.frames_hash_md5),
                "frames_hash_md5_v2": _as_str(r.frames_hash_md5_v2),
                "seed": _as_str(r.seed),
                "model_task": _as_str(r.model_task),
                "keep_camera_poses": _as_str(r.keep_camera_poses),
                "keep_main_agent_poses": _as_str(r.keep_main_agent_poses),
                "det_head_cfg": _as_str(r.det_head_cfg),
                "det_head_ckpt": _as_str(r.det_head_ckpt),
                "bev_png": f"`{r.bev_png_index}`" if r.bev_png_index else "",
            }
        )
    meta_headers = [
        "run",
        "summary",
        "model_arch",
        "frames_hash_md5",
        "frames_hash_md5_v2",
        "seed",
        "model_task",
        "keep_camera_poses",
        "keep_main_agent_poses",
        "det_head_cfg",
        "det_head_ckpt",
        "bev_png",
    ]
    lines.append("## Runs\n")
    lines.append(_md_table(meta_rows, meta_headers))
    lines.append("")

    # Metrics table.
    def f6(x: object) -> str:
        if x is MISSING:
            return ""
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    metric_rows: List[Dict[str, str]] = []
    for r in rows:
        metric_rows.append(
            {
                "run": r.run,
                "ap_single": f6(r.ap_single),
                "ap_coop": f6(r.ap_coop),
                "prec_single": f6(r.prec_single),
                "prec_coop": f6(r.prec_coop),
                "rec_single": f6(r.rec_single),
                "rec_coop": f6(r.rec_coop),
            }
        )
    metric_headers = ["run", "ap_single", "ap_coop", "prec_single", "prec_coop", "rec_single", "rec_coop"]
    lines.append("## Metrics (from summary_test.json)\n")
    lines.append(_md_table(metric_rows, metric_headers))
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
