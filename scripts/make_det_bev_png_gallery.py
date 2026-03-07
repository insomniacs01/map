#!/usr/bin/env python3
"""
Generate 2D BEV PNG overlays (+ offline index.html) for an existing det/e2e eval run.

This script is designed to be re-runnable after:
- changing BEV visualization code (`scripts/viz_batch_eval_bev_png.py`)
- fixing visualization thresholds / axis limits
- older eval runs that missed PNG generation due to scripting bugs

It scans `<eval_root>/**/_representatives/*.pcd`, finds the corresponding OPV2V YAML via the
frame contract, and renders BEV PNGs into `<eval_root>/bev_png/<model>/<mode>/...`.

Example:
  cd map-anything
  python scripts/make_det_bev_png_gallery.py \\
    --eval_root eval_runs/det_compose_posed_test50_20260226b_sfix_direct \\
    --frames_json eval_runs/frames_test50_det_e2e_v5.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_frames_lookup(frames_json: Path) -> tuple[str, Dict[tuple[str, str], str]]:
    doc = json.loads(frames_json.read_text(encoding="utf-8"))
    split = str(doc.get("split") or "test")
    frames = doc.get("frames") or []
    lookup: Dict[tuple[str, str], str] = {}
    if isinstance(frames, list):
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            seq = fr.get("sequence")
            frame = fr.get("frame")
            main = fr.get("main_agent")
            if isinstance(seq, str) and isinstance(frame, str) and main is not None:
                lookup[(seq, frame)] = str(main)
    return split, lookup


def _read_summary_meta(eval_root: Path) -> tuple[dict, dict]:
    summary_path = eval_root / "summary_test.json"
    if not summary_path.is_file():
        return {}, {}
    try:
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    meta = summ.get("meta") if isinstance(summ, dict) else {}
    metrics = summ.get("metrics") if isinstance(summ, dict) else {}
    return (meta if isinstance(meta, dict) else {}), (metrics if isinstance(metrics, dict) else {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument(
        "--frames_json",
        type=Path,
        default=None,
        help="Frame contract used for the eval. If omitted, try summary_test.json meta.frames_json, else default to canonical Test50.",
    )
    parser.add_argument("--pred_min_score", type=float, default=0.05)
    parser.add_argument("--det_x_min", type=float, default=-50.0)
    parser.add_argument("--det_x_max", type=float, default=120.0)
    parser.add_argument("--det_y_min", type=float, default=-50.0)
    parser.add_argument("--det_y_max", type=float, default=50.0)
    parser.add_argument(
        "--no_color_pred_by_match",
        action="store_true",
        help="Disable TP/FP/FN coloring (default: enabled if *_pred_boxes.json exists).",
    )
    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(eval_root)

    meta, metrics = _read_summary_meta(eval_root)
    frames_json = args.frames_json
    if frames_json is None:
        meta_frames_json = meta.get("frames_json")
        if isinstance(meta_frames_json, str) and meta_frames_json:
            frames_json = Path(meta_frames_json)
        else:
            frames_json = REPO_ROOT / "eval_runs" / "frames_test50_det_e2e_v5.json"
    frames_json = frames_json.expanduser().resolve()
    if not frames_json.is_file():
        raise FileNotFoundError(frames_json)

    split, lookup = _load_frames_lookup(frames_json)

    rep_dirs = [p for p in eval_root.rglob("*_representatives") if p.is_dir()]
    png_dir = eval_root / "bev_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    # Frame keys are saved as: <sequence>_<frame6>_<tag>.pcd
    pat = re.compile(r"^(?P<seq>.+)_(?P<frame>\d{6})_(?P<tag>best|median|worst)\.pcd$")

    items: List[Tuple[str, str, str]] = []
    for rep_dir in sorted(rep_dirs):
        mode = rep_dir.name.replace("_representatives", "")
        model_key = rep_dir.parent.name
        for pcd in sorted(rep_dir.glob("*.pcd")):
            m = pat.match(pcd.name)
            if not m:
                continue
            seq = m.group("seq")
            frame = m.group("frame")
            main = lookup.get((seq, frame))
            if main is None:
                continue
            yaml_path = REPO_ROOT / "data" / "opv2v" / split / seq / str(main) / f"{frame}.yaml"
            if not yaml_path.is_file():
                continue
            out_png = png_dir / model_key / mode / (pcd.stem + "_bev.png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "viz_batch_eval_bev_png.py"),
                "--pred_pcd",
                str(pcd),
                "--yaml",
                str(yaml_path),
                "--out_png",
                str(out_png),
                "--pred_min_score",
                str(float(args.pred_min_score)),
                "--det_x_min",
                str(float(args.det_x_min)),
                "--det_x_max",
                str(float(args.det_x_max)),
                "--det_y_min",
                str(float(args.det_y_min)),
                "--det_y_max",
                str(float(args.det_y_max)),
            ]
            if not args.no_color_pred_by_match:
                cmd.append("--color_pred_by_match")
            subprocess.run(cmd, check=True)
            items.append((model_key, mode, out_png.relative_to(png_dir).as_posix()))

    # Build a simple offline index.html.
    metric_lines: List[str] = []
    if isinstance(metrics, dict):
        for model_key, per_mode in metrics.items():
            if not isinstance(per_mode, dict):
                continue
            for mode_key, md in per_mode.items():
                if not isinstance(md, dict):
                    continue
                ap = md.get("det_ap_iou")
                prec = md.get("det_precision_iou")
                rec = md.get("det_recall_iou")
                metric_lines.append(f"{model_key}/{mode_key}: ap={ap} prec={prec} rec={rec}")

    meta_payload = {
        "frames_hash_md5": meta.get("frames_hash_md5"),
        "frames_json": str(frames_json),
        "det_head_cfg": meta.get("det_head_cfg"),
        "det_decode_cfg": meta.get("det_decode_cfg"),
        "model_task": meta.get("model_task"),
        "keep_camera_poses": meta.get("keep_camera_poses"),
        "keep_main_agent_poses": meta.get("keep_main_agent_poses"),
        "eval_script_md5": meta.get("eval_script_md5"),
    }

    # Group images by (model, mode) for a compact layout.
    by_group: Dict[tuple[str, str], List[str]] = {}
    for model_key, mode, rel in items:
        by_group.setdefault((model_key, mode), []).append(rel)
    for k in by_group:
        by_group[k].sort()

    index = png_dir / "index.html"
    parts: List[str] = []
    parts.append('<html><head><meta charset="utf-8"/>')
    parts.append(
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif}"
        " img{max-width:520px;height:auto;border:1px solid #ddd}"
        " .grid{display:flex;flex-wrap:wrap;gap:12px}"
        " .tile{width:540px}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append("<h1>Det BEV PNG Gallery</h1>")
    parts.append(f"<p>eval_root: <code>{eval_root}</code></p>")
    parts.append("<h2>Det Summary</h2>")
    parts.append("<pre>" + "\n".join(metric_lines) + "</pre>" if metric_lines else "<pre>(no det metrics)</pre>")
    parts.append("<h2>Protocol Meta</h2>")
    parts.append("<pre>" + json.dumps(meta_payload, indent=2) + "</pre>")

    if by_group:
        parts.append("<h2>Images</h2>")
        for (model_key, mode) in sorted(by_group.keys()):
            parts.append(f"<h3>{model_key} / {mode}</h3>")
            parts.append('<div class="grid">')
            for rel in by_group[(model_key, mode)]:
                parts.append('<div class="tile">')
                parts.append(f'<img src="{rel}"/>')
                parts.append(f"<div><code>{rel}</code></div>")
                parts.append("</div>")
            parts.append("</div>")
    else:
        parts.append("<h2>Images</h2><p>(no representative PCDs found or no YAMLs matched)</p>")

    parts.append("</body></html>")
    index.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[OK] wrote {index}")


if __name__ == "__main__":
    main()

