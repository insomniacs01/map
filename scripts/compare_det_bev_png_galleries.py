#!/usr/bin/env python3
"""
Side-by-side HTML comparison for detection BEV PNG galleries.

Given multiple det/e2e eval runs (each optionally already containing `bev_png/`),
this script:
  1) Reuses existing `bev_png/**/*_bev.png` tiles if present
  2) Otherwise (or when forced) regenerates them via `scripts/make_det_bev_png_gallery.py`
  3) Writes an offline HTML with rows=frame and cols=run, grouped by (model, mode)

Output:
  <repo>/eval_runs/<tag>/compare_bev/index.html

Example:
  cd map-anything
  python scripts/compare_det_bev_png_galleries.py \\
    --tag compare_posed_vs_predpose_20260227 \\
    --frames_json eval_runs/frames_test50_det_e2e_v5.json \\
    --eval_roots \\
      posed=eval_runs/det_compose_posed_test50_20260226b_sfix_direct \\
      predpose=eval_runs/det_compose_predpose_aligngt0_test50_20260226b_sfix_direct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

FrameKey = Tuple[str, str, str]  # (sequence, frame6, tag)
GroupKey = Tuple[str, str]  # (model_key, mode)


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _esc_attr(s: Any) -> str:
    # HTML attribute escape + keep newlines readable in tooltips.
    return _esc(s).replace("\n", "&#10;")


def _relpath_posix(path: Path, start: Path) -> str:
    return Path(os.path.relpath(str(path), str(start))).as_posix()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_det_metrics_lines(summary_test_json: Path) -> List[str]:
    doc = _read_json(summary_test_json)
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict):
        return []
    lines: List[str] = []
    for model_key, per_mode in metrics.items():
        if not isinstance(per_mode, dict):
            continue
        for mode_key, md in per_mode.items():
            if not isinstance(md, dict):
                continue
            ap = md.get("det_ap_iou")
            prec = md.get("det_precision_iou")
            rec = md.get("det_recall_iou")
            lines.append(f"{model_key}/{mode_key}: ap={ap} prec={prec} rec={rec}")
    return lines


@dataclass(frozen=True)
class RunSpec:
    name: str
    eval_root: Path


def _parse_eval_root_spec(spec: str) -> RunSpec:
    # Allow "name=path" or just "path".
    if "=" in spec:
        name, path_s = spec.split("=", 1)
        name = name.strip()
        path_s = path_s.strip()
    else:
        path_s = spec.strip()
        name = Path(path_s).name
    if not name:
        name = Path(path_s).name
    p = Path(path_s)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.expanduser().resolve()
    return RunSpec(name=name, eval_root=p)


def _ensure_unique_names(runs: Sequence[RunSpec]) -> List[RunSpec]:
    out: List[RunSpec] = []
    counts: Dict[str, int] = {}
    for r in runs:
        base = r.name
        n = counts.get(base, 0)
        counts[base] = n + 1
        name = base if n == 0 else f"{base}_{n+1}"
        out.append(RunSpec(name=name, eval_root=r.eval_root))
    return out


def _maybe_regen_bev_png(*, eval_root: Path, frames_json: Optional[Path], force: bool) -> None:
    bev_png_dir = eval_root / "bev_png"

    def is_complete() -> tuple[bool, int, int]:
        # Expected tiles are derived from the representative PCD contract:
        #   <eval_root>/**/<mode>_representatives/*.pcd  =>
        #   <eval_root>/bev_png/<model_key>/<mode>/<pcd_stem>_bev.png
        rep_dirs = [p for p in eval_root.rglob("*_representatives") if p.is_dir()]
        expected = 0
        missing = 0
        for rep_dir in rep_dirs:
            mode = rep_dir.name.replace("_representatives", "")
            model_key = rep_dir.parent.name
            for pcd in rep_dir.glob("*.pcd"):
                expected += 1
                out_png = bev_png_dir / model_key / mode / (pcd.stem + "_bev.png")
                if not out_png.is_file():
                    missing += 1
        if expected == 0:
            # Nothing to (re)generate.
            return True, 0, 0
        return missing == 0, missing, expected

    if not force:
        ok, missing, expected = is_complete()
        if ok:
            return
        print(f"[INFO] bev_png incomplete for {eval_root} ({missing}/{expected} missing); regenerating...")
    else:
        print(f"[INFO] force_regen set; regenerating bev_png for {eval_root}...")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "make_det_bev_png_gallery.py"),
        "--eval_root",
        str(eval_root),
    ]
    if frames_json is not None:
        cmd.extend(["--frames_json", str(frames_json)])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _scan_bev_png(eval_root: Path) -> Dict[GroupKey, Dict[FrameKey, Path]]:
    bev_png_dir = eval_root / "bev_png"
    if not bev_png_dir.is_dir():
        return {}

    pat = re.compile(r"^(?P<seq>.+)_(?P<frame>\d{6})_(?P<tag>best|median|worst)_bev\.png$")

    out: Dict[GroupKey, Dict[FrameKey, Path]] = {}
    for p in sorted(bev_png_dir.rglob("*_bev.png")):
        try:
            rel = p.relative_to(bev_png_dir)
        except Exception:
            continue
        if len(rel.parts) < 3:
            # Expected: <model>/<mode>/<file>.png (but keep it conservative).
            continue
        model_key = rel.parts[0]
        mode = rel.parts[1]
        m = pat.match(p.name)
        if not m:
            continue
        seq = m.group("seq")
        frame = m.group("frame")
        tag = m.group("tag")
        out.setdefault((model_key, mode), {})[(seq, frame, tag)] = p
    return out


def _sorted_frame_keys(keys: Sequence[FrameKey]) -> List[FrameKey]:
    tag_order = {"best": 0, "median": 1, "worst": 2}
    return sorted(keys, key=lambda k: (k[0], k[1], tag_order.get(k[2], 9), k[2]))


def _write_compare_html(
    out_dir: Path,
    *,
    tag: str,
    frames_json: Optional[Path],
    runs: Sequence[RunSpec],
    per_run_images: Sequence[Dict[GroupKey, Dict[FrameKey, Path]]],
    group_set: str,
    frame_set: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"

    # Determine (model, mode) groups to render.
    group_keys_all: List[set[GroupKey]] = [set(d.keys()) for d in per_run_images]
    if not group_keys_all:
        selected_groups: List[GroupKey] = []
    elif group_set == "union":
        selected_groups = sorted(set().union(*group_keys_all))
    else:
        selected_groups = sorted(set.intersection(*group_keys_all))

    css = """
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
    code { background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 4px; }
    table { border-collapse: collapse; margin-bottom: 26px; }
    th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
    th { position: sticky; top: 0; background: #fafafa; z-index: 2; }
    .framecell { min-width: 320px; }
    .tile img { width: 420px; max-width: 420px; border: 1px solid #ccc; }
    .tilecap { margin-top: 4px; font-size: 12px; line-height: 1.15; white-space: pre-line; }
    .badge { display: inline-block; padding: 0.12rem 0.45rem; border-radius: 999px; font-size: 11px; font-weight: 700; border: 1px solid transparent; margin-left: 0.35rem; }
    .badge.best { background: #dcfce7; color: #166534; border-color: #86efac; }
    .badge.median { background: #e5e7eb; color: #374151; border-color: #d1d5db; }
    .badge.worst { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
    .muted { color: #555; font-size: 12px; }
    .runmeta { margin-bottom: 18px; }
    """

    lines: List[str] = []
    lines.extend(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'/>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
            f"<title>Det BEV Compare: {_esc(tag)}</title>",
            f"<style>{css}</style>",
            "</head><body>",
            f"<h1>Det BEV Compare: {_esc(tag)}</h1>",
            "<p class='muted'>Rows are shared representative frames; columns are eval runs.</p>",
        ]
    )

    if frames_json is not None:
        try:
            frames_rel = frames_json.resolve().relative_to(REPO_ROOT)
            frames_disp = str(frames_rel)
        except Exception:
            frames_disp = str(frames_json)
        lines.append(f"<p class='muted'>frames_json: <code>{_esc(frames_disp)}</code></p>")

    # Per-run meta / quick metrics.
    lines.append("<h2>Runs</h2>")
    lines.append("<div class='runmeta'>")
    lines.append("<table>")
    lines.append("<tr><th>Run</th><th>eval_root</th><th>det metrics (summary_test.json)</th></tr>")
    for r in runs:
        summ = r.eval_root / "summary_test.json"
        metric_lines = _read_det_metrics_lines(summ) if summ.is_file() else []
        metric_text = "\n".join(metric_lines) if metric_lines else "(missing summary_test.json or det metrics)"
        lines.append("<tr>")
        lines.append(f"<td><code>{_esc(r.name)}</code></td>")
        lines.append(f"<td><code>{_esc(_relpath_posix(r.eval_root, out_dir))}</code></td>")
        lines.append(f"<td><pre style='margin:0'>{_esc(metric_text)}</pre></td>")
        lines.append("</tr>")
    lines.append("</table>")
    lines.append("</div>")

    if not selected_groups:
        lines.append("<h2>BEV Tiles</h2>")
        lines.append("<p>(no shared bev_png groups found across runs)</p>")
        lines.append("</body></html>")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    # Grouped tables.
    lines.append("<h2>BEV Tiles</h2>")
    lines.append(
        "<p class='muted'>"
        + _esc(f"group_set={group_set}, frame_set={frame_set}")
        + "</p>"
    )

    for group in selected_groups:
        model_key, mode = group
        lines.append(f"<h3>{_esc(model_key)} / {_esc(mode)}</h3>")

        per_run_group_maps: List[Dict[FrameKey, Path]] = []
        for run_map in per_run_images:
            per_run_group_maps.append(run_map.get(group, {}))

        frame_sets: List[set[FrameKey]] = [set(m.keys()) for m in per_run_group_maps]
        if not frame_sets:
            keys: List[FrameKey] = []
        elif frame_set == "union":
            keys = _sorted_frame_keys(list(set().union(*frame_sets)))
        else:
            keys = _sorted_frame_keys(list(set.intersection(*frame_sets)))

        if not keys:
            lines.append("<p class='muted'>(no shared frames across runs for this group)</p>")
            continue

        lines.append("<table>")
        lines.append("<tr><th>Frame</th>" + "".join([f"<th>{_esc(r.name)}</th>" for r in runs]) + "</tr>")

        for (seq, frame, tag) in keys:
            badge = f"<span class='badge {tag}'>{_esc(tag)}</span>"
            lines.append("<tr>")
            lines.append("<td class='framecell'>")
            lines.append(f"<div><code>{_esc(seq)}/{_esc(frame)}</code>{badge}</div>")
            lines.append("</td>")

            for run_idx, run in enumerate(runs):
                p = per_run_group_maps[run_idx].get((seq, frame, tag))
                if p is None or not p.is_file():
                    lines.append("<td class='tile'><div class='muted'>missing</div></td>")
                    continue
                rel = _relpath_posix(p, out_dir)
                title = f"{run.name}: {model_key}/{mode} {seq}/{frame} {tag}"
                lines.append("<td class='tile'>")
                lines.append(
                    f"<a href='{_esc(rel)}' target='_blank' rel='noopener'>"
                    f"<img src='{_esc(rel)}' loading='lazy' title='{_esc_attr(title)}'/>"
                    "</a>"
                )
                lines.append(f"<div class='tilecap muted'><code>{_esc(Path(rel).as_posix())}</code></div>")
                lines.append("</td>")

            lines.append("</tr>")

        lines.append("</table>")

    lines.append("</body></html>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, required=True, help="Output tag under eval_runs/<tag>/compare_bev/")
    parser.add_argument(
        "--eval_roots",
        type=str,
        nargs="+",
        required=True,
        help="Eval run dirs to compare. Format: PATH or NAME=PATH (PATH relative to repo root is ok).",
    )
    parser.add_argument(
        "--frames_json",
        type=Path,
        default=None,
        help="Frame contract used for the eval (passed through for PNG regeneration).",
    )
    parser.add_argument("--force_regen", action="store_true", help="Always regenerate bev_png tiles for each run.")
    parser.add_argument(
        "--group_set",
        choices=("intersection", "union"),
        default="intersection",
        help="Which (model,mode) groups to render across runs.",
    )
    parser.add_argument(
        "--frame_set",
        choices=("intersection", "union"),
        default="intersection",
        help="Which frames to render within each (model,mode) group.",
    )
    args = parser.parse_args()

    runs = _ensure_unique_names([_parse_eval_root_spec(s) for s in args.eval_roots])

    frames_json = args.frames_json
    if frames_json is None:
        # Keep in sync with `make_det_bev_png_gallery.py` default.
        frames_json = REPO_ROOT / "eval_runs" / "frames_test50_det_e2e_v5.json"
    frames_json = frames_json.expanduser().resolve()
    if not frames_json.is_file():
        raise FileNotFoundError(frames_json)

    for r in runs:
        if not r.eval_root.is_dir():
            raise FileNotFoundError(r.eval_root)
        _maybe_regen_bev_png(eval_root=r.eval_root, frames_json=frames_json, force=bool(args.force_regen))

    per_run_images = [_scan_bev_png(r.eval_root) for r in runs]

    out_dir = REPO_ROOT / "eval_runs" / str(args.tag) / "compare_bev"
    out_path = _write_compare_html(
        out_dir,
        tag=str(args.tag),
        frames_json=frames_json,
        runs=runs,
        per_run_images=per_run_images,
        group_set=str(args.group_set),
        frame_set=str(args.frame_set),
    )
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
