#!/usr/bin/env python3
"""Compare two frames contracts and report distinguishability.

This script quantifies how different two contracts are by reporting:
  - overlap (intersection, union, Jaccard, subset/coverage),
  - bucket count/ratio shifts on configurable baseline-distance edges,
  - optional Markdown + JSON outputs for protocol reports.

Default bucket edges follow fine_eval:
  [0, 15, 30, 60, 100, inf]

Self-check
----------
For a pure syntax check:
  python -m py_compile scripts/report_contract_distinguishability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from coop_bucket_presets import (
    FINE_EVAL_BUCKET_EDGES_STR as DEFAULT_BUCKET_EDGES_STR,
    bucket_index as _bucket_index,
    bucket_label as _bucket_label,
    parse_edges as _parse_edges,
)


def _load_contract(path: Path) -> Dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    frames = doc.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"contract missing list field frames: {path}")
    return doc


def _coerce_str_list(x: Any) -> List[str]:
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Best-effort fallback for stringified lists.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        if "," in s:
            return [tok.strip() for tok in s.split(",") if tok.strip()]
        return [s]
    return [str(x)]


def _frame_key(
    fr: Mapping[str, Any],
    *,
    mode: str,
) -> Tuple[str, ...]:
    seq = str(fr.get("sequence"))
    frame = str(fr.get("frame"))
    if mode == "v1":
        return (seq, frame)

    main = str(fr.get("main_agent"))
    coop = tuple(_coerce_str_list(fr.get("coop_agents")))
    if mode == "v2":
        return (seq, frame, main, ",".join(coop))

    ds = str(fr.get("dataset") or "")
    return (ds, seq, frame, main, ",".join(coop))


def _validate_frame_fields(*, fr: Mapping[str, Any], mode: str, index: int) -> None:
    required = ["sequence", "frame"]
    if mode in ("v2", "v3"):
        required.extend(["main_agent", "coop_agents"])
    missing = [k for k in required if fr.get(k, None) is None]
    if missing:
        raise ValueError(
            f"frame[{index}] missing required fields for key_mode={mode}: {missing}"
        )


def _choose_key_mode(*, requested: str, a_frames: Sequence[Mapping[str, Any]], b_frames: Sequence[Mapping[str, Any]]) -> str:
    if requested != "auto":
        return requested
    a_has_dataset = any(bool(str(fr.get("dataset") or "").strip()) for fr in a_frames)
    b_has_dataset = any(bool(str(fr.get("dataset") or "").strip()) for fr in b_frames)
    if a_has_dataset and b_has_dataset:
        return "v3"
    if a_has_dataset != b_has_dataset:
        print("[WARN] dataset field asymmetric; auto fallback to v2")
    return "v2"


def _key_stats(frames: Sequence[Mapping[str, Any]], *, mode: str) -> Dict[str, Any]:
    keys: List[Tuple[str, ...]] = []
    for i, fr in enumerate(frames):
        _validate_frame_fields(fr=fr, mode=mode, index=i)
        keys.append(_frame_key(fr, mode=mode))
    unique = set(keys)
    return {
        "keys": unique,
        "total": int(len(keys)),
        "unique": int(len(unique)),
        "duplicates": int(len(keys) - len(unique)),
    }


def _bucket_stats(frames: Sequence[Mapping[str, Any]], *, edges: Sequence[float]) -> Dict[str, Any]:
    counts = [0 for _ in range(len(edges) - 1)]
    missing_baseline = 0
    invalid_baseline = 0
    for fr in frames:
        x = fr.get("baseline_distance_m", None)
        if x is None:
            missing_baseline += 1
            continue
        try:
            xf = float(x)
        except Exception:
            invalid_baseline += 1
            continue
        bi = _bucket_index(edges, xf)
        if bi is None:
            invalid_baseline += 1
            continue
        counts[int(bi)] += 1

    valid = int(sum(counts))
    ratios = [float(c) / float(valid) if valid > 0 else 0.0 for c in counts]
    return {
        "counts": counts,
        "ratios": ratios,
        "valid_baseline_frames": valid,
        "missing_baseline_frames": int(missing_baseline),
        "invalid_baseline_frames": int(invalid_baseline),
    }


def _safe_div(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _render_markdown(*, report: Mapping[str, Any]) -> str:
    a = report["contract_a"]
    b = report["contract_b"]
    cmp = report["comparison"]
    ov = cmp["overlap"]
    rows = cmp["bucket_rows"]

    lines: List[str] = []
    lines.append("# Contract Distinguishability")
    lines.append("")
    lines.append("## Overlap")
    lines.append("")
    lines.append(f"- key_mode: `{cmp['key_mode']}`")
    lines.append(f"- A: `{a['label']}` (`n={a['num_frames']}`, unique={a['unique_keys']})")
    lines.append(f"- B: `{b['label']}` (`n={b['num_frames']}`, unique={b['unique_keys']})")
    lines.append(f"- intersection: `{ov['intersection']}`")
    lines.append(f"- union: `{ov['union']}`")
    lines.append(f"- Jaccard: `{ov['jaccard']:.6f}`")
    lines.append(f"- coverage A in B: `{ov['coverage_a_in_b']:.6f}`")
    lines.append(f"- coverage B in A: `{ov['coverage_b_in_a']:.6f}`")
    lines.append(f"- A subset of B: `{str(ov['is_a_subset_of_b']).lower()}`")
    lines.append(f"- B subset of A: `{str(ov['is_b_subset_of_a']).lower()}`")
    if str(cmp.get("key_mode")) == "v3":
        if "v2_collision" in a or "v2_collision" in b:
            lines.append(
                f"- v2_collision(ignore dataset): `A={int(a.get('v2_collision') or 0)}` `B={int(b.get('v2_collision') or 0)}`"
            )
    lines.append("")
    lines.append("## Bucket Shift")
    lines.append("")
    edge_tokens = [
        "inf" if float(x) >= 1e8 else str(int(x) if float(x).is_integer() else x)
        for x in cmp["bucket_edges_m"]
    ]
    lines.append(
        f"- edges_m: `{','.join(edge_tokens)}`"
    )
    lines.append(f"- max_abs_ratio_shift_pp: `{cmp['max_abs_ratio_shift_pp']:.3f}`")
    lines.append("")
    lines.append("| bucket | count_A | ratio_A | count_B | ratio_B | delta_count(B-A) | delta_ratio_pp(B-A) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            "| {bucket} | {count_a} | {ratio_a:.4f} | {count_b} | {ratio_b:.4f} | {delta_count_b_minus_a:+d} | {delta_ratio_pp_b_minus_a:+.3f} |".format(
                **r
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract_a", type=Path, required=True, help="Path to contract A JSON.")
    ap.add_argument("--contract_b", type=Path, required=True, help="Path to contract B JSON.")
    ap.add_argument("--label_a", type=str, default="", help="Optional display label for contract A.")
    ap.add_argument("--label_b", type=str, default="", help="Optional display label for contract B.")
    ap.add_argument(
        "--key_mode",
        type=str,
        default="auto",
        choices=("auto", "v1", "v2", "v3"),
        help=(
            "Overlap key mode. "
            "v1: sequence/frame; "
            "v2: sequence/frame/main_agent/coop_agents; "
            "v3: dataset + v2; "
            "auto: v3 when dataset is present, else v2."
        ),
    )
    ap.add_argument(
        "--bucket_edges",
        type=str,
        default=DEFAULT_BUCKET_EDGES_STR,
        help=(
            "Comma-separated baseline-distance bucket edges in meters. "
            f"Default: {DEFAULT_BUCKET_EDGES_STR}."
        ),
    )
    ap.add_argument("--out_json", type=Path, default=None, help="Optional JSON report output path.")
    ap.add_argument("--out_md", type=Path, default=None, help="Optional Markdown summary output path.")
    ap.add_argument(
        "--require_disjoint",
        action="store_true",
        help="Gate check: require intersection == 0 (exit!=0 if violated).",
    )
    ap.add_argument(
        "--max_jaccard",
        type=float,
        default=None,
        help="Gate check: require Jaccard <= this threshold (exit!=0 if violated).",
    )
    ap.add_argument(
        "--max_bucket_shift_pp",
        type=float,
        default=None,
        help="Gate check: require max_abs_ratio_shift_pp <= this threshold (exit!=0 if violated).",
    )
    args = ap.parse_args()

    a_path = args.contract_a.expanduser().resolve()
    b_path = args.contract_b.expanduser().resolve()
    if not a_path.is_file():
        raise FileNotFoundError(f"contract A not found: {a_path}")
    if not b_path.is_file():
        raise FileNotFoundError(f"contract B not found: {b_path}")

    edges = _parse_edges(str(args.bucket_edges))
    labels = [_bucket_label(edges, i) for i in range(len(edges) - 1)]

    doc_a = _load_contract(a_path)
    doc_b = _load_contract(b_path)
    frames_a = [fr for fr in doc_a.get("frames", []) if isinstance(fr, Mapping)]
    frames_b = [fr for fr in doc_b.get("frames", []) if isinstance(fr, Mapping)]

    mode = _choose_key_mode(requested=str(args.key_mode), a_frames=frames_a, b_frames=frames_b)
    ka = _key_stats(frames_a, mode=mode)
    kb = _key_stats(frames_b, mode=mode)
    v2_collision_a = None
    v2_collision_b = None
    unique_v2_a = None
    unique_v2_b = None
    if mode == "v3":
        # "v2-collision": additional key collisions introduced when dropping dataset from v3 -> v2.
        ka_v2 = _key_stats(frames_a, mode="v2")
        kb_v2 = _key_stats(frames_b, mode="v2")
        unique_v2_a = int(ka_v2["unique"])
        unique_v2_b = int(kb_v2["unique"])
        v2_collision_a = int(int(ka["unique"]) - unique_v2_a)
        v2_collision_b = int(int(kb["unique"]) - unique_v2_b)
    sa = ka["keys"]
    sb = kb["keys"]

    inter = int(len(sa & sb))
    uni = int(len(sa | sb))
    overlap = {
        "intersection": inter,
        "union": uni,
        "a_only": int(len(sa - sb)),
        "b_only": int(len(sb - sa)),
        "jaccard": _safe_div(inter, uni),
        "coverage_a_in_b": _safe_div(inter, int(ka["unique"])),
        "coverage_b_in_a": _safe_div(inter, int(kb["unique"])),
        "is_a_subset_of_b": bool(sa <= sb),
        "is_b_subset_of_a": bool(sb <= sa),
    }

    ba = _bucket_stats(frames_a, edges=edges)
    bb = _bucket_stats(frames_b, edges=edges)

    rows: List[Dict[str, Any]] = []
    max_abs_ratio_shift_pp = 0.0
    for i, label in enumerate(labels):
        ratio_a = float(ba["ratios"][i])
        ratio_b = float(bb["ratios"][i])
        delta_pp = (ratio_b - ratio_a) * 100.0
        max_abs_ratio_shift_pp = max(max_abs_ratio_shift_pp, abs(delta_pp))
        rows.append(
            {
                "bucket": label,
                "count_a": int(ba["counts"][i]),
                "ratio_a": ratio_a,
                "count_b": int(bb["counts"][i]),
                "ratio_b": ratio_b,
                "delta_count_b_minus_a": int(bb["counts"][i] - ba["counts"][i]),
                "delta_ratio_pp_b_minus_a": delta_pp,
            }
        )

    label_a = str(args.label_a).strip() or a_path.stem
    label_b = str(args.label_b).strip() or b_path.stem

    report: Dict[str, Any] = {
        "contract_a": {
            "label": label_a,
            "path": str(a_path),
            "split": doc_a.get("split"),
            "sample_size": doc_a.get("sample_size"),
            "num_frames": int(len(frames_a)),
            "unique_keys": int(ka["unique"]),
            "duplicate_keys": int(ka["duplicates"]),
            **(
                {
                    "unique_keys_v2": int(unique_v2_a),
                    "v2_collision": int(v2_collision_a),
                }
                if mode == "v3"
                else {}
            ),
            "frames_hash_md5_v2": doc_a.get("frames_hash_md5_v2"),
            "frames_hash_md5_v3": doc_a.get("frames_hash_md5_v3"),
            "bucket_valid_frames": int(ba["valid_baseline_frames"]),
            "bucket_missing_baseline_frames": int(ba["missing_baseline_frames"]),
            "bucket_invalid_baseline_frames": int(ba["invalid_baseline_frames"]),
        },
        "contract_b": {
            "label": label_b,
            "path": str(b_path),
            "split": doc_b.get("split"),
            "sample_size": doc_b.get("sample_size"),
            "num_frames": int(len(frames_b)),
            "unique_keys": int(kb["unique"]),
            "duplicate_keys": int(kb["duplicates"]),
            **(
                {
                    "unique_keys_v2": int(unique_v2_b),
                    "v2_collision": int(v2_collision_b),
                }
                if mode == "v3"
                else {}
            ),
            "frames_hash_md5_v2": doc_b.get("frames_hash_md5_v2"),
            "frames_hash_md5_v3": doc_b.get("frames_hash_md5_v3"),
            "bucket_valid_frames": int(bb["valid_baseline_frames"]),
            "bucket_missing_baseline_frames": int(bb["missing_baseline_frames"]),
            "bucket_invalid_baseline_frames": int(bb["invalid_baseline_frames"]),
        },
        "comparison": {
            "key_mode": mode,
            "bucket_edges_m": [float(x) for x in edges],
            "overlap": overlap,
            "bucket_rows": rows,
            "max_abs_ratio_shift_pp": float(max_abs_ratio_shift_pp),
        },
    }

    print("[COMPARE]", label_a, "vs", label_b)
    print("  key_mode:", mode)
    print("  n_a:", len(frames_a), "n_b:", len(frames_b))
    print(
        "  overlap:",
        f"inter={overlap['intersection']}",
        f"union={overlap['union']}",
        f"jaccard={overlap['jaccard']:.6f}",
        f"coverage_a_in_b={overlap['coverage_a_in_b']:.6f}",
        f"coverage_b_in_a={overlap['coverage_b_in_a']:.6f}",
    )
    print(
        "  subset:",
        f"a_in_b={str(overlap['is_a_subset_of_b']).lower()}",
        f"b_in_a={str(overlap['is_b_subset_of_a']).lower()}",
    )
    if mode == "v3":
        print(
            "  v2_collision(ignore dataset):",
            f"A={int(v2_collision_a)} (unique_v2={int(unique_v2_a)})",
            f"B={int(v2_collision_b)} (unique_v2={int(unique_v2_b)})",
        )
    print("  bucket max_abs_ratio_shift_pp:", f"{max_abs_ratio_shift_pp:.3f}")

    for row in rows:
        print(
            "   -",
            row["bucket"],
            f"A={row['count_a']} ({row['ratio_a']:.3%})",
            f"B={row['count_b']} ({row['ratio_b']:.3%})",
            f"delta_count={row['delta_count_b_minus_a']:+d}",
            f"delta_pp={row['delta_ratio_pp_b_minus_a']:+.3f}",
        )

    if args.out_json is not None:
        out_json = args.out_json.expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print("[OK] wrote:", out_json)

    if args.out_md is not None:
        out_md = args.out_md.expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(report=report), encoding="utf-8")
        print("[OK] wrote:", out_md)

    failures: List[str] = []
    if bool(args.require_disjoint) and int(overlap["intersection"]) != 0:
        failures.append(f"--require_disjoint failed: intersection={int(overlap['intersection'])} != 0")
    if args.max_jaccard is not None and float(overlap["jaccard"]) > float(args.max_jaccard):
        failures.append(f"--max_jaccard failed: jaccard={float(overlap['jaccard']):.6f} > {float(args.max_jaccard):.6f}")
    if args.max_bucket_shift_pp is not None and float(max_abs_ratio_shift_pp) > float(args.max_bucket_shift_pp):
        failures.append(
            f"--max_bucket_shift_pp failed: max_abs_ratio_shift_pp={float(max_abs_ratio_shift_pp):.3f} > {float(args.max_bucket_shift_pp):.3f}"
        )
    if failures:
        print("[GATE FAIL]")
        for msg in failures:
            print("  -", msg)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
