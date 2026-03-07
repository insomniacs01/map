#!/usr/bin/env python3
"""Export OPV2V parquet index to the generic coop-index schema.

Why
---
This repo supports a dataset-agnostic contract workflow:
  index parquet (pairing + baseline_distance precomputed)
    -> frames contract JSON (bucketed sampling)

OPV2V already has a canonical index:
  data/opv2v/opv2v_index/index_{train,validate,test}.parquet

but its schema is richer/different. This script converts it into a small
generic index parquet with the minimum required columns for:
  - scripts/report_index_bucket_stats.py
  - scripts/make_frames_contract_from_index.py

Output schema (generic)
-----------------------
- split: str
- dataset: str (optional; if `--include_dataset` is set)
- sequence: str
- frame: str
- main_agent: str
- coop_agents: list[str]  (ordered; [main, pair] for OPV2V 2-agent pairing)
- baseline_distance_m: float
- has_assets: bool
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _baseline_distance_xy_m(agent_xy: Dict[str, Any], a: str, b: str) -> float | None:
    try:
        pa = agent_xy.get(str(a))
        pb = agent_xy.get(str(b))
        if pa is None or pb is None:
            return None
        pa2 = np.asarray(pa, dtype=np.float64).reshape(-1)
        pb2 = np.asarray(pb, dtype=np.float64).reshape(-1)
        if pa2.size < 2 or pb2.size < 2:
            return None
        d = float(np.linalg.norm(pa2[:2] - pb2[:2]))
        if not math.isfinite(d) or d < 0.0:
            return None
        return d
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--opv2v_index_dir",
        type=Path,
        default=Path("data/opv2v/opv2v_index"),
        help="Directory containing index_{train,validate,test}.parquet.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,validate,test",
        help="Comma-separated splits to include (default: train,validate,test).",
    )
    ap.add_argument(
        "--include_dataset",
        action="store_true",
        help="Include a `dataset` column in the exported generic index (useful for multi-dataset merges).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="opv2v",
        help="Dataset identifier used when --include_dataset is set (default: opv2v).",
    )
    ap.add_argument("--out_parquet", type=Path, required=True)
    args = ap.parse_args()

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required to read/write parquet") from exc

    index_dir = args.opv2v_index_dir.expanduser().resolve()
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits provided.")

    rows: List[dict] = []
    for split in splits:
        p = index_dir / f"index_{split}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"Missing OPV2V index parquet: {p}")
        wanted = ["split", "sequence", "frame", "main_agent", "pair_agent", "agent_xy", "agent_has_assets"]
        # Best-effort column projection (works with pyarrow backend). Fall back to reading all columns.
        try:
            import pyarrow.parquet as pq  # type: ignore

            schema = pq.read_schema(p)
            available = set(schema.names)
            cols = [c for c in wanted if c in available]
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)

        # Ensure missing columns are filled for backward compatibility with older indices.
        for c in wanted:
            if c not in df.columns:
                df[c] = None
        for r in df.itertuples(index=False):
            seq = getattr(r, "sequence", None)
            frame = getattr(r, "frame", None)
            main = getattr(r, "main_agent", None)
            pair = getattr(r, "pair_agent", None)
            agent_xy = getattr(r, "agent_xy", None) or {}
            has_assets_map = getattr(r, "agent_has_assets", None) or {}
            if seq is None or frame is None or main is None or pair is None:
                continue
            if not isinstance(agent_xy, dict):
                continue
            d = _baseline_distance_xy_m(agent_xy, str(main), str(pair))
            if d is None:
                continue
            has_assets = bool(has_assets_map.get(str(main), False) and has_assets_map.get(str(pair), False))
            split_raw = getattr(r, "split", None)
            split_val = str(split)
            if split_raw is not None:
                try:
                    # Avoid stringifying NaN to "nan".
                    if isinstance(split_raw, float) and math.isnan(float(split_raw)):
                        pass
                    else:
                        s = str(split_raw).strip()
                        if s and s.lower() != "nan":
                            split_val = s
                except Exception:
                    split_val = str(split)
            row = dict(
                split=split_val,
                sequence=str(seq),
                frame=str(frame),
                main_agent=str(main),
                coop_agents=[str(main), str(pair)],
                baseline_distance_m=float(d),
                has_assets=bool(has_assets),
            )
            if args.include_dataset:
                row["dataset"] = str(args.dataset)
            rows.append(row)

    if not rows:
        raise SystemExit("No rows exported (unexpected).")

    out_df = pd.DataFrame(rows)
    out_path = args.out_parquet.expanduser()
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parents[1] / out_path).resolve()
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[OK] wrote generic index parquet: {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
