#!/usr/bin/env python3
"""Merge multiple generic coop-index parquets into one (optionally multi-dataset).

Why
---
`make_frames_contract_from_index.py` consumes a *single* generic index parquet.
When you want to build a contract over multiple datasets (or over sharded index
files), merge them first.

For multi-dataset merges, make sure each source has a stable `dataset` id so
downstream contracts can compute `frames_hash_md5_v3` (dataset-aware).

Schema expectation (per input parquet)
-------------------------------------
Required columns:
- split
- sequence
- frame
- main_agent
- coop_agents
- baseline_distance_m

Optional columns:
- dataset
- has_assets

Example
-------
PYTHONPATH=$(pwd) python scripts/merge_coop_index_parquets.py \\
  --index_parquets data/opv2v/opv2v_index/opv2v_index_generic.parquet data/other/other_index_generic.parquet \\
  --datasets opv2v,other \\
  --out_parquet data/merged/coop_index_merged.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _parse_datasets_arg(datasets: str, n: int) -> list[str | None]:
    ds = [s.strip() for s in str(datasets).split(",") if s.strip()]
    if not ds:
        return [None for _ in range(n)]
    if len(ds) == 1:
        return [str(ds[0]) for _ in range(n)]
    if len(ds) != n:
        raise ValueError(f"--datasets expects 1 or {n} values; got {len(ds)}: {ds}")
    return [str(x) for x in ds]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--index_parquets",
        type=Path,
        nargs="+",
        required=True,
        help="Input generic index parquet paths to merge.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Optional comma-separated dataset ids. If 1 value is given, it is applied to all inputs. "
            "If N values are given, must match --index_parquets order."
        ),
    )
    ap.add_argument(
        "--require_dataset",
        action="store_true",
        help="Fail if any row has missing/empty dataset after applying --datasets overrides.",
    )
    ap.add_argument("--out_parquet", type=Path, required=True, help="Output merged parquet path.")
    args = ap.parse_args()

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required to read/write parquet") from exc

    in_paths = [_resolve(p) for p in list(args.index_parquets)]
    for p in in_paths:
        if not p.is_file():
            raise FileNotFoundError(f"missing input parquet: {p}")

    ds_overrides = _parse_datasets_arg(str(args.datasets), n=len(in_paths))

    required_cols = ["split", "sequence", "frame", "main_agent", "coop_agents", "baseline_distance_m"]
    wanted_cols = required_cols + ["dataset", "has_assets"]

    dfs = []
    sources: List[Dict[str, Any]] = []
    for p, ds_override in zip(in_paths, ds_overrides):
        # Best-effort column projection (works with pyarrow backend).
        try:
            import pyarrow.parquet as pq  # type: ignore

            schema = pq.read_schema(p)
            available = set(schema.names)
            cols = [c for c in wanted_cols if c in available]
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)

        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"input index parquet missing required column {c!r}: {p}")

        df2 = df.copy()
        if ds_override is not None:
            df2["dataset"] = str(ds_override)
        elif "dataset" not in df2.columns:
            df2["dataset"] = None

        dfs.append(df2)
        sources.append(
            {
                "path": str(p),
                "rows": int(len(df2)),
                "dataset_override": str(ds_override) if ds_override is not None else None,
                "has_dataset_col": bool("dataset" in df.columns),
            }
        )

    out_df = pd.concat(dfs, axis=0, ignore_index=True, copy=False)

    if bool(args.require_dataset):
        ds = out_df.get("dataset")
        if ds is None:
            raise ValueError("--require_dataset set but merged df has no 'dataset' column (unexpected)")
        # Treat empty strings and NaN as missing.
        missing = ds.isna() | (ds.astype(str).str.strip() == "") | (ds.astype(str).str.lower() == "nan")
        num_missing = int(missing.sum())
        if num_missing:
            raise ValueError(
                f"--require_dataset set but merged index has {num_missing} rows with missing dataset id. "
                "Provide --datasets overrides or ensure each input parquet has a non-empty 'dataset' column."
            )

    out_path = _resolve(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    manifest = {"out_parquet": str(out_path), "num_rows": int(len(out_df)), "sources": sources}
    (out_path.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] wrote merged index parquet: {out_path} (rows={len(out_df)})")
    print(f"[OK] manifest: {out_path.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()

