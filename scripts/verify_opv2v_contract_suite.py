#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ContractSpec:
    name: str
    split: str
    sample_size: int
    canonical_rel: str
    bucket_counts: tuple[int, ...] | None = None

    @property
    def canonical_path(self) -> Path:
        return REPO_ROOT / self.canonical_rel

    @property
    def is_stress(self) -> bool:
        return self.bucket_counts is not None

    @property
    def bucket_preset(self) -> str | None:
        return "fine_eval" if self.is_stress else None


SPECS: tuple[ContractSpec, ...] = (
    ContractSpec("train_full", "train", 6374, "eval_runs/frames_train6374_nearest_seed42_full.json"),
    ContractSpec("validate_full", "validate", 1980, "eval_runs/frames_validate1980_nearest_seed42_full.json"),
    ContractSpec("test_full", "test", 2170, "eval_runs/frames_test2170_nearest_seed42_full.json"),
    ContractSpec("test200_main", "test", 200, "eval_runs/frames_test200_nearest_seed42.json"),
    ContractSpec("test200_stress", "test", 200, "eval_runs/frames_test200_nearest_stress_seed42.json", (60, 70, 40, 10, 20)),
    ContractSpec("test500_main", "test", 500, "eval_runs/frames_test500_nearest_seed42.json"),
    ContractSpec("test500_stress", "test", 500, "eval_runs/frames_test500_nearest_stress_seed42.json", (150, 200, 103, 18, 29)),
    ContractSpec("test1000_main", "test", 1000, "eval_runs/frames_test1000_nearest_seed42.json"),
    ContractSpec("test1000_stress", "test", 1000, "eval_runs/frames_test1000_nearest_stress_seed42.json", (278, 373, 220, 39, 90)),
    ContractSpec("test2000_main", "test", 2000, "eval_runs/frames_test2000_nearest_seed42.json"),
    ContractSpec("test2000_stress", "test", 2000, "eval_runs/frames_test2000_nearest_stress_seed42.json", (600, 800, 471, 39, 90)),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table(rows: Sequence[dict[str, str]], headers: Sequence[str]) -> str:
    def esc(value: str) -> str:
        return value.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _bool_str(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _build_regen_cmd(python_exe: str, spec: ContractSpec, out_json: Path, index_dir: Path, seed: int) -> list[str]:
    cmd = [
        python_exe,
        str(REPO_ROOT / "scripts" / "make_opv2v_frames_contract.py"),
        "--split",
        spec.split,
        "--index_parquet",
        str(index_dir / f"index_{spec.split}.parquet"),
        "--out_json",
        str(out_json),
        "--sample_size",
        str(spec.sample_size),
        "--seed",
        str(seed),
        "--pair_policy",
        "index_nearest",
        "--require_assets",
    ]
    if spec.bucket_preset:
        cmd.extend(["--bucket_preset", spec.bucket_preset])
    if spec.bucket_counts:
        cmd.extend(["--bucket_counts", ",".join(str(v) for v in spec.bucket_counts)])
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable, help="Python executable used to run the generator.")
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=REPO_ROOT / "data" / "opv2v" / "opv2v_index",
        help="Directory containing index_<split>.parquet files.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--keep-regenerated", action="store_true")
    args = ap.parse_args(argv)

    rows: list[dict[str, str]] = []
    report_items: list[dict[str, Any]] = []
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="opv2v_contract_suite_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        kept_root = tmp_root if args.keep_regenerated else None
        for spec in SPECS:
            regen_path = tmp_root / spec.canonical_path.name
            cmd = _build_regen_cmd(args.python, spec, regen_path, args.index_dir, args.seed)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            generated = regen_path.exists()
            canonical_exists = spec.canonical_path.exists()
            ok = generated and canonical_exists and proc.returncode == 0
            details: dict[str, Any] = {
                "name": spec.name,
                "canonical_path": str(spec.canonical_path),
                "generated_path": str(regen_path),
                "generator_cmd": cmd,
                "generator_returncode": proc.returncode,
                "generator_stdout": proc.stdout,
                "generator_stderr": proc.stderr,
                "split": spec.split,
                "sample_size": spec.sample_size,
                "bucket_counts": list(spec.bucket_counts) if spec.bucket_counts else None,
                "pass": False,
            }
            if ok:
                canonical = _load_json(spec.canonical_path)
                regen = _load_json(regen_path)
                hash_v1_match = canonical.get("frames_hash_md5_v1") == regen.get("frames_hash_md5_v1")
                hash_v2_match = canonical.get("frames_hash_md5_v2") == regen.get("frames_hash_md5_v2")
                count_match = len(canonical.get("frames", [])) == len(regen.get("frames", []))
                bucket_match = canonical.get("bucket_target_counts") == regen.get("bucket_target_counts")
                ok = hash_v1_match and hash_v2_match and count_match and bucket_match
                details.update(
                    {
                        "canonical_hash_v1": canonical.get("frames_hash_md5_v1"),
                        "canonical_hash_v2": canonical.get("frames_hash_md5_v2"),
                        "regen_hash_v1": regen.get("frames_hash_md5_v1"),
                        "regen_hash_v2": regen.get("frames_hash_md5_v2"),
                        "canonical_frames": len(canonical.get("frames", [])),
                        "regen_frames": len(regen.get("frames", [])),
                        "canonical_bucket_counts": canonical.get("bucket_target_counts"),
                        "regen_bucket_counts": regen.get("bucket_target_counts"),
                        "hash_v1_match": hash_v1_match,
                        "hash_v2_match": hash_v2_match,
                        "count_match": count_match,
                        "bucket_match": bucket_match,
                        "pass": ok,
                    }
                )
            else:
                details["failure_reason"] = "generation failed or canonical artifact missing"
            if not ok:
                failures.append(spec.name)
            rows.append(
                {
                    "contract": spec.name,
                    "canonical": spec.canonical_path.name,
                    "frames": str(details.get("canonical_frames", "-")),
                    "hash_v2": str(details.get("canonical_hash_v2", "-")),
                    "stress_counts": json.dumps(details.get("canonical_bucket_counts")) if details.get("canonical_bucket_counts") is not None else "-",
                    "status": _bool_str(ok),
                }
            )
            report_items.append(details)
        if kept_root:
            print(f"[INFO] regenerated contracts kept under {kept_root}")

    md_lines = ["# OPV2V Canonical Contract Suite Verification", "", f"- Python: `{args.python}`", f"- Index dir: `{args.index_dir}`", f"- Seed: `{args.seed}`", "", _md_table(rows, ["contract", "canonical", "frames", "hash_v2", "stress_counts", "status"]), "", "## Notes", "", "- Full contracts are validated against the checked-in nearest/full artifacts.", "- Stress contracts require explicit bucket targets; default stress sampling is not canonical.", "- The legacy det quick contract `frames_test50_det_e2e_v5.json` is intentionally outside this nearest-suite verifier.", ""]

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(md_lines), encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"pass": not failures, "failures": failures, "items": report_items}, indent=2), encoding="utf-8")

    print("\n".join(md_lines))
    if failures:
        print(f"VERIFY_OPV2V_CONTRACT_SUITE: FAIL ({', '.join(failures)})", file=sys.stderr)
        return 1
    print("VERIFY_OPV2V_CONTRACT_SUITE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
