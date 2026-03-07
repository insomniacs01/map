#!/usr/bin/env python3
"""
Utility script to sanity-check OPV2V cylindrical cooperative datasets.

The script composes the standard Hydra config, instantiates the configured
train / val datasets, samples a few entries, and reports statistics such as:
    - number of fused agent views per sample
    - panorama resolution
    - non-ambiguous / valid mask ratios
    - camera_model consistency

Usage example:
    python scripts/validate_opv2v_cyl_setup.py \\
        --num-samples 4 \\
        --hydra-overrides machine=local3090,dataset=opv2v_cyl_coop_ft,\\\\
model=mapanything,loss=overall_loss,train_params=opv2v_cyl_coop
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, List

import sys

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mapanything.datasets as dataset_registry  # noqa: E402


def _compose_cfg(repo_root: Path, overrides: List[str]) -> DictConfig:
    config_dir = repo_root / "configs"
    overrides = overrides or []
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    return cfg


def _eval_dataset(dataset_expr: str, *, verbose: bool = False):
    context = vars(dataset_registry).copy()
    context["np"] = np
    context["__builtins__"] = __builtins__
    if verbose:
        print(f"  instantiating dataset via eval: {dataset_expr!r}")
    return eval(dataset_expr, context)


def _describe_sample(views: List[dict]):
    num_views = len(views)
    camera_models = {view.get("camera_model", "pinhole") for view in views}
    resolutions = {tuple(view["img"].shape[-2:][::-1]) for view in views}
    mask_means = [float(view["non_ambiguous_mask"].mean()) for view in views]
    valid_means = [float(view["valid_mask"].mean()) for view in views]
    labels = {view.get("label", "unknown") for view in views}
    return {
        "num_views": num_views,
        "camera_models": camera_models,
        "resolutions": resolutions,
        "mask_mean": statistics.mean(mask_means),
        "valid_mean": statistics.mean(valid_means),
        "labels": labels,
    }


def _summarize(samples: Iterable[dict]):
    samples = list(samples)
    if not samples:
        return {}
    summary = {
        "num_views_range": (min(s["num_views"] for s in samples), max(s["num_views"] for s in samples)),
        "camera_models": set.union(*[s["camera_models"] for s in samples]),
        "resolutions": set.union(*[s["resolutions"] for s in samples]),
        "mask_mean_range": (min(s["mask_mean"] for s in samples), max(s["mask_mean"] for s in samples)),
        "valid_mean_range": (min(s["valid_mean"] for s in samples), max(s["valid_mean"] for s in samples)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate OPV2V cylindrical dataset setup.")
    parser.add_argument(
        "--hydra-overrides",
        type=str,
        default="machine=local3090,dataset=opv2v_cyl_coop_ft,model=mapanything,loss=overall_loss,train_params=opv2v_cyl_coop",
        help="Comma separated Hydra overrides passed to configs/train.yaml",
    )
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random samples to inspect per split.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma separated dataset splits to check (train,val,test).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print instantiated dataset expressions.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    overrides = [ov.strip() for ov in args.hydra_overrides.split(",") if ov.strip()]
    cfg = _compose_cfg(repo_root, overrides)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    def _extract_dataset_expr(dataset_spec: str) -> List[str]:
        text = dataset_spec.replace("\n", " ").strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        exprs = []
        idx = 0
        while idx < len(text):
            at_pos = text.find("@", idx)
            if at_pos == -1:
                break
            cursor = at_pos + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            start = cursor
            depth = 0
            seen_paren = False
            while cursor < len(text):
                ch = text[cursor]
                if ch == "(":
                    depth += 1
                    seen_paren = True
                elif ch == ")":
                    depth -= 1
                    if seen_paren and depth == 0:
                        cursor += 1
                        break
                cursor += 1
            exprs.append(text[start:cursor].strip())
            idx = cursor
        if not exprs and text:
            exprs.append(text.strip())
        return exprs

    for split in splits:
        if split == "train":
            exprs = _extract_dataset_expr(cfg.dataset.train_dataset)
        elif split == "val":
            exprs = _extract_dataset_expr(cfg.dataset.test_dataset)
        else:  # test
            exprs = _extract_dataset_expr(cfg.dataset.test_dataset)
        if not exprs:
            print(f"[{split.upper()}] No dataset expression found in config.")
            continue
        dataset_expr = exprs[0].strip()
        if dataset_expr.startswith('"') and dataset_expr.endswith('"'):
            dataset_expr = dataset_expr[1:-1]
        if args.verbose:
            print(f"[{split.upper()}] dataset expr resolved to: {dataset_expr!r}")
        dataset = _eval_dataset(dataset_expr, verbose=args.verbose)

        rng = np.random.default_rng(seed=0)
        max_samples = min(args.num_samples, len(dataset))
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        sample_stats = []
        for raw_idx in indices:
            scene_idx = int(raw_idx)
            if isinstance(dataset.num_views, int):
                views = dataset[scene_idx]
            else:
                views = dataset[(scene_idx, 0, 0)]
            sample_stats.append(_describe_sample(views))
        summary = _summarize(sample_stats)
        print(f"[{split.upper()}] dataset={dataset_expr}")
        if not summary:
            print("  Empty dataset.")
            continue
        print(f"  num samples checked: {max_samples}/{len(dataset)}")
        print(f"  num_views range: {summary['num_views_range']}")
        print(f"  camera models: {sorted(summary['camera_models'])}")
        print(f"  panorama resolutions (WxH): {sorted(summary['resolutions'])}")
        print(f"  non-ambiguous mask mean range: {summary['mask_mean_range']}")
        print(f"  valid mask mean range: {summary['valid_mean_range']}")


if __name__ == "__main__":
    main()
