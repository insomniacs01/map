#!/usr/bin/env python3
"""
Export OPV2V cooperative cylindrical samples to self-contained HTML for easy inspection.

Example:
PYTHONPATH=$(pwd) python scripts/visualize_opv2v_cyl_html.py \
    --split train --index 0 \
    --output-html eval_runs/opv2v_cyl_precheck/train_idx0/panorama.html
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from plotly.subplots import make_subplots

import mapanything.datasets as dataset_registry
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _compose_cfg(overrides: List[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=overrides)


def _extract_dataset_exprs(dataset_spec: str) -> List[str]:
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


def _instantiate_dataset(dataset_expr: str):
    context = vars(dataset_registry).copy()
    context["np"] = np
    context["__builtins__"] = __builtins__
    return eval(dataset_expr, context)


def _fetch_views(dataset, sample_idx: int):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    if isinstance(dataset.num_views, int):
        return dataset[sample_idx]
    else:
        return dataset[(sample_idx, 0, 0)]


def _tensor_to_rgb(image_tensor, norm_type: str) -> np.ndarray:
    tensor = image_tensor.detach().cpu().float()
    norm_cfg = IMAGE_NORMALIZATION_DICT.get(norm_type)
    if norm_cfg is not None:
        mean = np.array(norm_cfg.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(norm_cfg.std, dtype=np.float32).reshape(3, 1, 1)
        tensor = tensor.numpy() * std + mean
    else:
        tensor = tensor.numpy()
    tensor = np.clip(tensor, 0.0, 1.0)
    tensor = np.transpose(tensor, (1, 2, 0))
    return (tensor * 255.0).astype(np.uint8)


def _make_figure(views, depth_clip_percentile: float = 99.0):
    rows = len(views)
    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=sum(
            [
                [
                    f"Agent {v.get('agent_id', idx)} RGB",
                    "Depth (m)",
                    "Non-ambiguous mask",
                ]
                for idx, v in enumerate(views)
            ],
            [],
        ),
        horizontal_spacing=0.04,
        vertical_spacing=0.08 if rows > 1 else 0.12,
    )
    meta_lines = []
    for row_idx, view in enumerate(views, start=1):
        rgb = _tensor_to_rgb(view["img"], view.get("data_norm_type", "identity"))
        depth_np = view["depthmap"]
        if isinstance(depth_np, np.ndarray):
            depth = depth_np[..., 0]
        else:
            depth = depth_np.detach().cpu().numpy()[..., 0]
        mask_np = view["non_ambiguous_mask"]
        if not isinstance(mask_np, np.ndarray):
            mask_np = mask_np.detach().cpu().numpy()

        depth_positive = depth[depth > 0]
        if depth_positive.size > 0:
            vmax = np.percentile(depth_positive, depth_clip_percentile)
        else:
            vmax = 1.0
        vmax = max(vmax, 1e-3)

        fig.add_trace(
            go.Image(z=rgb),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=np.clip(depth, 0, vmax),
                colorscale="Turbo",
                zmin=0,
                zmax=vmax,
                showscale=False,
            ),
            row=row_idx,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=mask_np.astype(float),
                colorscale=[[0, "black"], [1, "lime"]],
                zmin=0,
                zmax=1,
                showscale=False,
            ),
            row=row_idx,
            col=3,
        )

        meta_lines.append(
            f"View {row_idx}: agent={view.get('agent_id','?')} "
            f"mask_mean={mask_np.mean():.3f} valid_mean={view['valid_mask'].mean():.3f} "
            f"label={view.get('label','')}/{view.get('instance','')}"
        )

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        height=350 * rows,
        title="<br>".join(meta_lines),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Export OPV2V cylindrical panoramas to HTML.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--expr-index", type=int, default=0, help="Index of dataset expression for split (0=first).")
    parser.add_argument("--index", type=int, default=0, help="Sample index within dataset.")
    parser.add_argument("--output-html", type=str, required=True)
    parser.add_argument(
        "--hydra-overrides",
        type=str,
        default="machine=local3090,dataset=opv2v_cyl_coop_ft,model=mapanything,loss=overall_loss,train_params=opv2v_cyl_coop",
    )
    parser.add_argument("--depth-percentile", type=float, default=99.0, help="Clip depth visualization at percentile.")
    args = parser.parse_args()

    overrides = [ov.strip() for ov in args.hydra_overrides.split(",") if ov.strip()]
    cfg = _compose_cfg(overrides)

    if args.split == "train":
        dataset_spec = cfg.dataset.train_dataset
    else:
        dataset_spec = cfg.dataset.test_dataset
    exprs = _extract_dataset_exprs(dataset_spec)
    if not exprs:
        raise ValueError(f"No dataset expressions found for split {args.split}")
    expr_idx = min(max(args.expr_index, 0), len(exprs) - 1)
    dataset_expr = exprs[expr_idx]

    dataset = _instantiate_dataset(dataset_expr)
    views = _fetch_views(dataset, args.index)
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = _make_figure(views, depth_clip_percentile=args.depth_percentile)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
