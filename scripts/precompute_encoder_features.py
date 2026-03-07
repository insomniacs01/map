# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Precompute and cache encoder features for MapAnything datasets.

Usage:
  python scripts/precompute_encoder_features.py \
    machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v_short \
    train_params.max_num_of_imgs_per_gpu=8 \
    +cache_dir=/path/to/cache +cache_dtype=fp16 +max_samples=800
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mapanything import datasets as datasets_module
from mapanything.datasets import get_test_data_loader
from mapanything.datasets.base.base_dataset import view_name
from mapanything.models import init_model

log = logging.getLogger(__name__)


def _resolve_cache_dtype(cache_dtype: str) -> torch.dtype:
    cache_dtype = str(cache_dtype).lower()
    if cache_dtype == "fp16":
        return torch.float16
    if cache_dtype == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log.warning("bf16 not supported; falling back to fp16.")
        return torch.float16
    if cache_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported cache_dtype={cache_dtype}")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def precompute(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    if not hasattr(cfg, "cache_dir"):
        raise ValueError("Provide +cache_dir=... to store cached features.")

    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dtype = _resolve_cache_dtype(getattr(cfg, "cache_dtype", "fp16"))
    max_samples = getattr(cfg, "max_samples", None)
    overwrite = bool(getattr(cfg, "cache_overwrite", False))
    log_freq = int(getattr(cfg, "cache_log_freq", 50))
    num_workers = int(getattr(cfg, "cache_num_workers", cfg.dataset.num_workers))

    dataset = eval(cfg.dataset.train_dataset, vars(datasets_module))
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    data_loader = get_test_data_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_mem=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(
        cfg.model.model_str,
        cfg.model.model_config,
        torch_hub_force_reload=cfg.model.torch_hub_force_reload,
    ).to(device)
    model.eval()

    cache_model_ckpt = getattr(cfg, "cache_model_ckpt", None)
    if cache_model_ckpt:
        ckpt = torch.load(cache_model_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(
            "Loaded cache model checkpoint. missing=%s unexpected=%s",
            missing,
            unexpected,
        )

    total_written = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            encoder_views = []
            for view in batch:
                encoder_views.append(
                    {
                        "img": view["img"].to(device, non_blocking=True),
                        "data_norm_type": view["data_norm_type"],
                    }
                )
            with torch.autocast(
                "cuda",
                enabled=(device.type == "cuda" and cache_dtype != torch.float32),
                dtype=cache_dtype,
            ):
                encoded = model._encode_n_views(encoder_views)

            for view_idx, feats in enumerate(encoded):
                feats_cpu = feats.detach().to("cpu")
                if feats_cpu.dtype != cache_dtype:
                    feats_cpu = feats_cpu.to(cache_dtype)
                batch_size = feats_cpu.shape[0]
                for b in range(batch_size):
                    sample_name = view_name(batch[view_idx], batch_index=b)
                    cache_path = cache_root / f"{sample_name}.pt"
                    if cache_path.exists() and not overwrite:
                        continue
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(feats_cpu[b], cache_path)
                    total_written += 1

            if log_freq and (batch_idx + 1) % log_freq == 0:
                log.info(
                    "Processed %d batches, cached %d feature tensors.",
                    batch_idx + 1,
                    total_written,
                )
            if max_samples is not None and (batch_idx + 1) >= int(max_samples):
                break

    log.info("Done. Cached %d feature tensors under %s", total_written, cache_root)


if __name__ == "__main__":
    precompute()
