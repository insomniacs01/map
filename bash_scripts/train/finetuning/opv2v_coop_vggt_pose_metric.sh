#!/bin/bash

# OPV2V cooperative (>=2 agents) VGGT finetuning without an explicit metric scale head.
#
# This variant keeps OPV2V's metric scale by using a loss that *does not* normalize
# camera translations / depths for metric-scale samples (norm_mode='?avg_dis').
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_coop_vggt_pose_metric.sh 1
#
# Notes:
# - Downloads `facebook/VGGT-1B` from HuggingFace on first run (set proxy if needed).
# - Uses `dataset=opv2v_coop_ft_a800_2a8v` (8 fixed views) and supervises metric pose+depth.

set -euo pipefail

NUM_GPUS=${1:-1}
EXTRA_ARGS=${@:2}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node "${NUM_GPUS}" \
  scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=false \
  dataset=opv2v_coop_ft_a800_2a8v \
  dataset.num_workers=12 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_coop_8v \
  ${EXTRA_ARGS} \
  hydra.run.dir='${root_experiments_dir}/vggt/training/opv2v_coop_vggt_pose_metric/${now:%Y%m%d_%H%M%S}'

