#!/bin/bash

# OPV2V single-agent (4 cameras) VGGT finetuning WITHOUT an explicit metric scale head.
#
# Metric scale is preserved by using a loss that does not normalize metric samples
# (norm_mode='?avg_dis') and supervising pose+depth in meters.
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_single_vggt_pose_metric.sh 2
#
# Notes:
# - Downloads `facebook/VGGT-1B` from HuggingFace on first run (set proxy if needed).
# - Uses `dataset=opv2v_ft` (4 fixed views).

set -euo pipefail

NUM_GPUS=${1:-2}
EXTRA_ARGS=${@:2}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

torchrun --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=false \
  dataset=opv2v_ft \
  dataset.num_workers=12 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_single_4v_long \
  ${EXTRA_ARGS} \
  hydra.run.dir="$(pwd)/experiments/vggt/training/opv2v_single_vggt_pose_metric_long/$(date +%Y%m%d_%H%M%S)"
