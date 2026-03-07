#!/bin/bash

# Local 1-GPU VGGT finetuning suite on OPV2V WITH metric scale head:
#   (1) cooperative multi-agent (2 agents, 8 views)
#   (2) single-agent (4 views)
#
# The metric scale head outputs a scalar `metric_scaling_factor` used to scale
# depth + pose translations. This can help VGGT recover metric scale faster.
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_vggt_suite_local_1gpu_scalehead.sh
#   bash .../opv2v_vggt_suite_local_1gpu_scalehead.sh train_params.epochs=50

set -euo pipefail

EXTRA_ARGS=${@:1}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="$(date +%Y%m%d_%H%M%S)"
COOP_DIR="$(pwd)/experiments/vggt/training/opv2v_vggt_suite_local_1gpu_scalehead/${RUN_ID}/coop"
SINGLE_DIR="$(pwd)/experiments/vggt/training/opv2v_vggt_suite_local_1gpu_scalehead/${RUN_ID}/single"

echo "[RUN] coop -> ${COOP_DIR}"
python scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=true \
  dataset=opv2v_coop_ft_2a8v_vggt_500 \
  dataset.num_workers=4 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_coop_8v_long_scalehead \
  ${EXTRA_ARGS} \
  hydra.run.dir="${COOP_DIR}"

echo "[RUN] single -> ${SINGLE_DIR}"
python scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=true \
  dataset=opv2v_ft_vggt_1k \
  dataset.num_workers=4 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_single_4v_long_scalehead \
  ${EXTRA_ARGS} \
  hydra.run.dir="${SINGLE_DIR}"
