#!/bin/bash

# OPV2V cooperative (>=2 agents, fixed 2-agent/8-view layout) VGGT finetuning
# using a scale-aware loss (normalized geometry + explicit scale term) AND an
# explicit `metric_scale_head` in the VGGT wrapper.
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_coop_vggt_pose_scale_strong_scalehead.sh 2
#
# Override epochs/batch/etc via extra hydra args, e.g.:
#   bash .../opv2v_coop_vggt_pose_scale_strong_scalehead.sh 2 train_params.epochs=50 train_params.max_num_of_imgs_per_gpu=32

set -euo pipefail

NUM_GPUS=${1:-2}
EXTRA_ARGS=${@:2}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$(pwd)/experiments/vggt/training/opv2v_coop_vggt_pose_scale_strong_scalehead/${RUN_ID}"

torchrun --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=true \
  dataset=opv2v_coop_ft_2a8v \
  dataset.num_workers=12 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_scale_loss_strong_pose \
  train_params=opv2v_vggt_coop_8v_long_scalehead \
  ${EXTRA_ARGS} \
  hydra.run.dir="${OUT_DIR}"

