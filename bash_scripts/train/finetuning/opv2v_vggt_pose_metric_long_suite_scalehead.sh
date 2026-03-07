#!/bin/bash

# Full VGGT finetuning suite on OPV2V WITH metric scale head:
#   (1) cooperative multi-agent (>=2 agents, 8 views)
#   (2) single-agent (4 views)
#
# The metric scale head predicts a global scalar used to scale depth + pose translations.
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_vggt_pose_metric_long_suite_scalehead.sh 2
#
# Override epochs etc. via:
#   bash .../opv2v_vggt_pose_metric_long_suite_scalehead.sh 2 train_params.epochs=50

set -euo pipefail

NUM_GPUS=${1:-2}
EXTRA_ARGS=${@:2}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

COOP_DIR="$(pwd)/experiments/vggt/training/opv2v_coop_vggt_pose_metric_long_scalehead/${RUN_ID}"
SINGLE_DIR="$(pwd)/experiments/vggt/training/opv2v_single_vggt_pose_metric_long_scalehead/${RUN_ID}"

echo "[RUN] coop -> ${COOP_DIR}"
torchrun --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=true \
  dataset=opv2v_coop_ft_2a8v \
  dataset.num_workers=12 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_coop_8v_long_scalehead \
  ${EXTRA_ARGS} \
  hydra.run.dir="${COOP_DIR}"

echo "[RUN] single -> ${SINGLE_DIR}"
torchrun --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.enable_metric_scale_head=true \
  dataset=opv2v_ft \
  dataset.num_workers=12 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_single_4v_long_scalehead \
  ${EXTRA_ARGS} \
  hydra.run.dir="${SINGLE_DIR}"
