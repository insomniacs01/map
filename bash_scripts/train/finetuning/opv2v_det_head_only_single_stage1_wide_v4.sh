#!/bin/bash

# OPV2V single-agent detection-head finetuning (BEV CenterNet) on top of the Stage1 MapAnything checkpoint.
#
# Usage:
#   cd map-anything
#   bash bash_scripts/train/finetuning/opv2v_det_head_only_single_stage1_wide_v4.sh
#
# Notes:
# - Uses GT poses as input (`model/task=posed_sfm`) and for det alignment (`point_pose_source=gt`).
# - If your environment cannot directly access the internet for torch hub, enable the local proxy first.

set -euo pipefail

EXTRA_ARGS=${@:1}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train.py \
  machine=local_a800 \
  model=mapanything_det \
  model/task=posed_sfm \
  model.pretrained=experiments/opv2v_ft_stage1/checkpoint-best.pth \
  model/det_head=bev_centernet_wide_v4 \
  dataset=opv2v_det_ft_a800_2k \
  dataset.num_workers=8 \
  dataset.principal_point_centered=true \
  loss=opv2v_det_only_wide_v2 \
  train_params=opv2v_det_head_only \
  train_params.max_num_of_imgs_per_gpu=32 \
  train_params.accum_iter=1 \
  train_params.epochs=10 \
  train_params.warmup_epochs=1 \
  train_params.eval_freq=999 \
  ${EXTRA_ARGS} \
  hydra.run.dir="$(pwd)/experiments/mapanything/training/opv2v_det_head_only_single_stage1_wide_v4/$(date +%Y%m%d_%H%M%S)"
