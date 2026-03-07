#!/bin/bash

# Multi-car cylindrical OPV2V fine-tuning helper.

set -euo pipefail

NUM_GPUS=${1:-4}
EXTRA_ARGS=${@:2}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=local3090 \
    model=mapanything \
    dataset=opv2v_cyl_coop_ft \
    dataset.num_workers=12 \
    dataset.principal_point_centered=true \
    train_params=opv2v_cyl_coop \
    ${EXTRA_ARGS} \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/opv2v_cyl_coop/${now:%Y%m%d_%H%M%S}'
