#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${1:-${REPO_ROOT}/experiments/long_runs/$(date +%Y%m%d_%H%M%S)_cyl_a800}"
RUN_ROOT="$(mkdir -p "${RUN_ROOT}" && cd "${RUN_ROOT}" && pwd)"

export PYTHONPATH="${REPO_ROOT}"

/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python -m torch.distributed.launch \
  --nproc_per_node "${MLP_WORKER_GPU}" \
  --master_addr "${MLP_WORKER_0_HOST}" \
  --node_rank "${MLP_ROLE_INDEX}" \
  --master_port "${MLP_WORKER_0_PORT}" \
  --nnodes "${MLP_WORKER_NUM}" \
  --use_env \
  "${REPO_ROOT}/scripts/train.py" \
  machine=local_a800 \
  model=mapanything_det \
  dataset=opv2v_cyl_coop_det_ft_r60_nearest \
  loss=opv2v_det_e2e_vehicle_aux_v1 \
  train_params=opv2v_cyl_det_e2e_a800_2gpu \
  model/det_head=bev_centernet_wide_e2e_v1 \
  train_params.max_num_of_imgs_per_gpu=8 \
  train_params.accum_iter=1 \
  train_params.eval_freq=0 \
  train_params.epochs=20 \
  train_params.disable_cudnn_benchmark=true \
  +train_params.disable_tensorboard=true \
  train_params.resume=true \
  +train_params.resume_ckpt=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/experiments/long_runs/20260126_002000_accel_split/cyl_e2e_fullres/checkpoint-last.pth \
  dataset.num_workers=8 \
  hydra.run.dir="${RUN_ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
