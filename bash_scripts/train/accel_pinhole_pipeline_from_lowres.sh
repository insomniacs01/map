#!/bin/bash

set -euo pipefail

ROOT="${1:?Usage: $0 <run_root> <pinhole_lowres_ckpt>}"
LOWRES_CKPT="${2:?Usage: $0 <run_root> <pinhole_lowres_ckpt>}"

TORCHRUN=/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/torchrun
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$(mkdir -p "${ROOT}" && cd "${ROOT}" && pwd)"

run_phase() {
  local name=$1
  local args=$2
  local out_dir="${ROOT}/${name}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$(pwd) "${TORCHRUN}" --nproc_per_node=4 --master_port=29630 \
    scripts/train.py ${args} hydra.run.dir="${out_dir}" 2>&1 | tee "${out_dir}/launch.log"
}

# Phase 2: e2e full-res long run (20 epochs), resume from low-res checkpoint
run_phase \
  "pinhole_e2e_fullres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v \
   loss=opv2v_det_e2e_vehicle_aux_v1 train_params=opv2v_det_e2e_ft model/det_head=bev_centernet_wide_e2e_v1 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=20 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=${LOWRES_CKPT} \
   dataset.num_workers=8"

# Phase 3: det-head-only low-res warmup (2 epochs)
run_phase \
  "pinhole_head_lowres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v_lowres \
   loss=opv2v_det_only_wide_v4 train_params=opv2v_det_head_only model/det_head=bev_centernet_wide_v4 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=2 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=$(pwd)/experiments/opv2v_full_runs/20260125_123515/pinhole_coop_det_e2e/checkpoint-last.pth \
   dataset.num_workers=8"

# Phase 4: det-head-only full-res long run (10 epochs)
run_phase \
  "pinhole_head_fullres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v \
   loss=opv2v_det_only_wide_v4 train_params=opv2v_det_head_only model/det_head=bev_centernet_wide_v4 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=10 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=${ROOT}/pinhole_head_lowres/checkpoint-last.pth \
   dataset.num_workers=8"

echo "Pinhole pipeline finished. Logs under ${ROOT}."
