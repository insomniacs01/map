#!/bin/bash

# Accelerated long-run plan: low-res warmup -> full-res for e2e + det-head-only
# Uses 4 GPUs per job (0-3 pinhole, 4-7 cyl) and runs phases sequentially.

set -euo pipefail

TORCHRUN=/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/torchrun
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="${1:-$(pwd)/experiments/long_runs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${ROOT}"

run_pair() {
  local pin_name=$1
  local cyl_name=$2
  local pin_args=$3
  local cyl_args=$4
  local pin_dir="${ROOT}/${pin_name}"
  local cyl_dir="${ROOT}/${cyl_name}"

  mkdir -p "${pin_dir}" "${cyl_dir}"

  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$(pwd) "${TORCHRUN}" --nproc_per_node=4 --master_port=29530 \
    scripts/train.py ${pin_args} hydra.run.dir="${pin_dir}" 2>&1 | tee "${pin_dir}/launch.log" &
  PIN_PID=$!

  CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=$(pwd) "${TORCHRUN}" --nproc_per_node=4 --master_port=29531 \
    scripts/train.py ${cyl_args} hydra.run.dir="${cyl_dir}" 2>&1 | tee "${cyl_dir}/launch.log" &
  CYL_PID=$!

  wait "${PIN_PID}" "${CYL_PID}"
}

echo "Root run dir: ${ROOT}"

# Phase 1: e2e low-res warmup (2 epochs)
run_pair \
  "pinhole_e2e_lowres" \
  "cyl_e2e_lowres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v_lowres \
   loss=opv2v_det_e2e_vehicle_aux_v1 train_params=opv2v_det_e2e_ft model/det_head=bev_centernet_wide_e2e_v1 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=2 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=$(pwd)/experiments/opv2v_full_runs/20260125_123515/pinhole_coop_det_e2e/checkpoint-last.pth \
   dataset.num_workers=8" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_cyl_coop_det_ft_r60_nearest_lowres \
   loss=opv2v_det_e2e_vehicle_aux_v1 train_params=opv2v_cyl_det_e2e_a800_2gpu model/det_head=bev_centernet_wide_e2e_v1 \
   train_params.max_num_of_imgs_per_gpu=8 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=2 \
   train_params.disable_cudnn_benchmark=true train_params.disable_tensorboard=true \
   train_params.resume=true \
   +train_params.resume_ckpt=$(pwd)/experiments/opv2v_full_runs/20260125_123515/cyl_det_e2e/checkpoint-last.pth \
   dataset.num_workers=8"

# Phase 2: e2e full-res long run (20 epochs)
run_pair \
  "pinhole_e2e_fullres" \
  "cyl_e2e_fullres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v \
   loss=opv2v_det_e2e_vehicle_aux_v1 train_params=opv2v_det_e2e_ft model/det_head=bev_centernet_wide_e2e_v1 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=20 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=${ROOT}/pinhole_e2e_lowres/checkpoint-last.pth \
   dataset.num_workers=8" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_cyl_coop_det_ft_r60_nearest \
   loss=opv2v_det_e2e_vehicle_aux_v1 train_params=opv2v_cyl_det_e2e_a800_2gpu model/det_head=bev_centernet_wide_e2e_v1 \
   train_params.max_num_of_imgs_per_gpu=8 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=20 \
   train_params.disable_cudnn_benchmark=true train_params.disable_tensorboard=true \
   train_params.resume=true \
   +train_params.resume_ckpt=${ROOT}/cyl_e2e_lowres/checkpoint-last.pth \
   dataset.num_workers=8"

# Phase 3: det-head-only low-res warmup (2 epochs)
run_pair \
  "pinhole_head_lowres" \
  "cyl_head_lowres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v_lowres \
   loss=opv2v_det_only_wide_v4 train_params=opv2v_det_head_only model/det_head=bev_centernet_wide_v4 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=2 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=$(pwd)/experiments/opv2v_full_runs/20260125_123515/pinhole_coop_det_e2e/checkpoint-last.pth \
   dataset.num_workers=8" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_cyl_coop_det_ft_r60_nearest_lowres \
   loss=opv2v_det_only_wide_v4 train_params=opv2v_cyl_det_head_only_a800_2gpu model/det_head=bev_centernet_wide_v4 \
   train_params.max_num_of_imgs_per_gpu=8 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=2 \
   train_params.disable_cudnn_benchmark=true train_params.disable_tensorboard=true \
   train_params.resume=true \
   +train_params.resume_ckpt=$(pwd)/experiments/opv2v_full_runs/20260125_123515/cyl_det_e2e/checkpoint-last.pth \
   dataset.num_workers=8"

# Phase 4: det-head-only full-res long run (10 epochs)
run_pair \
  "pinhole_head_fullres" \
  "cyl_head_fullres" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_coop_det_ft_2a8v \
   loss=opv2v_det_only_wide_v4 train_params=opv2v_det_head_only model/det_head=bev_centernet_wide_v4 \
   train_params.max_num_of_imgs_per_gpu=16 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=10 \
   train_params.disable_cudnn_benchmark=false train_params.disable_tensorboard=true train_params.resume=true \
   +train_params.resume_ckpt=${ROOT}/pinhole_head_lowres/checkpoint-last.pth \
   dataset.num_workers=8" \
  "machine=local_a800 model=mapanything_det dataset=opv2v_cyl_coop_det_ft_r60_nearest \
   loss=opv2v_det_only_wide_v4 train_params=opv2v_cyl_det_head_only_a800_2gpu model/det_head=bev_centernet_wide_v4 \
   train_params.max_num_of_imgs_per_gpu=8 train_params.accum_iter=1 train_params.eval_freq=0 train_params.epochs=10 \
   train_params.disable_cudnn_benchmark=true train_params.disable_tensorboard=true \
   train_params.resume=true \
   +train_params.resume_ckpt=${ROOT}/cyl_head_lowres/checkpoint-last.pth \
   dataset.num_workers=8"

echo "All phases finished. Logs under ${ROOT}."
