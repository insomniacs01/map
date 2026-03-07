#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"
run_dir="experiments/vggt/training/opv2v_coop_vggt_pose_metric_long_a800_b1acc2/${timestamp}"
mkdir -p "${run_dir}"

echo "Run dir: ${run_dir}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"

python scripts/train.py \
  machine=local_a800 \
  model=vggt \
  dataset=opv2v_coop_ft_a800_2a8v \
  dataset.num_workers=2 \
  loss=opv2v_vggt_pose_metric_loss_strong_pose \
  train_params=opv2v_vggt_coop_8v_long \
  hydra.run.dir="${run_dir}"
