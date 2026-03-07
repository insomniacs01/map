#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"
run_dir="experiments/vggt/training/opv2v_single_vggt_pose_metric_long_a800_b1acc4/${timestamp}"
mkdir -p "${run_dir}"

echo "Run dir: ${run_dir}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"

python scripts/train.py \
  machine=local_a800 \
  model=vggt \
  dataset=opv2v_ft_a800 \
  dataset.num_workers=2 \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_single_4v_long \
  hydra.run.dir="${run_dir}"
