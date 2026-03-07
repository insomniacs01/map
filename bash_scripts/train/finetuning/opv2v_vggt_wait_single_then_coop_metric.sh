#!/bin/bash

# Wait for a running single-agent OPV2V VGGT finetune job to finish, then launch the
# cooperative (2 agents / 8 views) finetune warm-started from the single-agent checkpoint.
#
# Usage:
#   # Start this right after launching the single-agent run (it will block until that run ends):
#   cd map-anything
#   nohup bash bash_scripts/train/finetuning/opv2v_vggt_wait_single_then_coop_metric.sh \
#     experiments/vggt/training/<single_run>/<timestamp> \
#     > experiments/vggt/training/<single_run>/<timestamp>/chain_coop.nohup.out 2>&1 &
#
# Optional: pass extra Hydra overrides after the single_run_dir, e.g.:
#   ... train_params.epochs=50

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <single_run_dir> [extra_hydra_args...]" >&2
  exit 1
fi

SINGLE_RUN_DIR="$1"
shift
EXTRA_ARGS=("$@")

# Resolve repo root even if launched from elsewhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SINGLE_RUN_DIR="$(readlink -f "$SINGLE_RUN_DIR")"
PID_FILE="${SINGLE_RUN_DIR}/nohup.pid"

if [ ! -d "$SINGLE_RUN_DIR" ]; then
  echo "Error: single_run_dir not found: ${SINGLE_RUN_DIR}" >&2
  exit 2
fi

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "${PID:-}" ]; then
    echo "[wait] single PID=${PID}"
    while kill -0 "$PID" 2>/dev/null; do
      echo "[wait] single still running: $(date '+%F %T')"
      sleep 60
    done
    echo "[wait] single finished: $(date '+%F %T')"
  fi
else
  echo "[wait] no nohup.pid found under ${SINGLE_RUN_DIR}; proceeding without waiting."
fi

CKPT="${SINGLE_RUN_DIR}/checkpoint-last.pth"
if [ ! -f "$CKPT" ]; then
  echo "[warn] ${CKPT} not found; searching for latest checkpoint-*.pth..."
  CKPT="$(ls -1t "${SINGLE_RUN_DIR}"/checkpoint-*.pth 2>/dev/null | head -n 1 || true)"
fi

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "Error: no checkpoint found in ${SINGLE_RUN_DIR}" >&2
  exit 3
fi

echo "[ckpt] warm-start from ${CKPT}"

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="$(date +%Y%m%d_%H%M%S)"
COOP_DIR="$(pwd)/experiments/vggt/training/opv2v_coop_vggt_pose_metric_warmstart/${RUN_ID}"

echo "[RUN] coop -> ${COOP_DIR}"
python scripts/train.py \
  machine=local_a800 \
  model=vggt \
  model.model_config.load_pretrained_weights=false \
  model.model_config.enable_metric_scale_head=false \
  model.model_config.pretrained_checkpoint_path="${CKPT}" \
  dataset=opv2v_coop_ft_2a8v_vggt_500 \
  dataset.num_workers=4 \
  dataset.principal_point_centered=true \
  loss=opv2v_vggt_pose_metric_loss \
  train_params=opv2v_vggt_coop_8v_long \
  train_params.resume=false \
  "${EXTRA_ARGS[@]}" \
  hydra.run.dir="${COOP_DIR}"

