#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT_INPUT="${1:-${REPO_ROOT}/experiments/local_runs/$(date +%Y%m%d_%H%M%S)_pinhole_det_head_only_single}"
mkdir -p "${OUT_ROOT_INPUT}"
OUT_ROOT="$(cd "${OUT_ROOT_INPUT}" && pwd)"

PYTHON=/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python
TORCHRUN=/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/torchrun
CUDA_DEV=${CUDA_DEV:-0}
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-29641}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_IMGS_PER_GPU=${MAX_IMGS_PER_GPU:-4}
ACCUM_ITER=${ACCUM_ITER:-4}
EPOCHS=${EPOCHS:-3}
PRINT_FREQ=${PRINT_FREQ:-20}
DATASET=${DATASET:-opv2v_coop_det_ft_2a8v_pair_nearest_deploy_full}
LOSS_CFG=${LOSS_CFG:-opv2v_det_only_wide_v5}
DET_HEAD_CFG=${DET_HEAD_CFG:-bev_centernet_wide_e2e_v1_predpose_aligngt0}
TRAIN_PARAMS=${TRAIN_PARAMS:-opv2v_det_head_only}
MODEL_TASK=${MODEL_TASK:-calibrated_sfm}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-${REPO_ROOT}/experiments/local_runs/20260226_srecover_near_from_midpose12_lr1e5/pinhole_pose_depth_scale/checkpoint-best.pth}
RUN_DIR="${OUT_ROOT}/pinhole_det_head_only"
GATE_EVAL=${GATE_EVAL:-1}
FINAL_EVAL=${FINAL_EVAL:-1}

resolve_ckpt_path() {
  local input_path="$1"
  local workspace_root
  workspace_root="$(cd "${REPO_ROOT}/.." && pwd)"

  if [[ -z "${input_path}" ]]; then
    return 1
  fi
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
    return 0
  fi
  if [ -e "${input_path}" ]; then
    readlink -f "${input_path}"
    return 0
  fi
  if [ -e "${REPO_ROOT}/${input_path}" ]; then
    readlink -f "${REPO_ROOT}/${input_path}"
    return 0
  fi
  if [ -e "${workspace_root}/${input_path}" ]; then
    readlink -f "${workspace_root}/${input_path}"
    return 0
  fi
  if [[ "${input_path}" == map-anything/* ]]; then
    printf '%s\n' "${workspace_root}/${input_path}"
    return 0
  fi
  printf '%s\n' "${REPO_ROOT}/${input_path}"
}

PRETRAIN_CKPT="$(resolve_ckpt_path "${PRETRAIN_CKPT}")"

mkdir -p "${RUN_DIR}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="${TORCH_HOME:-/J6P-perception/yijinxiong_workspace/.cache/torch}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
WORK_TMP_ROOT="${WORK_TMP_ROOT:-${REPO_ROOT}/experiments/local_tmp}"
mkdir -p "${WORK_TMP_ROOT}" || true
export TMPDIR="${TMPDIR:-${WORK_TMP_ROOT}}"
export TORCHELASTIC_TMPDIR="${TORCHELASTIC_TMPDIR:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}" "${TORCHELASTIC_TMPDIR}" || true
export MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION="${MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION:-1}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_DEV}" PYTHONPATH="${REPO_ROOT}" \
  "${TORCHRUN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  scripts/train.py \
    machine=local_a800 \
    model=mapanything_det \
    dataset="${DATASET}" \
    model/task="${MODEL_TASK}" \
    model/det_head="${DET_HEAD_CFG}" \
    loss="${LOSS_CFG}" \
    train_params="${TRAIN_PARAMS}" \
    train_params.max_num_of_imgs_per_gpu="${MAX_IMGS_PER_GPU}" \
    train_params.accum_iter="${ACCUM_ITER}" \
    train_params.epochs="${EPOCHS}" \
    train_params.eval_freq=0 \
    train_params.full_eval_freq=0 \
    train_params.resume=false \
    train_params.print_freq="${PRINT_FREQ}" \
    model.model_config.pretrained_checkpoint_path="${PRETRAIN_CKPT}" \
    dataset.num_workers="${NUM_WORKERS}" \
    hydra.run.dir="${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/launch.log"

pick_ckpt() {
  for p in "${RUN_DIR}/checkpoint-final.pth" "${RUN_DIR}/checkpoint-best.pth" "${RUN_DIR}/checkpoint-last.pth"; do
    if [ -f "$p" ]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

CKPT_PATH="$(pick_ckpt)"
printf '%s\n' "${CKPT_PATH}" > "${RUN_DIR}/selected_ckpt.txt"
unset DET_HEAD_CKPT || true

if [ "${GATE_EVAL}" = "1" ]; then
  bash scripts/eval_det_ckpt_opv2v_tier.sh \
    "${CKPT_PATH}" \
    "${DET_HEAD_CFG}" \
    gate \
    "${RUN_DIR}/eval_gate500" \
    "${CUDA_DEV}"
fi

if [ "${FINAL_EVAL}" = "1" ]; then
  bash scripts/eval_det_ckpt_full2170.sh \
    "${CKPT_PATH}" \
    "${DET_HEAD_CFG}" \
    "${RUN_DIR}/eval_full2170" \
    "${CUDA_DEV}"
fi
