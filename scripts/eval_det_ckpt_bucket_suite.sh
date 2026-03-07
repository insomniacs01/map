#!/bin/bash
set -euo pipefail

# Run the baseline-distance bucket protocol for det/e2e on multiple fixed contracts
# (representative + stress).
#
# This is the det counterpart of `scripts/eval_geom_ckpt_bucket_suite.sh`.
#
# Usage:
#   bash map-anything/scripts/eval_det_ckpt_bucket_suite.sh <ckpt.pth> <det_head_cfg> <out_root> [cuda_device=0]
#
# Defaults:
#   CONTRACTS = nearest + nearest_stress (Test200)
#   Protocol defaults are set in `eval_det_ckpt_contract_with_buckets.sh` (deployment-like).

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <ckpt_path> <det_head_cfg> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
DET_HEAD_CFG="$2"
OUT_ROOT="$3"
CUDA_DEV="${4:-0}"

CONTRACTS_DEFAULT="${REPO_ROOT}/eval_runs/frames_test200_nearest_seed42.json ${REPO_ROOT}/eval_runs/frames_test200_nearest_stress_seed42.json"
CONTRACTS="${CONTRACTS:-${CONTRACTS_DEFAULT}}"

mkdir -p "${OUT_ROOT}"

read -r -a _contracts <<< "${CONTRACTS}"
for contract in "${_contracts[@]}"; do
  stem="$(basename "${contract}" .json)"
  out_dir="${OUT_ROOT}/${stem}"
  echo "[SUITE] contract=${contract} out=${out_dir}"
  bash "${REPO_ROOT}/scripts/eval_det_ckpt_contract_with_buckets.sh" \
    "${CKPT_PATH}" \
    "${DET_HEAD_CFG}" \
    "${contract}" \
    "${out_dir}" \
    "${CUDA_DEV}"
done

echo "[DONE] suite out_root=${OUT_ROOT}"

