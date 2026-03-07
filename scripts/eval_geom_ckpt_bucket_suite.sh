#!/bin/bash
set -euo pipefail

# Run the bucket protocol on multiple fixed contracts (e.g. representative + stress).
#
# Usage:
#   bash map-anything/scripts/eval_geom_ckpt_bucket_suite.sh <ckpt.pth> <out_root> [cuda_device=0]
#
# Defaults:
#   CONTRACTS = nearest + nearest_stress (Test500)
#   Each contract is evaluated under the deployment-like protocol (calibrated_sfm, coop-first).

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <ckpt_path> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
OUT_ROOT="$2"
CUDA_DEV="${3:-0}"

MODEL_ARCH="${MODEL_ARCH:-}"
if [ -z "${MODEL_ARCH}" ]; then
  if [[ "${CKPT_PATH}" == *"/experiments/vggt/"* ]] || [[ "${CKPT_PATH}" == *"/vggt/"* ]]; then
    MODEL_ARCH="vggt"
  else
    MODEL_ARCH="mapanything"
  fi
fi

CONTRACTS_DEFAULT="${REPO_ROOT}/eval_runs/frames_test500_nearest_seed42.json ${REPO_ROOT}/eval_runs/frames_test500_nearest_stress_seed42.json"
CONTRACTS="${CONTRACTS:-${CONTRACTS_DEFAULT}}"

mkdir -p "${OUT_ROOT}"

read -r -a _contracts <<< "${CONTRACTS}"
for contract in "${_contracts[@]}"; do
  stem="$(basename "${contract}" .json)"
  out_dir="${OUT_ROOT}/${stem}"
  echo "[SUITE] contract=${contract} out=${out_dir}"
  MODEL_ARCH="${MODEL_ARCH}" \
  bash "${REPO_ROOT}/scripts/eval_geom_ckpt_contract_with_buckets.sh" \
    "${CKPT_PATH}" \
    "${contract}" \
    "${out_dir}" \
    "${CUDA_DEV}"
done

echo "[DONE] suite out_root=${OUT_ROOT}"
