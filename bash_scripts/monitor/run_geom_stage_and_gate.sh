#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STAGE=""
TRAIN_SCRIPT=""
PRETRAINED_CKPT=""
RUN_SUBDIR="pinhole_pose_depth_scale"
MAX_IMGS_PER_GPU=""
NUM_WORKERS=""
EPOCHS=""
LR=""
MIN_LR=""
FULL_EVAL_FREQ=""
WARMUP_EPOCHS=""
INFO_SHARING_GC=""
PRED_HEAD_GC=""
ENCODER_GC=""
RESUME=""
EXTRA_ARGS=""
STRICT_GATE="0"

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --train-script) TRAIN_SCRIPT="$2"; shift 2 ;;
    --pretrained) PRETRAINED_CKPT="$2"; shift 2 ;;
    --run-subdir) RUN_SUBDIR="$2"; shift 2 ;;
    --max-imgs) MAX_IMGS_PER_GPU="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --min-lr) MIN_LR="$2"; shift 2 ;;
    --full-eval-freq) FULL_EVAL_FREQ="$2"; shift 2 ;;
    --warmup-epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
    --info-sharing-gc) INFO_SHARING_GC="$2"; shift 2 ;;
    --pred-head-gc) PRED_HEAD_GC="$2"; shift 2 ;;
    --encoder-gc) ENCODER_GC="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --strict-gate) STRICT_GATE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${STAGE}" ] || [ -z "${TRAIN_SCRIPT}" ] || [ -z "${PRETRAINED_CKPT}" ]; then
  echo "Usage: $0 --stage <name> --train-script <path> --pretrained <ckpt> [opts]" >&2
  exit 2
fi

if [[ "${TRAIN_SCRIPT}" != /* ]]; then
  TRAIN_SCRIPT="${REPO_ROOT}/${TRAIN_SCRIPT}"
fi
if [[ "${PRETRAINED_CKPT}" != /* ]]; then
  PRETRAINED_CKPT="${REPO_ROOT}/${PRETRAINED_CKPT}"
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[geom-stage] train script missing: ${TRAIN_SCRIPT}" >&2
  exit 3
fi
if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "[geom-stage] pretrained checkpoint missing: ${PRETRAINED_CKPT}" >&2
  exit 4
fi

GEOMQ_REPO_ROOT="${GEOMQ_REPO_ROOT:-${REPO_ROOT}}"
GEOMQ_TAG="${GEOMQ_TAG:?GEOMQ_TAG is required}"
GEOMQ_LOCAL_ROOT="${GEOMQ_LOCAL_ROOT:?GEOMQ_LOCAL_ROOT is required}"
GEOMQ_BASELINE_SUMMARY="${GEOMQ_BASELINE_SUMMARY:-${REPO_ROOT}/eval_runs/geom_fairlock_test500_20260226_promoted_v2/summary_test.json}"

export NAS_RUN_ROOT="${GEOMQ_REPO_ROOT}/experiments/long_runs/${GEOMQ_TAG}_${STAGE}"
export LOCAL_ROOT="${GEOMQ_LOCAL_ROOT}/${STAGE}"
export PRETRAINED_CKPT
export MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION="${MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION:-1}"

[ -n "${MAX_IMGS_PER_GPU}" ] && export MAX_IMGS_PER_GPU
[ -n "${NUM_WORKERS}" ] && export NUM_WORKERS
[ -n "${EPOCHS}" ] && export EPOCHS
[ -n "${LR}" ] && export LR
[ -n "${MIN_LR}" ] && export MIN_LR
[ -n "${FULL_EVAL_FREQ}" ] && export FULL_EVAL_FREQ
[ -n "${WARMUP_EPOCHS}" ] && export WARMUP_EPOCHS
[ -n "${INFO_SHARING_GC}" ] && export INFO_SHARING_GC
[ -n "${PRED_HEAD_GC}" ] && export PRED_HEAD_GC
[ -n "${ENCODER_GC}" ] && export ENCODER_GC
[ -n "${RESUME}" ] && export RESUME
[ -n "${EXTRA_ARGS}" ] && export MLP_ENTRY_EXTRA_ARGS="${EXTRA_ARGS}"

mkdir -p "${NAS_RUN_ROOT}" "${LOCAL_ROOT}"
echo "[geom-stage] stage=${STAGE}"
echo "[geom-stage] train_script=${TRAIN_SCRIPT}"
echo "[geom-stage] pretrained=${PRETRAINED_CKPT}"
echo "[geom-stage] nas_run_root=${NAS_RUN_ROOT}"
echo "[geom-stage] local_root=${LOCAL_ROOT}"

bash "${TRAIN_SCRIPT}"

CKPT="${NAS_RUN_ROOT}/${RUN_SUBDIR}/checkpoint-best.pth"
OUT_ROOT="${GEOMQ_REPO_ROOT}/eval_runs/${GEOMQ_TAG}_${STAGE}_test500"
if [ ! -f "${CKPT}" ]; then
  echo "[geom-stage] missing checkpoint after training: ${CKPT}" >&2
  exit 5
fi

GEOM_GATE_STRICT="${STRICT_GATE}" GEOM_GATE_BASELINE="${GEOMQ_BASELINE_SUMMARY}" \
  bash "${GEOMQ_REPO_ROOT}/scripts/eval_geom_ckpt_test500_gate.sh" "${CKPT}" "${OUT_ROOT}" geom_model
