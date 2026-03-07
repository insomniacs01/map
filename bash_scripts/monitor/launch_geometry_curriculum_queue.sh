#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLASH_DIR="/J6P-perception/yijinxiong_workspace/yijinxiong/clash"

if [ "${GEOMQ_USE_PROXY:-1}" = "1" ] && [ -d "${CLASH_DIR}" ]; then
  (
    cd "${CLASH_DIR}"
    ./status.sh >/dev/null 2>&1 || ./start.sh >/dev/null 2>&1 || true
  )
  if [ -f "${CLASH_DIR}/env.sh" ]; then
    # shellcheck disable=SC1090
    source "${CLASH_DIR}/env.sh" || true
  fi
fi

export GEOMQ_REPO_ROOT="${REPO_ROOT}"
export GEOMQ_QUEUE_FILE="${GEOMQ_QUEUE_FILE:-${REPO_ROOT}/experiments/long_runs/auto_queue_geom_curriculum_final_v1.txt}"
export GEOMQ_TAG="${GEOMQ_TAG:-$(date +%Y%m%d_%H%M%S)_geom_curriculum_v1}"
export GEOMQ_LOCAL_ROOT="${GEOMQ_LOCAL_ROOT:-/tmp/mapanything_runs/${GEOMQ_TAG}}"
export GEOMQ_LOG_ROOT="${GEOMQ_LOG_ROOT:-${REPO_ROOT}/experiments/long_runs/${GEOMQ_TAG}_watchdog}"
export GEOMQ_BASELINE_SUMMARY="${GEOMQ_BASELINE_SUMMARY:-${REPO_ROOT}/eval_runs/geom_fairlock_test500_20260226_promoted_v2/summary_test.json}"
export GEOMQ_START_CKPT="${GEOMQ_START_CKPT:-${REPO_ROOT}/experiments/local_runs/20260226_srecover_near_from_midpose12_lr1e5/pinhole_pose_depth_scale/checkpoint-best.pth}"
export GEOMQ_INTERVAL_SEC="${GEOMQ_INTERVAL_SEC:-120}"
export GEOMQ_IDLE_UTIL="${GEOMQ_IDLE_UTIL:-10}"
export GEOMQ_IDLE_ITERS="${GEOMQ_IDLE_ITERS:-2}"
export GEOMQ_NUM_WORKERS="${GEOMQ_NUM_WORKERS:-8}"
export GEOMQ_WATCH_IDLE_SEC="${GEOMQ_WATCH_IDLE_SEC:-3600}"
export GEOMQ_WATCH_INTERVAL_SEC="${GEOMQ_WATCH_INTERVAL_SEC:-60}"
export GEOMQ_WATCH_IDLE_UTIL="${GEOMQ_WATCH_IDLE_UTIL:-5}"
export MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION="${MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION:-1}"
export ACTIVE_REGEX="${ACTIVE_REGEX:-torchrun|torch.distributed.launch|scripts/train.py|batch_eval.py}"
export STATUS_FILE="${STATUS_FILE:-${GEOMQ_LOG_ROOT}/STATUS.md}"
export SUMMARY_ROOT="${SUMMARY_ROOT:-${GEOMQ_LOG_ROOT}/auto_queue_summaries}"
export GATE_AUTO_FIX="0"
export STOP_ON_FAIL="0"
export ADVANCE_ON_FAIL="1"

mkdir -p "${GEOMQ_LOCAL_ROOT}" "${GEOMQ_LOG_ROOT}" "${SUMMARY_ROOT}"

if [ ! -f "${GEOMQ_QUEUE_FILE}" ]; then
  echo "[geom-queue] queue file missing: ${GEOMQ_QUEUE_FILE}" >&2
  exit 3
fi
if [ ! -f "${GEOMQ_START_CKPT}" ]; then
  echo "[geom-queue] start checkpoint missing: ${GEOMQ_START_CKPT}" >&2
  exit 4
fi
if [ ! -f "${GEOMQ_BASELINE_SUMMARY}" ]; then
  echo "[geom-queue] baseline summary missing: ${GEOMQ_BASELINE_SUMMARY}" >&2
  exit 5
fi

echo "[geom-queue] repo_root=${GEOMQ_REPO_ROOT}"
echo "[geom-queue] queue_file=${GEOMQ_QUEUE_FILE}"
echo "[geom-queue] tag=${GEOMQ_TAG}"
echo "[geom-queue] local_root=${GEOMQ_LOCAL_ROOT}"
echo "[geom-queue] log_root=${GEOMQ_LOG_ROOT}"
echo "[geom-queue] start_ckpt=${GEOMQ_START_CKPT}"
echo "[geom-queue] baseline=${GEOMQ_BASELINE_SUMMARY}"
echo "[geom-queue] note=watchdog stays alive after queue completion and hot-reloads appended queue items"

bash "${SCRIPT_DIR}/auto_queue_watchdog.sh" \
  "${GEOMQ_QUEUE_FILE}" \
  "${GEOMQ_LOG_ROOT}" \
  "${GEOMQ_INTERVAL_SEC}" \
  "${GEOMQ_IDLE_UTIL}" \
  "${GEOMQ_IDLE_ITERS}"
