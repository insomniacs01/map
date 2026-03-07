#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <train_out_root> [cuda_device=0]"
  exit 1
fi

TRAIN_OUT_ROOT="$1"
CUDA_DEV="${2:-0}"
REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
VALIDATOR="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/scripts/validate_batch_eval_summary_protocol.py"
PIPE_PID_FILE="${TRAIN_OUT_ROOT}/pipeline.pid"
TRAIN_RUN_DIR="${TRAIN_OUT_ROOT}/pinhole_det_head_only"
LOG_FILE="${TRAIN_OUT_ROOT}/queued_retries.log"
RUN_SUFFIX="${RUN_SUFFIX:-$(basename "${TRAIN_OUT_ROOT}")}"


wait_for_pipeline() {
  if [ -f "${PIPE_PID_FILE}" ]; then
    local pid
    pid="$(cat "${PIPE_PID_FILE}")"
    while ps -p "${pid}" > /dev/null 2>&1; do
      sleep 60
    done
  fi
}

pick_ckpt() {
  if [ -f "${TRAIN_RUN_DIR}/selected_ckpt.txt" ]; then
    cat "${TRAIN_RUN_DIR}/selected_ckpt.txt"
    return 0
  fi
  for p in "${TRAIN_RUN_DIR}/checkpoint-final.pth" "${TRAIN_RUN_DIR}/checkpoint-best.pth" "${TRAIN_RUN_DIR}/checkpoint-last.pth"; do
    if [ -f "$p" ]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

run_retry() {
  local name="$1"
  local ckpt="$2"
  local out_dir="$3"
  DET_HEAD_CKPT="${DET_HEAD_CKPT}" \
  bash "${REPO_ROOT}/scripts/eval_det_ckpt_full2170.sh" \
    "${ckpt}" \
    bev_centernet_wide_e2e_v1_predpose_aligngt0 \
    "${out_dir}" \
    "${CUDA_DEV}" >> "${LOG_FILE}" 2>&1

  python "${VALIDATOR}" \
    "${out_dir}/summary_test.json" \
    --expected-contract "${REPO_ROOT}/eval_runs/frames_test2170_nearest_seed42_full.json" \
    --expected-model-task calibrated_sfm \
    --expected-keep-camera-poses 0 \
    --expected-keep-main-agent-poses 1 \
    --require-det \
    --require-rerun-file "${out_dir}/rerun_command.sh" >> "${LOG_FILE}" 2>&1

  python - <<'PY' "${name}" "${out_dir}/summary_test.json" >> "${LOG_FILE}" 2>&1
import json, sys
name=sys.argv[1]
path=sys.argv[2]
d=json.load(open(path))
m=d['metrics']['det_model']
print({'name':name,'single_ap':m['single']['det_ap_iou'],'coop_ap':m['coop']['det_ap_iou']})
PY
}

mkdir -p "${TRAIN_OUT_ROOT}"
printf '%s\n' "[queue] waiting for training pipeline to finish" > "${LOG_FILE}"
wait_for_pipeline
DET_HEAD_CKPT="$(pick_ckpt || true)"
if [ -z "${DET_HEAD_CKPT}" ]; then
  printf '%s\n' "[queue] no trained checkpoint found under ${TRAIN_RUN_DIR}" >> "${LOG_FILE}"
  exit 1
fi
printf '%s\n' "[queue] using DET_HEAD_CKPT=${DET_HEAD_CKPT}" >> "${LOG_FILE}"

run_retry \
  sfix_direct \
  "${REPO_ROOT}/experiments/local_runs/a800_sfix_direct/pinhole_pose_depth_scale/checkpoint-best.pth" \
  "${REPO_ROOT}/eval_runs/det_full2170_predpose_sfix_direct_with_trained_fullhead_${RUN_SUFFIX}"

run_retry \
  mid30_60_detach_pose12 \
  "${REPO_ROOT}/experiments/local_runs/20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/pinhole_pose_depth_scale/checkpoint-best.pth" \
  "${REPO_ROOT}/eval_runs/det_full2170_predpose_mid30_60_with_trained_fullhead_${RUN_SUFFIX}"

run_retry \
  vggt_e30_best \
  "${REPO_ROOT}/experiments/vggt/training/20260224_plan2_coop_metric_e30_full/checkpoint-best.pth" \
  "${REPO_ROOT}/eval_runs/det_full2170_predpose_vggt_e30_best_with_trained_fullhead_${RUN_SUFFIX}"
