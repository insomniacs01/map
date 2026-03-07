#!/bin/bash
set -euo pipefail

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
RUN_ONE="${REPO_ROOT}/scripts/run_geom_ablation_test500.sh"

BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth}"
LOSS_NAME="${LOSS_NAME:-opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2_vr1p_noconf}"
EPOCHS="${EPOCHS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_EXTRA_ARGS="${BASE_EXTRA_ARGS:-train_params.print_freq=20}"

if [ ! -f "${BASE_CKPT}" ]; then
  echo "[ERR] BASE_CKPT not found: ${BASE_CKPT}"
  exit 2
fi

if [ ! -x "${RUN_ONE}" ]; then
  echo "[ERR] missing runner script: ${RUN_ONE}"
  exit 3
fi

run_variant() {
  local variant="$1"
  local min_d="$2"
  local max_d="$3"
  local gpus="$4"
  local port="$5"
  local run_tag="${RUN_ID}_plan1_${variant}"
  local extra_args="${BASE_EXTRA_ARGS}"
  # Only filter the *training* split; keep val/test unfiltered so tail buckets don't crash
  # on empty eval splits, and so `checkpoint-best.pth` selection stays comparable.
  extra_args="${extra_args} dataset.coop_min_agent_distance_train=${min_d}"
  extra_args="${extra_args} dataset.coop_max_agent_distance_train=${max_d}"
  extra_args="${extra_args} dataset.coop_min_agent_distance_eval=null"
  extra_args="${extra_args} dataset.coop_max_agent_distance_eval=1000000000"

  (
    export MLP_ENTRY_EXTRA_ARGS="${extra_args}"
    echo "[RUN] variant=${variant} tag=${run_tag} gpus=${gpus} port=${port} min=${min_d} max=${max_d}"
    bash "${RUN_ONE}" "${run_tag}" "${LOSS_NAME}" "${BASE_CKPT}" "${gpus}" "${EPOCHS}" "${port}"
  )
}

run_lane() {
  local lane="$1"
  case "${lane}" in
    near)
      run_variant "near_0_30m" "null" "30" "${GPU_NEAR:-0,1,2,3}" "${PORT_NEAR:-29610}"
      ;;
    mid)
      run_variant "mid_30_60m" "30" "60" "${GPU_MID:-4,5,6,7}" "${PORT_MID:-29620}"
      ;;
    far)
      run_variant "far_60_100m" "60" "100" "${GPU_FAR:-0,1,2,3}" "${PORT_FAR:-29630}"
      ;;
    tail)
      run_variant "tail_100_inf_m" "100" "1000000000" "${GPU_TAIL:-4,5,6,7}" "${PORT_TAIL:-29640}"
      ;;
    *)
      echo "[ERR] unknown lane=${lane}"
      exit 4
      ;;
  esac
}

echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] base_ckpt=${BASE_CKPT}"
echo "[INFO] loss_name=${LOSS_NAME} epochs=${EPOCHS}"
echo "[INFO] phase1 lanes: near + mid (parallel)"
run_lane near &
pid_near=$!
run_lane mid &
pid_mid=$!
wait "${pid_near}"
wait "${pid_mid}"

echo "[INFO] phase2 lanes: far + tail (parallel)"
run_lane far &
pid_far=$!
run_lane tail &
pid_tail=$!
wait "${pid_far}"
wait "${pid_tail}"

echo "[DONE] all variants finished for run_id=${RUN_ID}"
echo "[HINT] quick reports:"
echo "  ${REPO_ROOT}/eval_runs/geom_ablation_test500_${RUN_ID}_plan1_near_0_30m/quick_report.txt"
echo "  ${REPO_ROOT}/eval_runs/geom_ablation_test500_${RUN_ID}_plan1_mid_30_60m/quick_report.txt"
echo "  ${REPO_ROOT}/eval_runs/geom_ablation_test500_${RUN_ID}_plan1_far_60_100m/quick_report.txt"
echo "  ${REPO_ROOT}/eval_runs/geom_ablation_test500_${RUN_ID}_plan1_tail_100_inf_m/quick_report.txt"
