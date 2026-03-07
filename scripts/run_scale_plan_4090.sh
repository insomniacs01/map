#!/bin/bash
set -euo pipefail

# Legacy script guard:
# This plan gates on train-log legacy `scale_err_mean` (ratio-to-1), which is not a valid
# primary quality metric on OPV2V when `norm_mode=avg_dis` (GT scale is per-frame g).
# Keep this script only for historical debugging and require an explicit opt-in to run it.
if [ "${MA_ALLOW_LEGACY_SCALE_PLAN_4090:-0}" != "1" ]; then
  echo "[DEPRECATED] ${0##*/} is legacy and disabled by default." >&2
  echo "  - It gates on legacy train-log scale metrics (scale_err_mean), which can mislead OPV2V scale quality." >&2
  echo "  - Use fixed-contract eval (batch_eval.py) + scale_to_gt_* instead: see docs/master_plan.md and map-anything/docs/goal_based_test_plan.md." >&2
  echo "To run anyway (NOT recommended): export MA_ALLOW_LEGACY_SCALE_PLAN_4090=1" >&2
  exit 2
fi

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
ENTRY_SCRIPT="${REPO_ROOT}/bash_scripts/train/mlp_entry_pinhole_pose_depth_scale_recover.sh"

BASE_S5="${REPO_ROOT}/experiments/local_runs/20260210_a800_s5_direct_w0p2/pinhole_pose_depth_scale/checkpoint-best.pth"
BASE_S7A="${REPO_ROOT}/experiments/local_runs/20260210_s7a_full_from_warm_vr1p_retry/pinhole_pose_depth_scale/checkpoint-best.pth"

GPU_LIST="0,1,2,3"
NGPU=4

WORK_TMP_ROOT="/tmp/ma_tmp_4090"
mkdir -p "${WORK_TMP_ROOT}"
export WORK_TMP_ROOT
export TMPDIR="${WORK_TMP_ROOT}"
export TORCHELASTIC_TMPDIR="${WORK_TMP_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MLP_WORKER_GPU="${NGPU}"
export MLP_WORKER_NUM=1
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=127.0.0.1

export MAX_IMGS_PER_GPU=6
export NUM_WORKERS=8
export PRINT_FREQ=50
export FULL_EVAL_FREQ=1
export DINO_PREFETCH=0
export MLP_ENTRY_EXTRA_ARGS="train_params.accum_iter=2 ++train_params.bad_loss_dump_max=5 ++train_params.bad_loss_dump_stride=10"

pick_ckpt() {
  local run_dir="$1"
  if [ -f "${run_dir}/checkpoint-best.pth" ]; then
    echo "${run_dir}/checkpoint-best.pth"
  elif [ -f "${run_dir}/checkpoint-last.pth" ]; then
    echo "${run_dir}/checkpoint-last.pth"
  elif [ -f "${run_dir}/checkpoint-final.pth" ]; then
    echo "${run_dir}/checkpoint-final.pth"
  else
    echo ""
  fi
}

run_stage() {
  local run_tag="$1"
  local loss_name="$2"
  local pretrained="$3"
  local epochs="$4"
  local port="$5"
  local lr="$6"
  local nas_root="${REPO_ROOT}/experiments/long_runs/${run_tag}"
  local local_root="${REPO_ROOT}/experiments/local_runs/${run_tag}"

  mkdir -p "${nas_root}" "${local_root}"
  export MLP_WORKER_0_PORT="${port}"
  export LOSS_NAME="${loss_name}"
  export PRETRAINED_CKPT="${pretrained}"
  export EPOCHS="${epochs}"
  export WARMUP_EPOCHS=0
  export LR="${lr}"
  export NAS_RUN_ROOT="${nas_root}"
  export LOCAL_ROOT="${local_root}"

  echo "[RUN] ${run_tag} loss=${loss_name} ckpt=${pretrained} epochs=${epochs} lr=${lr} port=${port}"
  set +e
  bash "${ENTRY_SCRIPT}"
  local rc=$?
  set -e
  echo "[ENTRY_DONE] ${run_tag} rc=${rc}"
  return "${rc}"
}

parse_gate() {
  local log_file="$1"
  local gate_name="$2"
  python - "${log_file}" "${gate_name}" <<'PY'
import math
import re
import sys

log_file = sys.argv[1]
gate = sys.argv[2]
with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
    lines = [l.strip() for l in fh if "Test Epoch" in l and "scale_err_mean" in l]
if not lines:
    print("NO_TEST_LINE")
    sys.exit(2)
line = lines[-1]

def get_float(name):
    m = re.search(rf"{name}: ([^ ]+)", line)
    if not m:
        return None
    val = m.group(1)
    if val.lower() == "nan":
        return float("nan")
    return float(val)

metrics = {
    "scale_err_mean": get_float("scale_err_mean"),
    "scale_log_err_mean": get_float("scale_log_err_mean"),
    "pose_trans_l2_m": get_float("pose_trans_l2_m"),
    "pose_rot_deg": get_float("pose_rot_deg"),
    "depth_z_mae_m": get_float("depth_z_mae_m"),
    "depth_z_rmse_m": get_float("depth_z_rmse_m"),
    "loss": get_float("loss"),
}
print("METRICS", metrics)
if any(v is None for v in metrics.values()):
    print("MISSING_METRICS")
    sys.exit(3)
if any(math.isnan(v) for v in metrics.values()):
    print("INVALID_NAN")
    sys.exit(4)

if gate == "stage1":
    ok = metrics["scale_err_mean"] <= 7.0 and metrics["scale_log_err_mean"] <= 2.1
    ok = ok and metrics["pose_trans_l2_m"] <= 0.7260
    ok = ok and metrics["pose_rot_deg"] <= 1.5805
    ok = ok and metrics["depth_z_mae_m"] <= 1.8233
    ok = ok and metrics["depth_z_rmse_m"] <= 6.1311
elif gate == "stage2":
    ok = metrics["scale_err_mean"] <= 5.0 and metrics["scale_log_err_mean"] <= 1.8
    ok = ok and metrics["pose_trans_l2_m"] <= 0.7260
    ok = ok and metrics["pose_rot_deg"] <= 1.5805
    ok = ok and metrics["depth_z_mae_m"] <= 1.8233
    ok = ok and metrics["depth_z_rmse_m"] <= 6.1311
elif gate == "stage3":
    ok = metrics["scale_err_mean"] <= 3.0 and metrics["scale_log_err_mean"] <= 1.1
    ok = ok and metrics["pose_trans_l2_m"] <= 0.50
    ok = ok and metrics["pose_rot_deg"] <= 1.0
    ok = ok and metrics["depth_z_mae_m"] <= 1.5
    ok = ok and metrics["depth_z_rmse_m"] <= 5.0
else:
    ok = False

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
}

check_invalid() {
  local log_file="$1"
  if rg -n "Bad loss=nan" "${log_file}" >/dev/null 2>&1; then
    echo "INVALID_BADLOSS"
    return 0
  fi
  if rg -n "loss: nan" "${log_file}" >/dev/null 2>&1; then
    echo "INVALID_NAN"
    return 0
  fi
  return 1
}

run_with_eval() {
  local run_tag="$1"
  local loss_name="$2"
  local pretrained="$3"
  local epochs="$4"
  local port="$5"
  local lr="$6"
  local gate="$7"

  run_stage "${run_tag}" "${loss_name}" "${pretrained}" "${epochs}" "${port}" "${lr}"
  local run_dir="${REPO_ROOT}/experiments/local_runs/${run_tag}/pinhole_pose_depth_scale"
  local log_file="${run_dir}/train.log"
  if check_invalid "${log_file}"; then
    echo "[WARN] ${run_tag} invalid due to NaN/badloss"
    return 2
  fi
  parse_gate "${log_file}" "${gate}"
}

# Stage 0.2 + Stage 1 warmup (scale-only, no gating, noconf)
run_tag_warm="20260211_s1_warm_scaleonly_noconf_4090"
run_with_eval "${run_tag_warm}" "opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2_noconf" "${BASE_S5}" 1 29580 1e-5 stage1 || warm_status=$?
warm_status=${warm_status:-0}

warm_dir="${REPO_ROOT}/experiments/local_runs/${run_tag_warm}/pinhole_pose_depth_scale"
if ! rg -n "scale_valid_ratio_avg" "${warm_dir}/train.log" >/dev/null 2>&1; then
  echo "WARN: scale_valid_ratio_avg missing in ${warm_dir}/train.log; continue without this check"
fi

warm_ckpt=$(pick_ckpt "${warm_dir}")
if [ -z "${warm_ckpt}" ]; then
  echo "ERROR: warmup checkpoint missing"
  exit 11
fi

# Stage 1 full (no gating, noconf)
run_tag_s1="20260211_s1_full_w0p2_noconf_4090"
run_with_eval "${run_tag_s1}" "opv2v_vggt_pose_scale_loss_recover_direct_w0p2_noconf" "${warm_ckpt}" 2 29581 1e-5 stage1 || s1_status=$?
s1_status=${s1_status:-0}

s1_dir="${REPO_ROOT}/experiments/local_runs/${run_tag_s1}/pinhole_pose_depth_scale"
if check_invalid "${s1_dir}/train.log"; then
  echo "[WARN] Stage1 invalid; proceed to Stage2 with gating and noconf"
fi

# Stage 2 (gating, noconf)
s1_ckpt=$(pick_ckpt "${s1_dir}")
if [ -z "${s1_ckpt}" ]; then
  s1_ckpt="${warm_ckpt}"
fi
run_tag_s2="20260211_s2_full_w0p2_vr1p_noconf_4090"
run_with_eval "${run_tag_s2}" "opv2v_vggt_pose_scale_loss_recover_direct_w0p2_vr1p_noconf" "${s1_ckpt}" 2 29582 1e-5 stage2 || s2_status=$?
s2_status=${s2_status:-0}

# Stage 3 (higher scale weight, gating, noconf)
s2_dir="${REPO_ROOT}/experiments/local_runs/${run_tag_s2}/pinhole_pose_depth_scale"
s2_ckpt=$(pick_ckpt "${s2_dir}")
if [ -z "${s2_ckpt}" ]; then
  s2_ckpt="${s1_ckpt}"
fi
run_tag_s3="20260211_s3_full_w1p0_vr1p_noconf_4090"
run_with_eval "${run_tag_s3}" "opv2v_vggt_pose_scale_loss_recover_direct_w1p0_vr1p_noconf" "${s2_ckpt}" 2 29583 1e-5 stage3 || s3_status=$?
s3_status=${s3_status:-0}

# Exit code: 0 only when Stage3 passes; otherwise non-zero so loop can retry/advance.
if [ "${s3_status}" -eq 0 ]; then
  exit 0
fi
exit 2
