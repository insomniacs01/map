#!/bin/bash
set -euo pipefail

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
ENTRY_SCRIPT="${REPO_ROOT}/bash_scripts/train/mlp_entry_pinhole_pose_depth_scale_recover.sh"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"
CONTRACT_JSON="${CONTRACT_JSON:-${REPO_ROOT}/eval_runs/frames_test500_seed42.json}"
# Optional cached GT scale contract to speed up `batch_eval.py` on large frame sets.
# For the canonical Test500 contract, this is precomputed in-repo.
GT_SCALE_CONTRACT_JSON_DEFAULT=""
if [ -f "${CONTRACT_JSON}" ] && [ -f "${REPO_ROOT}/eval_runs/frames_test500_seed42.json" ]; then
  if [ "$(realpath "${CONTRACT_JSON}")" = "$(realpath "${REPO_ROOT}/eval_runs/frames_test500_seed42.json")" ]; then
    GT_SCALE_CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/gt_scale_test500_seed42_avg_dis.json"
  fi
fi
GT_SCALE_CONTRACT_JSON="${GT_SCALE_CONTRACT_JSON:-${GT_SCALE_CONTRACT_JSON_DEFAULT}}"
# Baseline summary used for quick-report comparisons / guardrails.
# Default is the current promoted coop-first checkpoint evaluated under the deployment protocol
# (`model_task=calibrated_sfm`, `keep_camera_poses=0`) on the fixed Test500 contract.
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${REPO_ROOT}/eval_runs/geom_ablation_test500_20260226_srecover_near_from_midpose12_lr1e5/summary_test.json}"
# NOTE: This script is for fast ablations + fixed-contract eval. Point-cloud metrics (Chamfer/BEV IoU)
# are expensive on 500 frames; keep them optional so "scale/psoe/depth" iterations stay fast.
PC_METRICS="${PC_METRICS:-0}"
# Eval knobs:
# - EVAL_MODES: space-separated list passed to `--modes` (default: "single coop")
# - EVAL_KEEP_CAMERA_POSES: if 1, evaluate in posed protocol (`--keep_camera_poses`)
# - COOP_FIRST: if 1, pass/fail is based on coop metrics only (pose is not used as a guardrail)
EVAL_MODES="${EVAL_MODES:-coop}"
EVAL_KEEP_CAMERA_POSES="${EVAL_KEEP_CAMERA_POSES:-0}"
COOP_FIRST="${COOP_FIRST:-1}"
MODEL_TASK="${MODEL_TASK:-calibrated_sfm}"

if [ ! -x "${PYTHON}" ]; then
  PYTHON="python"
fi

if [ $# -lt 3 ]; then
  echo "Usage: $0 <run_tag> <loss_name> <pretrained_ckpt> [gpus=0,1,2,3] [epochs=1] [port=29510]"
  exit 1
fi

RUN_TAG="$1"
LOSS_NAME="$2"
PRETRAINED_CKPT="$3"
GPU_LIST="${4:-0,1,2,3}"
EPOCHS="${5:-1}"
PORT="${6:-29510}"

# Normalize checkpoint path so training remains robust when the entry script
# changes working directory (workspace root vs repo root).
if [ ! -f "${PRETRAINED_CKPT}" ]; then
  if [ -f "${REPO_ROOT}/${PRETRAINED_CKPT}" ]; then
    PRETRAINED_CKPT="${REPO_ROOT}/${PRETRAINED_CKPT}"
  fi
fi
if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "[ERR] pretrained checkpoint not found: ${PRETRAINED_CKPT}"
  exit 4
fi
PRETRAINED_CKPT="$(realpath "${PRETRAINED_CKPT}")"

LOCAL_ROOT="${REPO_ROOT}/experiments/local_runs/${RUN_TAG}"
NAS_RUN_ROOT="${REPO_ROOT}/experiments/long_runs/${RUN_TAG}"
RUN_DIR="${LOCAL_ROOT}/pinhole_pose_depth_scale"
EVAL_OUT_ROOT="${REPO_ROOT}/eval_runs/geom_ablation_test500_${RUN_TAG}"
SUMMARY_OUT="${EVAL_OUT_ROOT}/summary_test.json"
EVAL_LOG="${EVAL_OUT_ROOT}/eval.log"

mkdir -p "${LOCAL_ROOT}" "${NAS_RUN_ROOT}" "${EVAL_OUT_ROOT}"

NGPU="$(awk -F, '{print NF}' <<< "${GPU_LIST}")"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MLP_WORKER_GPU="${NGPU}"
export MLP_WORKER_NUM=1
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=127.0.0.1
export MLP_WORKER_0_PORT="${PORT}"

export LOSS_NAME
export PRETRAINED_CKPT
export EPOCHS
export WARMUP_EPOCHS=0
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-2e-6}"
# For ablation fine-tuning from a pretrained ckpt, avoid inheriting stale
# optimizer/scheduler states unless explicitly requested by caller.
export RESUME="${RESUME:-false}"
# Fail fast on sustained NaN/bad-loss storms so we do not spend a full epoch
# producing unusable checkpoints.
export MAX_BAD_LOSS_COUNT="${MAX_BAD_LOSS_COUNT:-50}"
export NAS_RUN_ROOT
export LOCAL_ROOT
export MAX_IMGS_PER_GPU="${MAX_IMGS_PER_GPU:-6}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PRINT_FREQ="${PRINT_FREQ:-50}"
export FULL_EVAL_FREQ="${FULL_EVAL_FREQ:-1}"
export DINO_PREFETCH="${DINO_PREFETCH:-0}"
export MLP_ENTRY_EXTRA_ARGS="${MLP_ENTRY_EXTRA_ARGS:-train_params.accum_iter=2 ++train_params.bad_loss_dump_max=5 ++train_params.bad_loss_dump_stride=10}"
# Default acceptance threshold for coop-first geometry readiness.
export TARGET_SCALE_MULT="${TARGET_SCALE_MULT:-1.3}"

echo "[RUN] tag=${RUN_TAG} loss=${LOSS_NAME} ckpt=${PRETRAINED_CKPT} gpus=${GPU_LIST} epochs=${EPOCHS} port=${PORT} pc_metrics=${PC_METRICS}"
# Training sometimes exits non-zero due to "bad loss storms" (e.g. persistent NaNs).
# If a usable checkpoint exists, we still want to run the fixed-contract eval so we
# can compare runs fairly and avoid wasting compute.
set +e
bash "${ENTRY_SCRIPT}"
TRAIN_RC=$?
set -e
echo "[TRAIN] rc=${TRAIN_RC}"

pick_ckpt() {
  local d="$1"
  for name in checkpoint-best.pth checkpoint-last.pth checkpoint-final.pth; do
    if [ -f "${d}/${name}" ]; then
      echo "${d}/${name}"
      return 0
    fi
  done
  return 1
}

CKPT_PATH="$(pick_ckpt "${RUN_DIR}")"
if [ -z "${CKPT_PATH}" ]; then
  echo "[ERR] no checkpoint in ${RUN_DIR}"
  exit 2
fi
echo "[EVAL] ckpt=${CKPT_PATH}"

read -r -a _modes <<< "${EVAL_MODES}"
if [ "${#_modes[@]}" -eq 0 ]; then
  _modes=(single coop)
fi

EVAL_ARGS=(
  --split test
  --frames_json "${CONTRACT_JSON}"
  --modes "${_modes[@]}"
  --model_task "${MODEL_TASK}"
  --models "geom_model=${CKPT_PATH}"
  --model_filter geom_model
  --output_root "${EVAL_OUT_ROOT}"
)
if [ -n "${GT_SCALE_CONTRACT_JSON}" ] && [ -f "${GT_SCALE_CONTRACT_JSON}" ]; then
  EVAL_ARGS+=( --gt_scale_contract_json "${GT_SCALE_CONTRACT_JSON}" )
fi
if [ "${EVAL_KEEP_CAMERA_POSES}" = "1" ]; then
  EVAL_ARGS+=( --keep_camera_poses )
fi
if [ "${PC_METRICS}" = "1" ]; then
  EVAL_ARGS+=( --pc_metrics )
fi

PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/batch_eval.py" \
  "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"

if [ ! -f "${SUMMARY_OUT}" ]; then
  echo "[ERR] missing ${SUMMARY_OUT}"
  exit 3
fi

REPORT_PATH="${EVAL_OUT_ROOT}/quick_report.txt"
python - "${SUMMARY_OUT}" "${BASELINE_SUMMARY}" "${REPORT_PATH}" "${TRAIN_RC}" "${EVAL_MODES}" "${COOP_FIRST}" "${EVAL_KEEP_CAMERA_POSES}" <<'PY'
import json, math, os, pathlib, sys
summary_path = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
report_path = pathlib.Path(sys.argv[3])
train_rc = int(sys.argv[4])
eval_modes = sys.argv[5].split()
coop_first = bool(int(sys.argv[6]))
keep_camera_poses = bool(int(sys.argv[7]))
cur = json.loads(summary_path.read_text())
base = json.loads(baseline_path.read_text())

def m(doc, mode, key):
    return float(doc["metrics"]["geom_model"][mode][key])

def mf(doc, mode, key):
    try:
        val = doc["metrics"]["geom_model"][mode].get(key)
    except Exception:
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")

def has_mode(doc, mode):
    return (
        "metrics" in doc
        and "geom_model" in doc["metrics"]
        and mode in doc["metrics"]["geom_model"]
        and isinstance(doc["metrics"]["geom_model"][mode], dict)
        and len(doc["metrics"]["geom_model"][mode]) > 0
    )

lines = []
lines.append(f"summary={summary_path}")
lines.append(f"baseline={baseline_path}")
lines.append(f"train_rc={train_rc}")
lines.append(f"eval_modes={' '.join(eval_modes) if eval_modes else 'single coop'}")
lines.append(f"eval_keep_camera_poses={keep_camera_poses}")
lines.append(f"coop_first={coop_first}")
ok = True
modes_to_check = ["coop"] if coop_first else (eval_modes or ["single", "coop"])
for mode in modes_to_check:
    if not has_mode(cur, mode):
        ok = False
        lines.append(f"{mode}: missing metrics (all frames failed or filtered); pass=False")
        continue

    cur_mult = m(cur, mode, "scale_to_gt_mult_err_mean")
    cur_ratio = m(cur, mode, "scale_to_gt_ratio_mean")
    cur_depth = m(cur, mode, "depth_rel_mean")

    base_mult = m(base, mode, "scale_to_gt_mult_err_mean")
    base_depth = m(base, mode, "depth_rel_mean")

    improve = (base_mult - cur_mult) / max(1e-8, base_mult)

    # Scale criterion: prefer absolute threshold when provided.
    target_mult = os.environ.get("TARGET_SCALE_MULT", "").strip()
    if target_mult:
        try:
            target_mult_f = float(target_mult)
        except ValueError:
            target_mult_f = float("nan")
    else:
        target_mult_f = float("nan")
    if math.isfinite(target_mult_f):
        scale_ok = bool(cur_mult <= target_mult_f)
        scale_details = f"scale_ok={scale_ok} (<= {target_mult_f:.4f})"
    else:
        scale_ok = bool(improve >= 0.10)
        scale_details = f"scale_ok={scale_ok} (improve>={0.10*100:.1f}%)"

    # Guardrails:
    # - Coop-first deployments care about cross-agent relative pose quality, not single-agent pose.
    # - If cross-agent metrics are missing (legacy summaries), fall back to pose_abs_mean.
    pose_ok = True
    pose_details = "pose_guard=ignored"
    if mode == "coop" and not keep_camera_poses:
        cur_cross_t = mf(cur, mode, "cross_agent_pose_trans_mean")
        cur_cross_r = mf(cur, mode, "cross_agent_pose_rot_mean")
        base_cross_t = mf(base, mode, "cross_agent_pose_trans_mean")
        base_cross_r = mf(base, mode, "cross_agent_pose_rot_mean")
        if math.isfinite(cur_cross_t) and math.isfinite(base_cross_t):
            thr_t = base_cross_t * 1.10
            ok_t = cur_cross_t <= thr_t
            pose_details = f"cross_trans={cur_cross_t:.4f} (<= {thr_t:.4f} => {ok_t})"
            ok_r = True
            if math.isfinite(cur_cross_r) and math.isfinite(base_cross_r):
                thr_r = base_cross_r * 1.10
                ok_r = cur_cross_r <= thr_r
                pose_details += f", cross_rot={cur_cross_r:.3f} (<= {thr_r:.3f} => {ok_r})"
            pose_ok = bool(ok_t and ok_r)
        else:
            cur_pose = m(cur, mode, "pose_abs_mean")
            base_pose = m(base, mode, "pose_abs_mean")
            thr_pose = base_pose * 1.10
            pose_ok = bool(cur_pose <= thr_pose)
            pose_details = f"pose_abs={cur_pose:.4f} (<= {thr_pose:.4f} => {pose_ok})"
    elif not keep_camera_poses:
        # Single-mode: keep old guardrail unless caller explicitly sets coop_first or posed eval.
        cur_pose = m(cur, mode, "pose_abs_mean")
        base_pose = m(base, mode, "pose_abs_mean")
        if coop_first:
            pose_ok = True
            pose_details = f"pose_abs={cur_pose:.4f} (coop_first => ignored)"
        else:
            thr_pose = base_pose * 1.10
            pose_ok = bool(cur_pose <= thr_pose)
            pose_details = f"pose_abs={cur_pose:.4f} (<= {thr_pose:.4f} => {pose_ok})"
    elif keep_camera_poses:
        pose_details = "pose_guard=ignored (keep_camera_poses)"

    depth_thr = base_depth * 1.10
    depth_ok = bool(cur_depth <= depth_thr)

    trial_ok = scale_ok and pose_ok and depth_ok
    ok = ok and trial_ok
    lines.append(
        f"{mode}: mult={cur_mult:.4f} (base={base_mult:.4f}, improve={improve*100:.1f}%), "
        f"ratio={cur_ratio:.4f}, {scale_details}, {pose_details}, "
        f"depth={cur_depth:.4f} (<= {depth_thr:.4f} => {depth_ok}), pass={trial_ok}"
    )

lines.append(f"overall_pass={ok}")
text = "\n".join(lines) + "\n"
report_path.write_text(text, encoding="utf-8")
print(text, end="")
PY

echo "[DONE] report=${REPORT_PATH}"
