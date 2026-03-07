#!/bin/bash
set -euo pipefail

# Evaluate a geometry checkpoint on the fixed Test500 contract and write a quick report.
#
# Default protocol (2026-02-26+): deployment-like, coop-first.
#   - model_task=calibrated_sfm
#   - keep_camera_poses=0 (do NOT feed cross-agent GT pose)
#   - modes=coop (gate is on coop metrics)
#
# LEGACY NOTE:
#   Older versions of this script defaulted to `model_task=posed_sfm` (and compared
#   against an old posed baseline). That protocol is NOT `protocol_fair` comparable
#   with the current deployment gate. To reproduce legacy runs explicitly, set:
#     MODEL_TASK=posed_sfm EVAL_KEEP_CAMERA_POSES=1 EVAL_MODES="single coop" \
#       BASELINE_SUMMARY="<your posed baseline summary>" \
#       bash map-anything/scripts/eval_geom_ckpt_test500.sh <ckpt> <out_root> <cuda_dev>
#
# Default: fast eval (no point-cloud metrics). Enable PC metrics explicitly:
#   PC_METRICS=1 bash map-anything/scripts/eval_geom_ckpt_test500.sh <ckpt> <out_root> <cuda_dev>

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"

# Fixed-contract defaults (override via env if needed).
CONTRACT_JSON="${CONTRACT_JSON:-${REPO_ROOT}/eval_runs/frames_test500_seed42.json}"
# Optional cached GT-scale contract to speed up `batch_eval.py` on Test500.
GT_SCALE_CONTRACT_JSON_DEFAULT=""
if [ -f "${CONTRACT_JSON}" ] && [ -f "${REPO_ROOT}/eval_runs/frames_test500_seed42.json" ]; then
  if [ "$(realpath "${CONTRACT_JSON}")" = "$(realpath "${REPO_ROOT}/eval_runs/frames_test500_seed42.json")" ]; then
    GT_SCALE_CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/gt_scale_test500_seed42_avg_dis.json"
  fi
fi
GT_SCALE_CONTRACT_JSON="${GT_SCALE_CONTRACT_JSON:-${GT_SCALE_CONTRACT_JSON_DEFAULT}}"

# Baseline used for quick-report comparisons / guardrails (protocol_fair, coop-first).
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${REPO_ROOT}/eval_runs/geom_fairlock_test500_20260226_nm4_base/summary_test.json}"

# Eval knobs (override via env if needed).
PC_METRICS="${PC_METRICS:-0}"
MODEL_TASK="${MODEL_TASK:-calibrated_sfm}"
EVAL_KEEP_CAMERA_POSES="${EVAL_KEEP_CAMERA_POSES:-0}"
EVAL_MODES="${EVAL_MODES:-coop}"
TARGET_SCALE_MULT="${TARGET_SCALE_MULT:-1.3}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <ckpt_path> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
OUT_ROOT="$2"
CUDA_DEV="${3:-0}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERR] missing checkpoint: ${CKPT_PATH}"
  exit 2
fi

mkdir -p "${OUT_ROOT}"

SUMMARY_OUT="${OUT_ROOT}/summary_test.json"
EVAL_LOG="${OUT_ROOT}/eval.log"
REPORT_PATH="${OUT_ROOT}/quick_report.txt"

read -r -a _modes <<< "${EVAL_MODES}"
if [ "${#_modes[@]}" -eq 0 ]; then
  _modes=(coop)
fi

EVAL_ARGS=(
  --split test
  --frames_json "${CONTRACT_JSON}"
  --modes "${_modes[@]}"
  --model_task "${MODEL_TASK}"
  --models "geom_model=${CKPT_PATH}"
  --model_filter geom_model
  --output_root "${OUT_ROOT}"
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

echo "[EVAL] ckpt=${CKPT_PATH}"
echo "[EVAL] out=${OUT_ROOT}"
echo "[EVAL] modes=${EVAL_MODES}"
echo "[EVAL] model_task=${MODEL_TASK} keep_camera_poses=${EVAL_KEEP_CAMERA_POSES}"
echo "[EVAL] contract=${CONTRACT_JSON}"
echo "[EVAL] gt_scale_contract_json=${GT_SCALE_CONTRACT_JSON:-<none>}"
echo "[EVAL] pc_metrics=${PC_METRICS}"

CUDA_VISIBLE_DEVICES="${CUDA_DEV}" \
PYTHONPATH="${REPO_ROOT}" \
"${PYTHON}" "${REPO_ROOT}/scripts/batch_eval.py" \
  "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"

if [ ! -f "${SUMMARY_OUT}" ]; then
  echo "[ERR] missing ${SUMMARY_OUT}"
  exit 3
fi

"${PYTHON}" - "${SUMMARY_OUT}" "${BASELINE_SUMMARY}" "${REPORT_PATH}" <<'PY'
import json
import math
import os
import pathlib
import sys
summary_path = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
report_path = pathlib.Path(sys.argv[3])
cur = json.loads(summary_path.read_text(encoding="utf-8"))
base = json.loads(baseline_path.read_text(encoding="utf-8"))

def m(doc, mode, key):
    return float(doc["metrics"]["geom_model"][mode][key])

def has_mode(doc, mode):
    return (
        "metrics" in doc
        and "geom_model" in doc["metrics"]
        and mode in doc["metrics"]["geom_model"]
        and isinstance(doc["metrics"]["geom_model"][mode], dict)
        and len(doc["metrics"]["geom_model"][mode]) > 0
    )

def mf(doc, mode, key):
    try:
        val = doc["metrics"]["geom_model"][mode].get(key)
    except Exception:
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")

def get_meta(doc):
    meta = doc.get("meta")
    return meta if isinstance(meta, dict) else {}

def meta_get(meta, key):
    return meta.get(key, "<missing>")

lines = []
lines.append(f"summary={summary_path}")
lines.append(f"baseline={baseline_path}")
meta_cur = get_meta(cur)
meta_base = get_meta(base)

lines.append("protocol_meta(summary):")
for k in ("frames_hash_md5", "modes", "model_task", "keep_camera_poses", "gt_scale_contract_hash_md5", "eval_script_md5"):
    lines.append(f"  - {k}={meta_get(meta_cur, k)}")
lines.append("protocol_meta(baseline):")
for k in ("frames_hash_md5", "modes", "model_task", "keep_camera_poses", "gt_scale_contract_hash_md5", "eval_script_md5"):
    lines.append(f"  - {k}={meta_get(meta_base, k)}")

target_mult = os.environ.get("TARGET_SCALE_MULT", "1.3").strip()
try:
    target_mult_f = float(target_mult)
except Exception:
    target_mult_f = 1.3

def protocol_equal(key: str) -> bool:
    # Treat "<missing>" as not comparable.
    a = meta_get(meta_cur, key)
    b = meta_get(meta_base, key)
    if a == "<missing>" or b == "<missing>":
        return False
    return a == b

protocol_ok = True
for k in ("frames_hash_md5", "model_task", "keep_camera_poses", "eval_script_md5", "gt_scale_contract_hash_md5"):
    same = protocol_equal(k)
    protocol_ok = protocol_ok and same
    lines.append(f"protocol_check.{k}={'PASS' if same else 'FAIL'}")

# Decide which modes to report:
# - Prefer summary meta.modes when available (FairLock uses this).
# - Fall back to checking common modes.
raw_modes = meta_cur.get("modes")
if isinstance(raw_modes, list) and raw_modes:
    modes = [m for m in raw_modes if isinstance(m, str)]
else:
    modes = ["coop", "single"]

ok = True
for mode in modes:
    if not (has_mode(cur, mode) and has_mode(base, mode)):
        ok = False
        lines.append(f"{mode}: missing metrics (cur={has_mode(cur, mode)}, base={has_mode(base, mode)}); pass=False")
        continue

    cur_mult = m(cur, mode, "scale_to_gt_mult_err_mean")
    cur_ratio = m(cur, mode, "scale_to_gt_ratio_mean")
    cur_depth = m(cur, mode, "depth_rel_mean")
    cur_cross_t = mf(cur, mode, "cross_agent_pose_trans_mean")
    cur_cross_r = mf(cur, mode, "cross_agent_pose_rot_mean")
    cur_pose_abs = mf(cur, mode, "pose_abs_mean")

    base_mult = m(base, mode, "scale_to_gt_mult_err_mean")
    base_ratio = m(base, mode, "scale_to_gt_ratio_mean")
    base_depth = m(base, mode, "depth_rel_mean")
    base_cross_t = mf(base, mode, "cross_agent_pose_trans_mean")
    base_cross_r = mf(base, mode, "cross_agent_pose_rot_mean")
    base_pose_abs = mf(base, mode, "pose_abs_mean")

    scale_ok = bool(cur_mult <= target_mult_f)
    depth_ok = bool(cur_depth <= base_depth * 1.10)

    # Cross-agent guardrails (fallback to pose_abs_mean when cross_* is absent).
    if math.isfinite(cur_cross_t) and math.isfinite(base_cross_t):
        pose_t_ok = bool(cur_cross_t <= base_cross_t * 1.10)
        pose_t_str = f"cross_t={cur_cross_t:.4f} (<= {base_cross_t*1.10:.4f} => {pose_t_ok})"
    else:
        pose_t_ok = bool(cur_pose_abs <= base_pose_abs * 1.10)
        pose_t_str = f"pose_abs={cur_pose_abs:.4f} (<= {base_pose_abs*1.10:.4f} => {pose_t_ok})"

    if math.isfinite(cur_cross_r) and math.isfinite(base_cross_r):
        pose_r_ok = bool(cur_cross_r <= base_cross_r * 1.10)
        pose_r_str = f"cross_r={cur_cross_r:.4f} (<= {base_cross_r*1.10:.4f} => {pose_r_ok})"
    else:
        pose_r_ok = True
        pose_r_str = "cross_r=<missing>"

    trial_ok = scale_ok and depth_ok and pose_t_ok and pose_r_ok and protocol_ok
    ok = ok and trial_ok
    lines.append(
        f"{mode}: mult={cur_mult:.4f} (<= {target_mult_f:.4f} => {scale_ok}; base={base_mult:.4f}), "
        f"ratio={cur_ratio:.4f} (base={base_ratio:.4f}), "
        f"{pose_t_str}, {pose_r_str}, "
        f"depth_rel={cur_depth:.4f} (<= {base_depth*1.10:.4f} => {depth_ok}), "
        f"pass={trial_ok}"
    )

lines.append(f"protocol_pass={protocol_ok}")
lines.append(f"overall_pass={ok}")
text = "\n".join(lines) + "\n"
report_path.write_text(text, encoding="utf-8")
print(text, end="")
PY

echo "[DONE] report=${REPORT_PATH}"
