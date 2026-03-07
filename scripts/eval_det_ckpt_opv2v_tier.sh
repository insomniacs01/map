#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 <ckpt_path> <det_head_cfg> <tier: quick|gate|nearfull|final> <out_root> [cuda_device=0]"
  exit 1
fi

CKPT_PATH="$1"
DET_HEAD_CFG="$2"
TIER="$3"
OUT_ROOT="$4"
CUDA_DEV="${5:-0}"

REPO_ROOT="/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything"
PYTHON="/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python"
WRAPPER="${REPO_ROOT}/scripts/eval_det_ckpt_contract_with_buckets.sh"

mkdir -p "${OUT_ROOT}"

STRESS="${STRESS:-0}"
MODEL_ARCH="${MODEL_ARCH:-mapanything}"
EVAL_MODES="${EVAL_MODES:-single coop}"
KEEP_CAMERA_POSES="${KEEP_CAMERA_POSES:-0}"
KEEP_MAIN_AGENT_POSES="${KEEP_MAIN_AGENT_POSES:-1}"
MODEL_TASK="${MODEL_TASK:-calibrated_sfm}"
DET_HEAD_CKPT="${DET_HEAD_CKPT:-}"

case "${TIER}" in
  quick)
    CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/frames_test50_det_e2e_v5.json"
    ;;
  gate)
    if [ "${STRESS}" = "1" ]; then
      CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/frames_test500_nearest_stress_seed42.json"
    else
      CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/frames_test500_nearest_seed42.json"
    fi
    ;;
  nearfull)
    if [ "${STRESS}" = "1" ]; then
      CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/frames_test2000_nearest_stress_seed42.json"
    else
      CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/frames_test2000_nearest_seed42.json"
    fi
    ;;
  final)
    CONTRACT_JSON_DEFAULT="${REPO_ROOT}/eval_runs/frames_test2170_nearest_seed42_full.json"
    ;;
  *)
    echo "[ERR] unknown tier: ${TIER}"
    exit 2
    ;;
esac

CONTRACT_JSON="${CONTRACT_JSON:-${CONTRACT_JSON_DEFAULT}}"
if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERR] missing checkpoint: ${CKPT_PATH}"
  exit 3
fi
if [ ! -f "${CONTRACT_JSON}" ]; then
  echo "[ERR] missing contract: ${CONTRACT_JSON}"
  exit 4
fi

CONTRACT_HASH_V2=$("${PYTHON}" - <<'PY' "${CONTRACT_JSON}"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    doc = json.load(fh)
print(doc.get('frames_hash_md5_v2', ''))
PY
)

cat > "${OUT_ROOT}/rerun_command.sh" <<EOF2
#!/bin/bash
set -euo pipefail
cd /J6P-perception/yijinxiong_workspace/vggt_series_4_coop
KEEP_CAMERA_POSES='${KEEP_CAMERA_POSES}' \\
KEEP_MAIN_AGENT_POSES='${KEEP_MAIN_AGENT_POSES}' \\
MODEL_TASK='${MODEL_TASK}' \\
MODEL_ARCH='${MODEL_ARCH}' \\
EVAL_MODES='${EVAL_MODES}' \\
STRESS='${STRESS}' \\
CONTRACT_JSON='${CONTRACT_JSON}' \\
DET_HEAD_CKPT='${DET_HEAD_CKPT}' \\
bash map-anything/scripts/eval_det_ckpt_opv2v_tier.sh \\
  '${CKPT_PATH}' \\
  '${DET_HEAD_CFG}' \\
  '${TIER}' \\
  '${OUT_ROOT}' \\
  '${CUDA_DEV}'
EOF2
chmod +x "${OUT_ROOT}/rerun_command.sh"

cat > "${OUT_ROOT}/protocol_manifest.json" <<EOF2
{
  "tier": "${TIER}",
  "stress": ${STRESS},
  "contract_json": "${CONTRACT_JSON}",
  "contract_hash_md5_v2": "${CONTRACT_HASH_V2}",
  "model_task": "${MODEL_TASK}",
  "keep_camera_poses": ${KEEP_CAMERA_POSES},
  "keep_main_agent_poses": ${KEEP_MAIN_AGENT_POSES},
  "model_arch": "${MODEL_ARCH}",
  "eval_modes": "${EVAL_MODES}",
  "det_head_cfg": "${DET_HEAD_CFG}",
  "det_head_ckpt": "${DET_HEAD_CKPT}",
  "ckpt_path": "${CKPT_PATH}",
  "wrapper": "map-anything/scripts/eval_det_ckpt_opv2v_tier.sh"
}
EOF2

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] wrote ${OUT_ROOT}/rerun_command.sh"
  echo "[DRY-RUN] wrote ${OUT_ROOT}/protocol_manifest.json"
  exit 0
fi

export KEEP_CAMERA_POSES
export KEEP_MAIN_AGENT_POSES
export MODEL_TASK
export MODEL_ARCH
export EVAL_MODES
export DET_HEAD_CKPT
export CONTRACT_JSON

bash "${WRAPPER}" "${CKPT_PATH}" "${DET_HEAD_CFG}" "${CONTRACT_JSON}" "${OUT_ROOT}" "${CUDA_DEV}"
