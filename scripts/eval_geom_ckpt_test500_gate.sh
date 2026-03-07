#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <geom_ckpt> <output_root> [model_name]" >&2
  exit 2
fi

CKPT="$1"
OUT_ROOT="$2"
MODEL_NAME="${3:-geom_model}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="python"
fi

FRAMES_JSON="${GEOM_TEST500_FRAMES_JSON:-${REPO_ROOT}/eval_runs/frames_test500_seed42.json}"
BASELINE_SUMMARY="${GEOM_GATE_BASELINE:-${REPO_ROOT}/eval_runs/geom_fairlock_test500_20260226_promoted_v2/summary_test.json}"
STRICT_GATE="${GEOM_GATE_STRICT:-0}"
export MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION="${MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION:-1}"

if [ ! -f "${CKPT}" ]; then
  echo "[geom-eval] checkpoint not found: ${CKPT}" >&2
  exit 3
fi
if [ ! -f "${FRAMES_JSON}" ]; then
  echo "[geom-eval] frames_json not found: ${FRAMES_JSON}" >&2
  exit 4
fi
if [ ! -f "${BASELINE_SUMMARY}" ]; then
  echo "[geom-eval] baseline summary not found: ${BASELINE_SUMMARY}" >&2
  exit 5
fi

if [[ "${OUT_ROOT}" != /* ]]; then
  OUT_ROOT="${REPO_ROOT}/${OUT_ROOT}"
fi
mkdir -p "${OUT_ROOT}"
LOG_FILE="${OUT_ROOT}/run.log"
RERUN_FILE="${OUT_ROOT}/rerun_command.sh"
REPORT_JSON="${OUT_ROOT}/gate_geom_readiness.json"
REPORT_TXT="${OUT_ROOT}/gate_geom_readiness.txt"

cat > "${RERUN_FILE}" <<EOF
#!/bin/bash
set -euo pipefail
PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/batch_eval.py" \
  --split test \
  --frames_json "${FRAMES_JSON}" \
  --modes coop \
  --model_task calibrated_sfm \
  --models "${MODEL_NAME}=${CKPT}" \
  --model_filter "${MODEL_NAME}" \
  --output_root "${OUT_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/bash_scripts/monitor/gate_geom_readiness.py" \
  --summary "${OUT_ROOT}/summary_test.json" \
  --baseline "${BASELINE_SUMMARY}" \
  --model-name "${MODEL_NAME}" \
  --report-json "${REPORT_JSON}"
EOF
chmod +x "${RERUN_FILE}"

{
  echo "[geom-eval] ckpt=${CKPT}"
  echo "[geom-eval] out_root=${OUT_ROOT}"
  echo "[geom-eval] frames_json=${FRAMES_JSON}"
  echo "[geom-eval] baseline=${BASELINE_SUMMARY}"
  PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/batch_eval.py" \
    --split test \
    --frames_json "${FRAMES_JSON}" \
    --modes coop \
    --model_task calibrated_sfm \
    --models "${MODEL_NAME}=${CKPT}" \
    --model_filter "${MODEL_NAME}" \
    --output_root "${OUT_ROOT}"
} 2>&1 | tee "${LOG_FILE}"

SUMMARY_PATH="${OUT_ROOT}/summary_test.json"
if [ ! -f "${SUMMARY_PATH}" ]; then
  echo "[geom-eval] missing summary: ${SUMMARY_PATH}" >&2
  exit 6
fi

set +e
PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/bash_scripts/monitor/gate_geom_readiness.py" \
  --summary "${SUMMARY_PATH}" \
  --baseline "${BASELINE_SUMMARY}" \
  --model-name "${MODEL_NAME}" \
  --report-json "${REPORT_JSON}" 2>&1 | tee "${REPORT_TXT}"
GATE_STATUS=${PIPESTATUS[0]}
set -e

echo "[geom-eval] gate_status=${GATE_STATUS} strict=${STRICT_GATE}" | tee -a "${REPORT_TXT}"
if [ "${STRICT_GATE}" = "1" ]; then
  exit "${GATE_STATUS}"
fi
exit 0
