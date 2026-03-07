#!/bin/bash

# Create a Codex-friendly debug prompt from a failed run.
#
# Env (optional):
#   CODEX_PROJECT_TAG   group name (default: map-anything)
#   CODEX_TASK_TAG      task name (default: auto)
#   SNAPSHOT_OUT_ROOT   override output root
#   SNAPSHOT_OUT_FILE   write RUN_DIR to this file if set
#   SNAPSHOT_PROMPT_FILE write PROMPT_FILE path to this file if set
#   CODEX_AUTO_DEBUG    set to 1 to auto-run codex exec (default: 1)
#   CODEX_MODEL         override model (default: gpt-5.2-codex)
#   CODEX_WORKDIR       override working dir for codex exec (default: repo root)
#   CODEX_BIN           override codex binary path
#   CODEX_PROXY         override proxy (default: from .codex/config.toml if present)
#   CODEX_REASONING_EFFORT override reasoning effort (default: xhigh)
#   CLASH_DIR           override clash dir (default: /J6P-perception/yijinxiong_workspace/yijinxiong/clash)
#
# Args (optional):
#   --reason <text>
#   --status <code>
#   --log <path>
#   --cmd <command...>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_TAG="${CODEX_PROJECT_TAG:-map-anything}"
TASK_TAG="${CODEX_TASK_TAG:-auto}"

OUT_ROOT="${SNAPSHOT_OUT_ROOT:-${REPO_ROOT}/experiments/long_runs/codex_threads/${PROJECT_TAG}}"
mkdir -p "${OUT_ROOT}"
CODEX_AUTO_DEBUG="${CODEX_AUTO_DEBUG:-1}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.2-codex}"
CODEX_WORKDIR="${CODEX_WORKDIR:-${REPO_ROOT}}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
CLASH_DIR="${CLASH_DIR:-/J6P-perception/yijinxiong_workspace/yijinxiong/clash}"

REASON=""
STATUS=""
LOG_PATH=""
CMD_TEXT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --reason)
      REASON="${2:-}"; shift 2;;
    --status)
      STATUS="${2:-}"; shift 2;;
    --log)
      LOG_PATH="${2:-}"; shift 2;;
    --cmd)
      shift
      CMD_TEXT="$*"
      break;;
    *)
      # ignore unknown args to keep script robust
      shift;;
  esac
done

TS="$(date -u '+%Y%m%d_%H%M%S')"
RUN_ID="${TS}_${TASK_TAG}"
RUN_DIR="${OUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

PROMPT_FILE="${RUN_DIR}/prompt.md"
META_FILE="${RUN_DIR}/meta.txt"

if [ -n "${SNAPSHOT_OUT_FILE:-}" ]; then
  echo "${RUN_DIR}" > "${SNAPSHOT_OUT_FILE}"
fi
if [ -n "${SNAPSHOT_PROMPT_FILE:-}" ]; then
  echo "${PROMPT_FILE}" > "${SNAPSHOT_PROMPT_FILE}"
fi

{
  echo "run_id=${RUN_ID}"
  echo "utc_time=${TS}"
  echo "project=${PROJECT_TAG}"
  echo "task=${TASK_TAG}"
  echo "reason=${REASON}"
  echo "status=${STATUS}"
  echo "log=${LOG_PATH}"
} > "${META_FILE}"

{
  echo "# Codex Debug Prompt"
  echo
  echo "You are Codex. Please debug the failed training run and propose a fix."
  echo
  echo "## Summary"
  echo "- Project: ${PROJECT_TAG}"
  echo "- Task: ${TASK_TAG}"
  echo "- Failure reason: ${REASON}"
  echo "- Exit status: ${STATUS}"
  echo "- Log: ${LOG_PATH}"
  if [ -n "${CODEX_QUEUE_FILE:-}" ]; then
    echo "- QueueFile: ${CODEX_QUEUE_FILE}"
  fi
  echo
  echo "## Last Command"
  if [ -n "${CMD_TEXT}" ]; then
    echo '```bash'
    echo "${CMD_TEXT}"
    echo '```'
  else
    echo "_(command unavailable)_"
  fi
  echo
  echo "## Recent Log Tail (last 200 lines)"
  if [ -n "${LOG_PATH}" ] && [ -f "${LOG_PATH}" ]; then
    echo '```'
    tail -n 200 "${LOG_PATH}" || true
    echo '```'
  else
    echo "_(log missing)_"
  fi
  echo
  echo "## System Snapshot"
  echo '```'
  date -u '+UTC %F %T'
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  df -h 2>/dev/null | head -n 20 || true
  free -h 2>/dev/null || true
  echo "--- ps (train-related) ---"
  ps -eo pid,ppid,cmd --sort=start_time 2>/dev/null | rg -i 'torchrun|torch.distributed.launch|scripts/train.py|map-anything' || true
  echo '```'
  echo
  echo "## Repo State"
  echo '```'
  cd "${REPO_ROOT}"
  git status -sb 2>/dev/null || true
  git diff --stat 2>/dev/null || true
  echo '```'
  echo
  echo "## Ask"
  echo "- Identify root cause."
  echo "- Propose the smallest safe fix."
  echo "- Suggest validation steps."
} > "${PROMPT_FILE}"

# Update index
INDEX_FILE="${OUT_ROOT}/INDEX.md"
if [ ! -f "${INDEX_FILE}" ]; then
  echo "# Codex Debug Threads (${PROJECT_TAG})" > "${INDEX_FILE}"
  echo "" >> "${INDEX_FILE}"
  echo "| run_id | utc_time | task | reason | log | prompt |" >> "${INDEX_FILE}"
  echo "|---|---|---|---|---|---|" >> "${INDEX_FILE}"
fi

LOG_REL="${LOG_PATH}"
PROMPT_REL="${PROMPT_FILE}"

echo "| ${RUN_ID} | ${TS} | ${TASK_TAG} | ${REASON} | ${LOG_REL} | ${PROMPT_REL} |" >> "${INDEX_FILE}"

echo "[codex_snapshot] wrote ${PROMPT_FILE}"

if [ "${CODEX_AUTO_DEBUG}" != "1" ]; then
  exit 0
fi

CODEX_HOME="${CODEX_HOME:-/J6P-perception/yijinxiong_workspace/.codex}"
CODEX_BIN="${CODEX_BIN:-}"
if [ -z "${CODEX_BIN}" ]; then
  if [ -x "/J6P-perception/yijinxiong_workspace/node_modules/.bin/codex" ]; then
    CODEX_BIN="/J6P-perception/yijinxiong_workspace/node_modules/.bin/codex"
  elif command -v codex >/dev/null 2>&1; then
    CODEX_BIN="$(command -v codex)"
  fi
fi

if [ -z "${CODEX_BIN}" ] || [ ! -x "${CODEX_BIN}" ]; then
  echo "[codex_snapshot] codex binary not found; skip auto-debug"
  exit 0
fi

PROXY="${CODEX_PROXY:-}"
if [ -z "${PROXY}" ] && [ -f "${CODEX_HOME}/config.toml" ]; then
  if command -v rg >/dev/null 2>&1; then
    PROXY="$(rg -n '^proxy\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" 2>/dev/null | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/' || true)"
  else
    PROXY="$(grep -E '^proxy\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" 2>/dev/null | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/' || true)"
  fi
fi
BASE_URL=""
if [ -f "${CODEX_HOME}/config.toml" ]; then
  if command -v rg >/dev/null 2>&1; then
    BASE_URL="$(rg -n '^base_url\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" 2>/dev/null | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/' || true)"
  else
    BASE_URL="$(grep -E '^base_url\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" 2>/dev/null | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/' || true)"
  fi
fi

if [ -n "${PROXY}" ] && [ -n "${BASE_URL}" ]; then
  if ! curl -sS -I -x "${PROXY}" --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; then
    if [ -x "${CLASH_DIR}/status.sh" ]; then
      (cd "${CLASH_DIR}" && ./status.sh) >/dev/null 2>&1 || (cd "${CLASH_DIR}" && ./start.sh) >/dev/null 2>&1 || true
    fi
  fi
fi

CODEX_LOG="${RUN_DIR}/codex_exec.log"
CODEX_LAST="${RUN_DIR}/codex_last_message.txt"

nohup env \
  CODEX_HOME="${CODEX_HOME}" \
  http_proxy="${PROXY}" https_proxy="${PROXY}" \
  "${CODEX_BIN}" exec \
  --cd "${CODEX_WORKDIR}" \
  --model "${CODEX_MODEL}" \
  -c "model_reasoning_effort=${CODEX_REASONING_EFFORT}" \
  --dangerously-bypass-approvals-and-sandbox \
  --output-last-message "${CODEX_LAST}" \
  - < "${PROMPT_FILE}" > "${CODEX_LOG}" 2>&1 &

echo "[codex_snapshot] launched codex exec -> ${CODEX_LOG}"
