#!/bin/bash

# Auto decision agent: optionally rewrite the queue based on the latest summary.json.
#
# Usage:
#   bash auto_queue_decider.sh --summary <summary.json> --queue <queue_file> --state <state_file>
#
# Behavior:
#   - Calls Codex to propose a new queue file content or "NO_CHANGE".
#   - Validates that already-completed commands (prefix) are unchanged.
#   - Writes a backup and replaces the queue if valid.
#
# Env (optional):
#   CODEX_MODEL               (default: gpt-5.2-codex)
#   CODEX_REASONING_EFFORT    (default: xhigh)
#   CODEX_HOME                (default: /J6P-perception/yijinxiong_workspace/.codex)
#   CODEX_BIN                 (default: auto-discover)
#   CODEX_PROXY               (default: from config.toml)
#   DECIDE_LOG                (default: <queue_dir>/auto_queue_decider.log)
#   DECIDE_MAX_LINES          (default: 2000)
#   CLASH_DIR                 (default: /J6P-perception/yijinxiong_workspace/yijinxiong/clash)

set -euo pipefail

SUMMARY=""
QUEUE=""
STATE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --summary) SUMMARY="${2:-}"; shift 2;;
    --queue) QUEUE="${2:-}"; shift 2;;
    --state) STATE="${2:-}"; shift 2;;
    *) shift;;
  esac
done

if [ -z "${SUMMARY}" ] || [ -z "${QUEUE}" ] || [ -z "${STATE}" ]; then
  echo "[decider] missing required args" >&2
  exit 2
fi

if [ ! -f "${SUMMARY}" ] || [ ! -f "${QUEUE}" ]; then
  echo "[decider] summary or queue missing" >&2
  exit 3
fi

IDX=0
if [ -f "${STATE}" ]; then
  IDX="$(cat "${STATE}" 2>/dev/null || echo 0)"
fi

CODEX_MODEL="${CODEX_MODEL:-gpt-5.2-codex}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
CODEX_HOME="${CODEX_HOME:-/J6P-perception/yijinxiong_workspace/.codex}"
CLASH_DIR="${CLASH_DIR:-/J6P-perception/yijinxiong_workspace/yijinxiong/clash}"

CODEX_BIN="${CODEX_BIN:-}"
if [ -z "${CODEX_BIN}" ]; then
  if [ -x "/J6P-perception/yijinxiong_workspace/node_modules/.bin/codex" ]; then
    CODEX_BIN="/J6P-perception/yijinxiong_workspace/node_modules/.bin/codex"
  elif command -v codex >/dev/null 2>&1; then
    CODEX_BIN="$(command -v codex)"
  fi
fi
if [ -z "${CODEX_BIN}" ] || [ ! -x "${CODEX_BIN}" ]; then
  echo "[decider] codex binary not found; skip"
  exit 0
fi

PROXY="${CODEX_PROXY:-}"
if [ -z "${PROXY}" ] && [ -f "${CODEX_HOME}/config.toml" ]; then
  if command -v rg >/dev/null 2>&1; then
    PROXY="$(rg -n '^proxy\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/')"
  else
    PROXY="$(grep -E '^proxy\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/')"
  fi
fi
BASE_URL=""
if [ -f "${CODEX_HOME}/config.toml" ]; then
  if command -v rg >/dev/null 2>&1; then
    BASE_URL="$(rg -n '^base_url\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/')"
  else
    BASE_URL="$(grep -E '^base_url\\s*=\\s*\"' \"${CODEX_HOME}/config.toml\" | head -n 1 | sed -E 's/.*\"(.*)\".*/\\1/')"
  fi
fi
if [ -n "${PROXY}" ] && [ -n "${BASE_URL}" ]; then
  if ! curl -sS -I -x "${PROXY}" --max-time 5 "${BASE_URL}/models" >/dev/null 2>&1; then
    if [ -x "${CLASH_DIR}/status.sh" ]; then
      (cd "${CLASH_DIR}" && ./status.sh) >/dev/null 2>&1 || (cd "${CLASH_DIR}" && ./start.sh) >/dev/null 2>&1 || true
    fi
  fi
fi

QUEUE_DIR="$(dirname "${QUEUE}")"
DECIDE_LOG="${DECIDE_LOG:-${QUEUE_DIR}/auto_queue_decider.log}"
DECIDE_MAX_LINES="${DECIDE_MAX_LINES:-2000}"

prompt_file="$(mktemp /tmp/queue_decider_prompt_XXXX.md)"
out_file="$(mktemp /tmp/queue_decider_out_XXXX.txt)"

orig_lines="$(grep -v '^[[:space:]]*#' "${QUEUE}" | sed '/^[[:space:]]*$/d')"
prefix_lines="$(echo "${orig_lines}" | head -n "${IDX}")"

{
  echo "# Auto Queue Decision"
  echo
  echo "Summary JSON:"
  echo '```json'
  cat "${SUMMARY}"
  echo '```'
  echo
  echo "Current queue file (non-empty, non-comment lines):"
  echo '```'
  echo "${orig_lines}"
  echo '```'
  echo
  echo "Already completed lines (must remain unchanged, first ${IDX} lines):"
  echo '```'
  echo "${prefix_lines}"
  echo '```'
  echo
  echo "Task:"
  echo "- Output the FULL updated queue file content (one command per line)."
  echo "- Keep the first ${IDX} lines unchanged."
  echo "- You may reorder or replace future lines or append new lines."
  echo "- If no change is needed, output exactly: NO_CHANGE"
} > "${prompt_file}"

env CODEX_HOME="${CODEX_HOME}" http_proxy="${PROXY}" https_proxy="${PROXY}" \
  "${CODEX_BIN}" exec \
  --cd "/J6P-perception/yijinxiong_workspace/vggt_series_4_coop" \
  --model "${CODEX_MODEL}" \
  -c "model_reasoning_effort=${CODEX_REASONING_EFFORT}" \
  --dangerously-bypass-approvals-and-sandbox \
  - < "${prompt_file}" > "${out_file}" 2>> "${DECIDE_LOG}" || true

if [ ! -s "${out_file}" ]; then
  echo "[decider] empty output" >> "${DECIDE_LOG}"
  exit 0
fi

if grep -q "^NO_CHANGE$" "${out_file}"; then
  echo "[decider] no change" >> "${DECIDE_LOG}"
  exit 0
fi

new_lines="$(grep -v '^[[:space:]]*#' "${out_file}" | sed '/^[[:space:]]*$/d' | head -n "${DECIDE_MAX_LINES}")"
new_prefix="$(echo "${new_lines}" | head -n "${IDX}")"

if [ "${IDX}" -gt 0 ] && [ "${new_prefix}" != "${prefix_lines}" ]; then
  echo "[decider] prefix mismatch; reject change" >> "${DECIDE_LOG}"
  exit 0
fi

backup="${QUEUE}.bak.$(date +%Y%m%d_%H%M%S)"
cp "${QUEUE}" "${backup}"
echo "${new_lines}" > "${QUEUE}"
echo "[decider] queue updated; backup=${backup}" >> "${DECIDE_LOG}"

exit 0
