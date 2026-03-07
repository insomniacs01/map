#!/bin/bash

# Auto-run Codex to propose a fix patch and apply it before retrying.
#
# Env (optional):
#   CODEX_MODEL         override model (default: gpt-5.2-codex)
#   CODEX_WORKDIR       override working dir for codex exec (default: repo root)
#   CODEX_BIN           override codex binary path
#   CODEX_HOME          override codex home (default: /J6P-perception/yijinxiong_workspace/.codex)
#   CODEX_PROXY         override proxy (default: from .codex/config.toml if present)
#   CODEX_REASONING_EFFORT override reasoning effort (default: xhigh)
#   CODEX_MAX_PATCH_BYTES  max patch size to apply (default: 200000)
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
SNAPSHOT_SCRIPT="${SNAPSHOT_SCRIPT:-${SCRIPT_DIR}/codex_failure_snapshot.sh}"

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
      shift;;
  esac
done

tmp_dir="${TMPDIR:-/tmp}"
snapshot_out="${tmp_dir}/codex_snapshot_dir_$$.txt"
snapshot_prompt="${tmp_dir}/codex_snapshot_prompt_$$.txt"

CODEX_AUTO_DEBUG=0 \
SNAPSHOT_OUT_FILE="${snapshot_out}" \
SNAPSHOT_PROMPT_FILE="${snapshot_prompt}" \
"${SNAPSHOT_SCRIPT}" --reason "${REASON}" --status "${STATUS}" --log "${LOG_PATH}" --cmd "${CMD_TEXT}"

if [ ! -f "${snapshot_out}" ] || [ ! -f "${snapshot_prompt}" ]; then
  echo "[codex_auto_fix] snapshot outputs missing; skip auto-fix"
  exit 0
fi

RUN_DIR="$(cat "${snapshot_out}")"
PROMPT_SRC="$(cat "${snapshot_prompt}")"

if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ] || [ -z "${PROMPT_SRC}" ] || [ ! -f "${PROMPT_SRC}" ]; then
  echo "[codex_auto_fix] invalid snapshot paths; skip auto-fix"
  exit 0
fi

FIX_PROMPT="${RUN_DIR}/prompt_fix.md"
PATCH_FILE="${RUN_DIR}/codex_fix.patch"
CODEX_LOG="${RUN_DIR}/codex_fix_exec.log"
CODEX_LAST="${RUN_DIR}/codex_fix_last_message.txt"
APPLY_LOG="${RUN_DIR}/codex_fix_apply.log"

touch "${APPLY_LOG}" "${CODEX_LOG}" "${PATCH_FILE}" "${CODEX_LAST}" 2>/dev/null || true
{
  echo "[codex_auto_fix] start $(date -u '+%F %T UTC')"
  echo "[codex_auto_fix] PATH=${PATH}"
} >> "${APPLY_LOG}" 2>&1

{
  cat "${PROMPT_SRC}"
  echo
  echo "## Auto-fix Instructions"
  echo "- Output a single unified diff patch (diff --git format) against the repo root."
  echo "- If the fix must touch files outside the repo, use absolute paths in diff --git."
  echo "- Do not include any explanation or markdown fences."
  echo "- If no code change is needed, but you want to adjust the watchdog queue, output:"
  echo "  QUEUE_APPEND"
  echo "  <one command per line>"
  echo "- If nothing should be changed, output exactly: NO_PATCH"
} > "${FIX_PROMPT}"

CODEX_MODEL="${CODEX_MODEL:-gpt-5.2-codex}"
CODEX_WORKDIR="${CODEX_WORKDIR:-${REPO_ROOT}}"
CODEX_HOME="${CODEX_HOME:-/J6P-perception/yijinxiong_workspace/.codex}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
CODEX_TIMEOUT_SEC="${CODEX_TIMEOUT_SEC:-600}"
CODEX_TIMEOUT_KILL_SEC="${CODEX_TIMEOUT_KILL_SEC:-30}"
CLASH_DIR="${CLASH_DIR:-/J6P-perception/yijinxiong_workspace/yijinxiong/clash}"

CODEX_BIN="${CODEX_BIN:-}"
if [ -z "${CODEX_BIN}" ]; then
  if [ -x "/J6P-perception/yijinxiong_workspace/node_modules/.bin/codex" ]; then
    CODEX_BIN="/J6P-perception/yijinxiong_workspace/node_modules/.bin/codex"
  elif command -v codex >/dev/null 2>&1; then
    CODEX_BIN="$(command -v codex)"
  fi
fi

NODE_BIN="$(command -v node 2>/dev/null || true)"
if [ -z "${NODE_BIN}" ] && [ -x "/J6P-perception/yijinxiong_workspace/.nvm/versions/node/v20.19.6/bin/node" ]; then
  NODE_BIN="/J6P-perception/yijinxiong_workspace/.nvm/versions/node/v20.19.6/bin/node"
fi

{
  echo "[codex_auto_fix] CODEX_BIN=${CODEX_BIN}"
  echo "[codex_auto_fix] NODE_BIN=${NODE_BIN}"
} >> "${APPLY_LOG}" 2>&1

if [ -z "${CODEX_BIN}" ] || { [ ! -x "${CODEX_BIN}" ] && [ -z "${NODE_BIN}" ]; }; then
  echo "[codex_auto_fix] codex binary not found; skip auto-fix" | tee -a "${APPLY_LOG}"
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

echo "[codex_auto_fix] running codex exec -> ${CODEX_LOG} (timeout=${CODEX_TIMEOUT_SEC}s)" | tee -a "${APPLY_LOG}"
if [ -n "${NODE_BIN}" ]; then
  CODEX_CMD=("${NODE_BIN}" "${CODEX_BIN}")
else
  CODEX_CMD=("${CODEX_BIN}")
fi
set +e
env CODEX_HOME="${CODEX_HOME}" http_proxy="${PROXY}" https_proxy="${PROXY}" \
  timeout --kill-after "${CODEX_TIMEOUT_KILL_SEC}" "${CODEX_TIMEOUT_SEC}" \
  "${CODEX_CMD[@]}" exec \
  --cd "${CODEX_WORKDIR}" \
  --model "${CODEX_MODEL}" \
  -c "model_reasoning_effort=${CODEX_REASONING_EFFORT}" \
  --dangerously-bypass-approvals-and-sandbox \
  --output-last-message "${CODEX_LAST}" \
  - < "${FIX_PROMPT}" > "${PATCH_FILE}" 2> "${CODEX_LOG}"
CODEX_STATUS=$?
set -e

if [ "${CODEX_STATUS}" -eq 124 ]; then
  echo "[codex_auto_fix] codex exec timed out after ${CODEX_TIMEOUT_SEC}s" | tee -a "${APPLY_LOG}"
elif [ "${CODEX_STATUS}" -ne 0 ]; then
  echo "[codex_auto_fix] codex exec exited status=${CODEX_STATUS}" | tee -a "${APPLY_LOG}"
fi

if [ ! -s "${PATCH_FILE}" ]; then
  if [ -s "${CODEX_LAST}" ]; then
    cp -f "${CODEX_LAST}" "${PATCH_FILE}" 2>/dev/null || true
  elif rg -n '^diff --git' "${CODEX_LOG}" >/dev/null 2>&1; then
    awk 'BEGIN{s=0} /^diff --git/{s=1} s{print}' "${CODEX_LOG}" > "${PATCH_FILE}" || true
  fi
fi

if [ ! -s "${PATCH_FILE}" ]; then
  echo "[codex_auto_fix] empty patch output; skip" | tee -a "${APPLY_LOG}"
  tail -n 50 "${CODEX_LOG}" >> "${APPLY_LOG}" 2>&1 || true
  exit 0
fi

# Strip code fences to make control tokens detectable.
tmp_nofences="${PATCH_FILE}.nofences"
if rg -n '^```' "${PATCH_FILE}" >/dev/null 2>&1; then
  sed '/^```/d' "${PATCH_FILE}" > "${tmp_nofences}" || true
  if [ -s "${tmp_nofences}" ]; then
    mv -f "${tmp_nofences}" "${PATCH_FILE}"
  else
    rm -f "${tmp_nofences}"
  fi
fi

if rg -n "^[[:space:]]*QUEUE_APPEND[[:space:]]*$" "${PATCH_FILE}" >/dev/null 2>&1; then
  QUEUE_FILE="${CODEX_QUEUE_FILE:-}"
  if [ -z "${QUEUE_FILE}" ] || [ ! -f "${QUEUE_FILE}" ]; then
    echo "[codex_auto_fix] QUEUE_APPEND requested but CODEX_QUEUE_FILE is missing" | tee -a "${APPLY_LOG}"
    exit 0
  fi
  append_line="$(rg -n '^[[:space:]]*QUEUE_APPEND[[:space:]]*$' "${PATCH_FILE}" | head -n 1 | cut -d: -f1)"
  awk -v start="$((append_line + 1))" 'NR>=start && $0 !~ /^[[:space:]]*$/ {print}' "${PATCH_FILE}" >> "${QUEUE_FILE}"
  echo "[codex_auto_fix] appended to queue: ${QUEUE_FILE}" | tee -a "${APPLY_LOG}"
  exit 0
fi

if rg -n "^[[:space:]]*NO_PATCH[[:space:]]*$" "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[codex_auto_fix] no patch suggested" | tee -a "${APPLY_LOG}"
  exit 0
fi

if ! head -n 1 "${PATCH_FILE}" | grep -q "^diff --git" && rg -n '^diff --git' "${PATCH_FILE}" >/dev/null 2>&1; then
  tmp_patch="${PATCH_FILE}.trim"
  awk 'BEGIN{s=0} /^diff --git/{s=1} s{print}' "${PATCH_FILE}" > "${tmp_patch}" || true
  if [ -s "${tmp_patch}" ]; then
    mv -f "${tmp_patch}" "${PATCH_FILE}"
  else
    rm -f "${tmp_patch}"
  fi
fi

max_bytes="${CODEX_MAX_PATCH_BYTES:-200000}"
patch_bytes="$(stat -c %s "${PATCH_FILE}" 2>/dev/null || echo 0)"
if [ "${patch_bytes}" -gt "${max_bytes}" ]; then
  echo "[codex_auto_fix] patch too large (${patch_bytes} bytes); skip" | tee -a "${APPLY_LOG}"
  exit 0
fi

if ! grep -q "^diff --git" "${PATCH_FILE}"; then
  echo "[codex_auto_fix] patch format missing diff header; skip" | tee -a "${APPLY_LOG}"
  exit 0
fi

git_check_log="${RUN_DIR}/codex_fix_git_check.log"
if git -C "${REPO_ROOT}" apply --check --verbose "${PATCH_FILE}" >"${git_check_log}" 2>&1; then
  if rg -n "Skipped patch" "${git_check_log}" >/dev/null 2>&1; then
    echo "[codex_auto_fix] git apply check skipped patch; will try patch(1)" | tee -a "${APPLY_LOG}"
  else
    if git -C "${REPO_ROOT}" apply "${PATCH_FILE}" >/dev/null 2>&1; then
      echo "[codex_auto_fix] patch applied (git apply)" | tee -a "${APPLY_LOG}"
      exit 0
    fi
    echo "[codex_auto_fix] git apply failed after check; will try patch(1)" | tee -a "${APPLY_LOG}"
  fi
else
  echo "[codex_auto_fix] git apply check failed; will try patch(1)" | tee -a "${APPLY_LOG}"
fi

# Try patch(1) against repo root (handles skipped patches).
if (cd "${REPO_ROOT}" && patch -p1 --dry-run < "${PATCH_FILE}" >/dev/null 2>&1); then
  (cd "${REPO_ROOT}" && patch -p1 --forward -N < "${PATCH_FILE}" >/dev/null 2>&1 || true)
  echo "[codex_auto_fix] patch applied (patch -p1)" | tee -a "${APPLY_LOG}"
  exit 0
fi

# Fallback: allow absolute-path patches for external locations.
ext_patch=0
strip_level=1
if rg -n '^diff --git /' "${PATCH_FILE}" >/dev/null 2>&1; then
  ext_patch=1
  strip_level=0
else
  while read -r a b; do
    a="${a#a/}"; b="${b#b/}"
    for p in "$a" "$b"; do
      [ -z "${p}" ] && continue
      if [[ "${p}" == /* ]]; then
        ext_patch=1
        strip_level=0
        break
      fi
      if command -v realpath >/dev/null 2>&1; then
        rp="$(realpath -m "${REPO_ROOT}/${p}")"
        if [[ "${rp}" != "${REPO_ROOT}"* ]]; then
          ext_patch=1
          strip_level=1
          break
        fi
      fi
    done
    [ "${ext_patch}" -eq 1 ] && break
  done < <(awk '/^diff --git /{print $3, $4}' "${PATCH_FILE}")
fi

if [ "${ext_patch}" -eq 1 ]; then
  echo "[codex_auto_fix] attempting external patch with patch -p${strip_level}" | tee -a "${APPLY_LOG}"
  if patch -p"${strip_level}" --dry-run < "${PATCH_FILE}" >/dev/null 2>&1; then
    patch -p"${strip_level}" --forward -N < "${PATCH_FILE}" >/dev/null 2>&1 || true
    echo "[codex_auto_fix] patch applied (patch -p${strip_level})" | tee -a "${APPLY_LOG}"
    exit 0
  fi
fi

echo "[codex_auto_fix] patch failed check; skip" | tee -a "${APPLY_LOG}"

exit 0
