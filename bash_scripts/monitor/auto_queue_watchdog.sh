#!/bin/bash

# Auto-queue watchdog: when GPUs are idle and no training process is running,
# launch the next command from a queue file.
#
# Usage:
#   bash bash_scripts/monitor/auto_queue_watchdog.sh <queue_file> [log_root] [interval_sec] [idle_util] [idle_iters]
#
# Env overrides:
#   ACTIVE_REGEX   regex for running training processes (default: torchrun|torch.distributed.launch|scripts/train.py)
#   STATE_FILE     path to queue progress state file
#   STOP_ON_FAIL   if set to 1, stop after a command exits non-zero (default: 0)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <queue_file> [log_root] [interval_sec] [idle_util] [idle_iters]" >&2
  exit 2
fi

QUEUE_FILE="$1"
LOG_ROOT="${2:-$(dirname "$QUEUE_FILE")}"
INTERVAL="${3:-60}"
IDLE_UTIL="${4:-5}"
IDLE_ITERS="${5:-3}"

ACTIVE_REGEX="${ACTIVE_REGEX:-torchrun|torch.distributed.launch|scripts/train.py}"
STATE_FILE="${STATE_FILE:-${LOG_ROOT}/auto_queue.state}"
DONE_FILE="${DONE_FILE:-${STATE_FILE}.done}"
LOCK_FILE="${LOG_ROOT}/auto_queue.lock"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
ADVANCE_ON_FAIL="${ADVANCE_ON_FAIL:-1}"
STATUS_FILE="${STATUS_FILE:-$(dirname "$QUEUE_FILE")/STATUS.md}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${LOG_ROOT}/auto_queue_summaries}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$(dirname "$0")/summarize_run.py}"
STATUS_SCRIPT="${STATUS_SCRIPT:-$(dirname "$0")/write_status.sh}"
AGGREGATE_SCRIPT="${AGGREGATE_SCRIPT:-$(dirname "$0")/aggregate_summaries.py}"
TREND_NOTE_SCRIPT="${TREND_NOTE_SCRIPT:-$(dirname "$0")/extract_trend_note.py}"
TREND_NOTE="${TREND_NOTE:-1}"
GATE_SCRIPT="${GATE_SCRIPT:-}"
STOP_ON_GATE_FAIL="${STOP_ON_GATE_FAIL:-1}"
GATE_TUNE="${GATE_TUNE:-0}"
GATE_TUNER_SCRIPT="${GATE_TUNER_SCRIPT:-}"
GATE_AUTO_FIX="${GATE_AUTO_FIX:-1}"
AUTO_FIX_SCRIPT="${AUTO_FIX_SCRIPT:-$(dirname "$0")/codex_auto_fix_apply.sh}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AUTO_EVAL="${AUTO_EVAL:-0}"
AUTO_EVAL_SCRIPT="${AUTO_EVAL_SCRIPT:-${REPO_ROOT}/bash_scripts/utils/opv2v_det_eval_and_html.sh}"
AUTO_EVAL_DET_HEAD_CFG="${AUTO_EVAL_DET_HEAD_CFG:-bev_centernet_wide_v4}"
AUTO_EVAL_OUT_ROOT="${AUTO_EVAL_OUT_ROOT:-${REPO_ROOT}/eval_runs/auto_queue_eval}"
AUTO_EVAL_SAMPLE_SIZE="${AUTO_EVAL_SAMPLE_SIZE:-50}"
AUTO_EVAL_SEED="${AUTO_EVAL_SEED:-42}"
AUTO_EVAL_MODES="${AUTO_EVAL_MODES:-single}"
AUTO_EVAL_STRICT="${AUTO_EVAL_STRICT:-0}"

QUEUE_DIR="$(dirname "$QUEUE_FILE")"
if [ -z "${GATE_CONFIG:-}" ]; then
  if [ -f "${QUEUE_FILE}.gate.env" ]; then
    GATE_CONFIG="${QUEUE_FILE}.gate.env"
  elif [ -f "${QUEUE_DIR}/gate.env" ]; then
    GATE_CONFIG="${QUEUE_DIR}/gate.env"
  fi
fi
if [ -n "${GATE_CONFIG:-}" ] && [ -f "${GATE_CONFIG}" ]; then
  # shellcheck disable=SC1090
  source "${GATE_CONFIG}"
fi

mkdir -p "${LOG_ROOT}"
mkdir -p "${SUMMARY_ROOT}"

if [ -f "${LOCK_FILE}" ]; then
  old_pid="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "[watchdog] Another watchdog is running (pid=${old_pid})."
    exit 1
  fi
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

QUEUE_MTIME=0
COMMANDS=()
TOTAL=0
DONE_FORMAT="legacy"

hash_cmd() {
  local input="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf "%s" "${input}" | sha256sum | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    printf "%s" "${input}" | shasum -a 256 | awk '{print $1}'
    return
  fi
  if command -v python >/dev/null 2>&1; then
    printf "%s" "${input}" | python - <<'PY'
import hashlib, sys
data = sys.stdin.buffer.read()
sys.stdout.write(hashlib.sha256(data).hexdigest())
PY
    return
  fi
  printf "%s" "${input}" | cksum | awk '{print $1}'
}

append_done_cmd() {
  local entry="$1"
  if [ "${DONE_FORMAT}" = "hash" ] || [[ "${entry}" == *$'\n'* ]]; then
    local h
    h="$(hash_cmd "${entry}")"
    printf "sha256:%s\n" "${h}" >> "${DONE_FILE}"
  else
    printf "%s\n" "${entry}" >> "${DONE_FILE}"
  fi
}

load_queue() {
  local new_mtime
  local new_commands
  local done_lines
  local line
  local in_heredoc
  local heredoc_end
  local heredoc_strip_tabs
  local current
  local done_file_is_hash
  local cmd
  local heredoc_re
  new_mtime="$(stat -c %Y "${QUEUE_FILE}" 2>/dev/null || echo 0)"
  if [ "${new_mtime}" -eq "${QUEUE_MTIME}" ] && [ "${#COMMANDS[@]}" -gt 0 ]; then
    return 0
  fi
  new_commands=()
  in_heredoc=0
  heredoc_end=""
  heredoc_strip_tabs=0
  current=""
  heredoc_re="<<-?[[:space:]]*['\"]?([A-Za-z0-9_]+)['\"]?"
  while IFS= read -r line || [ -n "${line}" ]; do
    if [ "${in_heredoc}" -eq 0 ]; then
      if [[ "${line}" =~ ^[[:space:]]*$ ]]; then
        continue
      fi
      if [[ "${line}" =~ ^[[:space:]]*# ]]; then
        continue
      fi
      current="${line}"
      if [[ "${line}" == *"<<<"* ]]; then
        new_commands+=("${current}")
        current=""
        continue
      fi
      if [[ "${line}" =~ ${heredoc_re} ]]; then
        heredoc_end="${BASH_REMATCH[1]}"
        if [[ "${line}" == *"<<-"* ]]; then
          heredoc_strip_tabs=1
        else
          heredoc_strip_tabs=0
        fi
        in_heredoc=1
        continue
      fi
      new_commands+=("${current}")
      current=""
    else
      current+=$'\n'"${line}"
      if [ "${heredoc_strip_tabs}" -eq 1 ]; then
        if [[ "${line}" =~ ^$'\t'*${heredoc_end}$ ]]; then
          new_commands+=("${current}")
          current=""
          in_heredoc=0
          heredoc_end=""
          heredoc_strip_tabs=0
        fi
      else
        if [ "${line}" = "${heredoc_end}" ]; then
          new_commands+=("${current}")
          current=""
          in_heredoc=0
          heredoc_end=""
          heredoc_strip_tabs=0
        fi
      fi
    fi
  done < "${QUEUE_FILE}"
  if [ "${in_heredoc}" -eq 1 ] && [ -n "${current}" ]; then
    new_commands+=("${current}")
  fi
  if [ "${#new_commands[@]}" -eq 0 ]; then
    echo "[watchdog] No commands found in queue file: ${QUEUE_FILE}" >&2
    return 1
  fi
  if [ -f "${DONE_FILE}" ]; then
    done_file_is_hash=0
    DONE_FORMAT="legacy"
    while IFS= read -r line; do
      [ -z "${line}" ] && continue
      if [[ "${line}" == sha256:* ]]; then
        done_file_is_hash=1
        DONE_FORMAT="hash"
      fi
      break
    done < "${DONE_FILE}"
  fi
  for cmd in "${new_commands[@]}"; do
    if [[ "${cmd}" == *$'\n'* ]]; then
      DONE_FORMAT="hash"
      break
    fi
  done
  if [ -f "${DONE_FILE}" ]; then
    if [ "${DONE_FORMAT}" = "hash" ]; then
      if [ "${done_file_is_hash:-0}" -eq 0 ]; then
        : > "${DONE_FILE}"
        if [ "${IDX:-0}" -gt 0 ]; then
          for ((i=0; i<IDX; i++)); do
            printf "sha256:%s\n" "$(hash_cmd "${new_commands[$i]}")" >> "${DONE_FILE}"
          done
        fi
      fi
      mapfile -t done_lines < "${DONE_FILE}"
      if [ "${#done_lines[@]}" -gt 0 ]; then
        for i in "${!done_lines[@]}"; do
          line="${done_lines[$i]#sha256:}"
          if [ -z "${line}" ]; then
            continue
          fi
          if [ -z "${new_commands[$i]:-}" ]; then
            echo "[watchdog] queue changed before completed index; ignoring reload"
            return 0
          fi
          if [ "$(hash_cmd "${new_commands[$i]}")" != "${line}" ]; then
            echo "[watchdog] queue changed before completed index; ignoring reload"
            return 0
          fi
        done
      fi
    else
      mapfile -t done_lines < "${DONE_FILE}"
      if [ "${#done_lines[@]}" -gt 0 ]; then
        for i in "${!done_lines[@]}"; do
          if [ "${new_commands[$i]:-}" != "${done_lines[$i]}" ]; then
            echo "[watchdog] queue changed before completed index; ignoring reload"
            return 0
          fi
        done
      fi
    fi
  fi
  COMMANDS=("${new_commands[@]}")
  TOTAL="${#COMMANDS[@]}"
  QUEUE_MTIME="${new_mtime}"
  if [ "${IDX:-0}" -gt "${TOTAL}" ]; then
    IDX="${TOTAL}"
    echo "${IDX}" > "${STATE_FILE}"
  fi
  echo "[watchdog] queue reloaded: total=${TOTAL}"
  return 0
}

IDX=0
if [ -f "${STATE_FILE}" ]; then
  IDX="$(cat "${STATE_FILE}" 2>/dev/null || echo 0)"
fi

if ! load_queue; then
  exit 3
fi
if [ "${IDX}" -gt 0 ] && [ ! -f "${DONE_FILE}" ] && [ "${TOTAL}" -ge "${IDX}" ]; then
  : > "${DONE_FILE}"
  for ((i=0; i<IDX; i++)); do
    append_done_cmd "${COMMANDS[$i]}"
  done
fi

idle_count=0

avg_util() {
  local util status
  set +e
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum+=$1; n++} END { if (n>0) printf("%d", sum/n); else print 0 }')"
  status=$?
  set -e
  if [ "${status}" -ne 0 ] || [ -z "${util}" ]; then
    echo 0
  else
    echo "${util}"
  fi
}

has_active_proc() {
  pgrep -fa "${ACTIVE_REGEX}" >/dev/null 2>&1
}

if [ -z "${GATE_SCRIPT}" ] && { [ -n "${GATE_MAX:-}" ] || [ -n "${GATE_MIN:-}" ] || [ -n "${GATE_REQUIRE_KEYS:-}" ] || [ -n "${GATE_SPLIT:-}" ]; }; then
  GATE_SCRIPT="$(dirname "$0")/gate_metrics.py"
fi
if [ -z "${GATE_TUNER_SCRIPT}" ] && [ "${GATE_TUNE}" = "1" ]; then
  GATE_TUNER_SCRIPT="$(dirname "$0")/gate_tuner.py"
fi

echo "[watchdog] queue_file=${QUEUE_FILE}"
echo "[watchdog] log_root=${LOG_ROOT}"
echo "[watchdog] interval_sec=${INTERVAL} idle_util<=${IDLE_UTIL}% idle_iters=${IDLE_ITERS}"
echo "[watchdog] active_regex=${ACTIVE_REGEX}"
if [ -n "${GATE_SCRIPT}" ]; then
  echo "[watchdog] gate_script=${GATE_SCRIPT} gate_max=${GATE_MAX:-} gate_min=${GATE_MIN:-} gate_split=${GATE_SPLIT:-} gate_only_index=${GATE_ONLY_INDEX:-} gate_only_regex=${GATE_ONLY_REGEX:-}"
fi
if [ "${GATE_TUNE}" = "1" ]; then
  echo "[watchdog] gate_tune=on tuner=${GATE_TUNER_SCRIPT:-} gate_config=${GATE_CONFIG:-}"
fi
echo "[watchdog] start_idx=${IDX} total=${TOTAL}"
  if [ -x "${STATUS_SCRIPT}" ]; then
    STATUS_FILE="${STATUS_FILE}" QUEUE_FILE="${QUEUE_FILE}" LOG_ROOT="${LOG_ROOT}" \
      "${STATUS_SCRIPT}" --state "idle" --queue "${IDX}/${TOTAL}" --note "watchdog active"
  fi

while true; do
  load_queue >/dev/null 2>&1 || true
  if [ "${IDX}" -ge "${TOTAL}" ]; then
    sleep "${INTERVAL}"
    continue
  fi

  if has_active_proc; then
    idle_count=0
    sleep "${INTERVAL}"
    continue
  fi

  util="$(avg_util)"
  if [ "${util}" -gt "${IDLE_UTIL}" ]; then
    idle_count=0
    sleep "${INTERVAL}"
    continue
  fi

  idle_count=$((idle_count + 1))
  if [ "${idle_count}" -lt "${IDLE_ITERS}" ]; then
    sleep "${INTERVAL}"
    continue
  fi

  cmd="${COMMANDS[$IDX]}"
  next_cmd=""
  if [ $((IDX + 1)) -lt "${TOTAL}" ]; then
    next_cmd="${COMMANDS[$((IDX + 1))]}"
  fi
  ts="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_ROOT}/auto_queue_${IDX}_${ts}.log"
  echo "[watchdog] launching idx=${IDX}/${TOTAL} at ${ts}" | tee -a "${log}"
  echo "[watchdog] cmd=${cmd}" | tee -a "${log}"
  if [ -x "${STATUS_SCRIPT}" ]; then
    STATUS_FILE="${STATUS_FILE}" QUEUE_FILE="${QUEUE_FILE}" LOG_ROOT="${LOG_ROOT}" \
      "${STATUS_SCRIPT}" --state "running" --queue "${IDX}/${TOTAL}" --cmd "${cmd}" --log "${log}" --next "${next_cmd}"
  fi

  # Run command in foreground so we can observe success/failure.
  set +e
  export CODEX_QUEUE_FILE="${QUEUE_FILE}"
  export CODEX_QUEUE_LOG_ROOT="${LOG_ROOT}"
  bash -lc "${cmd}" >> "${log}" 2>&1
  status=$?
  set -e

  summary_dir="${SUMMARY_ROOT}/auto_queue_${IDX}_${ts}"
  summary_md=""
  summary_json=""
  trend_note=""
  if [ -x "${SUMMARY_SCRIPT}" ]; then
    python "${SUMMARY_SCRIPT}" --log "${log}" --out_dir "${summary_dir}" --exit_status "${status}" --cmd "${cmd}" || true
    summary_md="${summary_dir}/summary.md"
    summary_json="${summary_dir}/summary.json"
    if [ "${TREND_NOTE}" = "1" ] && [ -x "${TREND_NOTE_SCRIPT}" ]; then
      trend_note="$(${TREND_NOTE_SCRIPT} --summary "${summary_json}" 2>/dev/null || true)"
    fi
  fi
  if [ -x "${AGGREGATE_SCRIPT}" ]; then
    python "${AGGREGATE_SCRIPT}" --summary_root "${SUMMARY_ROOT}" --out_md "$(dirname "$STATUS_FILE")/PROGRESS.md" || true
  fi
  if [ "${GATE_TUNE}" = "1" ] && [ -n "${GATE_TUNER_SCRIPT:-}" ] && [ -x "${GATE_TUNER_SCRIPT}" ] && [ -n "${summary_json}" ]; then
    if [ -n "${GATE_CONFIG:-}" ]; then
      "${GATE_TUNER_SCRIPT}" --summary "${summary_json}" --gate "${GATE_CONFIG}" || true
      # Reload gate env to pick up updated thresholds.
      # shellcheck disable=SC1090
      source "${GATE_CONFIG}" || true
    fi
  fi
  gate_status=0
  gate_should_run=0
  if [ -n "${GATE_SCRIPT}" ] && [ -x "${GATE_SCRIPT}" ] && [ -n "${summary_json}" ]; then
    gate_should_run=1
    if [ -n "${GATE_ONLY_INDEX:-}" ]; then
      gate_should_run=0
      IFS=',' read -r -a gate_indices <<< "${GATE_ONLY_INDEX}"
      for gate_idx in "${gate_indices[@]}"; do
        gate_idx="$(echo "${gate_idx}" | tr -d ' ')"
        if [ -n "${gate_idx}" ] && [ "${IDX}" -eq "${gate_idx}" ] 2>/dev/null; then
          gate_should_run=1
          break
        fi
      done
    fi
    if [ -n "${GATE_ONLY_REGEX:-}" ] && [ "${gate_should_run}" -eq 1 ]; then
      if ! [[ "${cmd}" =~ ${GATE_ONLY_REGEX} ]]; then
        gate_should_run=0
      fi
    fi
  fi
  if [ -n "${DECIDE_SCRIPT:-}" ] && [ -x "${DECIDE_SCRIPT}" ] && [ -n "${summary_json}" ]; then
    "${DECIDE_SCRIPT}" --summary "${summary_json}" --queue "${QUEUE_FILE}" --state "${STATE_FILE}" || true
  fi
  if [ "${gate_should_run}" -eq 1 ]; then
    "${GATE_SCRIPT}" --summary "${summary_json}" || gate_status=$?
  fi

  if [ "${status}" -eq 0 ]; then
    echo "[watchdog] cmd finished OK" | tee -a "${log}"
    if [ "${gate_status}" -eq 0 ]; then
      auto_eval_note=""
      if [ "${AUTO_EVAL}" = "1" ] && [ -x "${AUTO_EVAL_SCRIPT}" ] && [ -n "${summary_json}" ]; then
        ckpt="$(python - "${summary_json}" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(data.get("last_checkpoint") or "")
except Exception:
    print("")
PY
)"
        if [ -n "${ckpt}" ] && [ -f "${ckpt}" ]; then
          eval_out="${AUTO_EVAL_OUT_ROOT}/auto_queue_${IDX}_${ts}"
          if [[ "${eval_out}" != /* ]]; then
            eval_out="${REPO_ROOT}/${eval_out}"
          fi
          set +e
          EVAL_MODES="${AUTO_EVAL_MODES}" bash "${AUTO_EVAL_SCRIPT}" \
            "${ckpt}" "${AUTO_EVAL_DET_HEAD_CFG}" "${eval_out}" \
            "${AUTO_EVAL_SAMPLE_SIZE}" "${AUTO_EVAL_SEED}" >> "${log}" 2>&1
          auto_eval_status=$?
          set -e
          if [ "${auto_eval_status}" -eq 0 ]; then
            auto_eval_note=", auto_eval=ok"
          else
            auto_eval_note=", auto_eval=fail(${auto_eval_status})"
            if [ "${AUTO_EVAL_STRICT}" = "1" ]; then
              gate_status="${auto_eval_status}"
            fi
          fi
        else
          auto_eval_note=", auto_eval=skipped"
        fi
      fi
      if [ "${gate_status}" -eq 0 ]; then
        append_done_cmd "${cmd}"
        IDX=$((IDX + 1))
        echo "${IDX}" > "${STATE_FILE}"
        if [ -x "${STATUS_SCRIPT}" ]; then
          STATUS_FILE="${STATUS_FILE}" QUEUE_FILE="${QUEUE_FILE}" LOG_ROOT="${LOG_ROOT}" \
            "${STATUS_SCRIPT}" --state "finished" --queue "${IDX}/${TOTAL}" --cmd "${cmd}" --log "${log}" --summary "${summary_md}" --note "exit=0${auto_eval_note}${trend_note:+, ${trend_note}}"
        fi
      else
        if [ -x "${STATUS_SCRIPT}" ]; then
          STATUS_FILE="${STATUS_FILE}" QUEUE_FILE="${QUEUE_FILE}" LOG_ROOT="${LOG_ROOT}" \
            "${STATUS_SCRIPT}" --state "blocked" --queue "${IDX}/${TOTAL}" --cmd "${cmd}" --log "${log}" --summary "${summary_md}" --note "gate_fail=${gate_status}${auto_eval_note}${trend_note:+, ${trend_note}}"
        fi
        if [ "${GATE_AUTO_FIX}" = "1" ] && [ -x "${AUTO_FIX_SCRIPT}" ]; then
          "${AUTO_FIX_SCRIPT}" --reason "gate_fail" --status "${gate_status}" --log "${log}" --cmd "${cmd}" || true
        fi
        if [ "${STOP_ON_GATE_FAIL}" = "1" ]; then
          exit "${gate_status}"
        fi
      fi
    else
      if [ -x "${STATUS_SCRIPT}" ]; then
        STATUS_FILE="${STATUS_FILE}" QUEUE_FILE="${QUEUE_FILE}" LOG_ROOT="${LOG_ROOT}" \
          "${STATUS_SCRIPT}" --state "blocked" --queue "${IDX}/${TOTAL}" --cmd "${cmd}" --log "${log}" --summary "${summary_md}" --note "gate_fail=${gate_status}${trend_note:+, ${trend_note}}"
      fi
      if [ "${GATE_AUTO_FIX}" = "1" ] && [ -x "${AUTO_FIX_SCRIPT}" ]; then
        "${AUTO_FIX_SCRIPT}" --reason "gate_fail" --status "${gate_status}" --log "${log}" --cmd "${cmd}" || true
      fi
      if [ "${STOP_ON_GATE_FAIL}" = "1" ]; then
        exit "${gate_status}"
      fi
    fi
  else
    echo "[watchdog] cmd failed (status=${status})" | tee -a "${log}"
    if [ -x "${STATUS_SCRIPT}" ]; then
      STATUS_FILE="${STATUS_FILE}" QUEUE_FILE="${QUEUE_FILE}" LOG_ROOT="${LOG_ROOT}" \
        "${STATUS_SCRIPT}" --state "failed" --queue "${IDX}/${TOTAL}" --cmd "${cmd}" --log "${log}" --summary "${summary_md}" --note "exit=${status}${trend_note:+, ${trend_note}}"
    fi
    if [ "${GATE_AUTO_FIX}" = "1" ] && [ -x "${AUTO_FIX_SCRIPT}" ]; then
      "${AUTO_FIX_SCRIPT}" --reason "cmd_fail" --status "${status}" --log "${log}" --cmd "${cmd}" || true
    fi
    if [ "${STOP_ON_FAIL}" = "1" ]; then
      echo "[watchdog] STOP_ON_FAIL=1; exiting." | tee -a "${log}"
      exit "${status}"
    fi
    if [ "${ADVANCE_ON_FAIL}" = "1" ]; then
      append_done_cmd "${cmd}"
      IDX=$((IDX + 1))
      echo "${IDX}" > "${STATE_FILE}"
    fi
  fi

  idle_count=0
  sleep "${INTERVAL}"
done
