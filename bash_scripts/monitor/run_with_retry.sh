#!/bin/bash

# Run a command with retry.
#
# Usage:
#   bash run_with_retry.sh <max_retries> <sleep_sec> -- <command...>
#
# Example:
#   bash run_with_retry.sh 5 120 -- env -i PATH=/usr/bin:/bin HOME=/root bash -lc "echo hello"
#
# Optional hang watchdog (env vars):
#   WATCH_LOG          path to log file to monitor (mtime)
#   WATCH_IDLE_SEC     seconds of no log update before treating as hang (default: 0 = disabled)
#   WATCH_INTERVAL_SEC seconds between checks (default: 60)
#   WATCH_IDLE_UTIL    avg GPU util threshold (%) to consider idle (default: 5)
#   FAIL_REGEX         if set, kill/retry when new log lines match this regex

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_SCRIPT="${SNAPSHOT_SCRIPT:-${SCRIPT_DIR}/codex_failure_snapshot.sh}"
SNAPSHOT_ON_FAIL="${SNAPSHOT_ON_FAIL:-1}"
AUTO_FIX_SCRIPT="${AUTO_FIX_SCRIPT:-${SCRIPT_DIR}/codex_auto_fix_apply.sh}"
CODEX_AUTO_APPLY="${CODEX_AUTO_APPLY:-1}"

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <max_retries> <sleep_sec> -- <command...>" >&2
  exit 2
fi

MAX_RETRIES="$1"
SLEEP_SEC="$2"
shift 2

if [ "${1:-}" = "--" ]; then
  shift 1
fi

if [ "$#" -lt 1 ]; then
  echo "Error: missing command after --" >&2
  exit 3
fi

attempt=0
infinite=0
if [ "${MAX_RETRIES}" -eq 0 ] 2>/dev/null; then
  infinite=1
fi
while true; do
  attempt=$((attempt + 1))
  echo "[retry] attempt ${attempt}/${MAX_RETRIES} @ $(date -u '+%F %T UTC')"
  echo "[retry] cmd: $*"
  set +e
  if [ -n "${WATCH_LOG:-}" ] && [ "${WATCH_IDLE_SEC:-0}" -gt 0 ]; then
    # Refresh log mtime so stale logs don't trigger immediate hang detection.
    if [ -n "${WATCH_LOG:-}" ]; then
      mkdir -p "$(dirname "${WATCH_LOG}")" 2>/dev/null || true
      touch "${WATCH_LOG}" 2>/dev/null || true
    fi
    # Run in its own session so we can kill the whole process group on hang.
    setsid "$@" &
    cmd_pid=$!
    echo "[retry] launched pid=${cmd_pid} (watching ${WATCH_LOG})"
    watch_interval="${WATCH_INTERVAL_SEC:-60}"
    idle_util="${WATCH_IDLE_UTIL:-5}"
    fail_regex="${FAIL_REGEX:-}"
    log_seen=0
    if [ -f "${WATCH_LOG}" ]; then
      log_seen="$(stat -c %s "${WATCH_LOG}" 2>/dev/null || echo 0)"
    fi
    while true; do
      if ! kill -0 "${cmd_pid}" 2>/dev/null; then
        wait "${cmd_pid}" 2>/dev/null
        status=$?
        break
      fi
      if [ -f "${WATCH_LOG}" ]; then
        last_ts="$(stat -c %Y "${WATCH_LOG}" 2>/dev/null || echo 0)"
        now_ts="$(date +%s)"
        idle_log=$((now_ts - last_ts))
      else
        idle_log=0
      fi
      if [ -n "${fail_regex}" ] && [ -f "${WATCH_LOG}" ]; then
        log_size="$(stat -c %s "${WATCH_LOG}" 2>/dev/null || echo 0)"
        if [ "${log_size}" -lt "${log_seen}" ]; then
          log_seen=0
        fi
        if [ "${log_size}" -gt "${log_seen}" ]; then
          if tail -c +$((log_seen + 1)) "${WATCH_LOG}" | grep -E -m1 "${fail_regex}" >/dev/null 2>&1; then
            echo "[retry] failure pattern detected (${fail_regex}); killing pid=${cmd_pid}"
            kill -- -${cmd_pid} 2>/dev/null || kill "${cmd_pid}" 2>/dev/null || true
            wait "${cmd_pid}" 2>/dev/null
            status=125
            if [ "${CODEX_AUTO_APPLY}" = "1" ] && [ -x "${AUTO_FIX_SCRIPT}" ]; then
              "${AUTO_FIX_SCRIPT}" --reason "fail_regex:${fail_regex}" --status "${status}" --log "${WATCH_LOG:-}" --cmd "$*" || true
            elif [ "${SNAPSHOT_ON_FAIL}" = "1" ] && [ -x "${SNAPSHOT_SCRIPT}" ]; then
              "${SNAPSHOT_SCRIPT}" --reason "fail_regex:${fail_regex}" --status "${status}" --log "${WATCH_LOG:-}" --cmd "$*" || true
            fi
            break
          fi
          log_seen="${log_size}"
        fi
      fi
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{sum+=$1; n++} END { if (n>0) printf("%d", sum/n); else print 0 }')"
      if [ "${idle_log}" -ge "${WATCH_IDLE_SEC}" ] && [ "${util}" -le "${idle_util}" ]; then
        echo "[retry] hang detected: log_idle=${idle_log}s avg_util=${util}% -> killing pid=${cmd_pid}"
        kill -- -${cmd_pid} 2>/dev/null || kill "${cmd_pid}" 2>/dev/null || true
        wait "${cmd_pid}" 2>/dev/null
        status=124
        break
      fi
      sleep "${watch_interval}"
    done
  else
    "$@"
    status=$?
  fi
  set -e
  if [ "${status}" -eq 0 ]; then
    echo "[retry] success"
    exit 0
  fi
  if [ "${SNAPSHOT_ON_FAIL}" = "1" ] && [ -x "${SNAPSHOT_SCRIPT}" ]; then
    if [ "${CODEX_AUTO_APPLY}" = "1" ] && [ -x "${AUTO_FIX_SCRIPT}" ]; then
      "${AUTO_FIX_SCRIPT}" --reason "retry_failure" --status "${status}" --log "${WATCH_LOG:-}" --cmd "$*" || true
    else
      "${SNAPSHOT_SCRIPT}" --reason "retry_failure" --status "${status}" --log "${WATCH_LOG:-}" --cmd "$*" || true
    fi
  fi
  if [ "${infinite}" -eq 0 ] && [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
    echo "[retry] failed after ${attempt} attempts (last status=${status})"
    exit "${status}"
  fi
  echo "[retry] exit ${status}; sleeping ${SLEEP_SEC}s before retry"
  sleep "${SLEEP_SEC}"
done
