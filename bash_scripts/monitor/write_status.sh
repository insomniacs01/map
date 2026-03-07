#!/bin/bash

# Write a simple Markdown status file for auto-queue progress.
#
# Env:
#   STATUS_FILE   Path to status markdown (default: ./STATUS.md)
#   QUEUE_FILE    Queue file path (optional)
#   LOG_ROOT      Log root path (optional)
#
# Args:
#   --state <state>
#   --queue <idx/total>
#   --cmd <command>
#   --log <log_path>
#   --summary <summary_md>
#   --note <note>
#   --next <next_cmd>

set -euo pipefail

STATE=""
QUEUE=""
CMD=""
LOG=""
SUMMARY=""
NOTE=""
NEXT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --state) STATE="${2:-}"; shift 2;;
    --queue) QUEUE="${2:-}"; shift 2;;
    --cmd) CMD="${2:-}"; shift 2;;
    --log) LOG="${2:-}"; shift 2;;
    --summary) SUMMARY="${2:-}"; shift 2;;
    --note) NOTE="${2:-}"; shift 2;;
    --next) NEXT="${2:-}"; shift 2;;
    *) shift;;
  esac
done

STATUS_FILE="${STATUS_FILE:-./STATUS.md}"
NOW_BJ="$(TZ=Asia/Shanghai date '+%F %T (Beijing)')"

{
  echo "# Auto Queue Status"
  echo "- Updated: ${NOW_BJ}"
  echo "- State: ${STATE}"
  if [ -n "${QUEUE}" ]; then echo "- Queue: ${QUEUE}"; fi
  if [ -n "${QUEUE_FILE:-}" ]; then echo "- QueueFile: ${QUEUE_FILE}"; fi
  if [ -n "${LOG_ROOT:-}" ]; then echo "- LogRoot: ${LOG_ROOT}"; fi
  if [ -n "${CMD}" ]; then echo "- Command: ${CMD}"; fi
  if [ -n "${LOG}" ]; then echo "- Log: ${LOG}"; fi
  if [ -n "${SUMMARY}" ]; then echo "- Summary: ${SUMMARY}"; fi
  if [ -n "${NOTE}" ]; then echo "- Note: ${NOTE}"; fi
  if [ -n "${NEXT}" ]; then echo "- Next: ${NEXT}"; fi
} > "${STATUS_FILE}"

# Optional: update workspace-wide status aggregation (non-blocking).
if [ -n "${STATUS_AGG_SCRIPT:-}" ] && [ -x "${STATUS_AGG_SCRIPT}" ]; then
  ("${STATUS_AGG_SCRIPT}" >/dev/null 2>&1) &
fi
