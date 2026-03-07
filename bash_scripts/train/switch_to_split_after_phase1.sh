#!/bin/bash

set -euo pipefail

PHASE1_ROOT="${1:?Usage: $0 <phase1_root> <train_tmux_session> <monitor_tmux_session>}"
TRAIN_SESS="${2:?Usage: $0 <phase1_root> <train_tmux_session> <monitor_tmux_session>}"
MON_SESS="${3:?Usage: $0 <phase1_root> <train_tmux_session> <monitor_tmux_session>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE1_ROOT="$(cd "${PHASE1_ROOT}" && pwd)"

PIN_LOG="${PHASE1_ROOT}/pinhole_e2e_lowres/train.log"
CYL_LOG="${PHASE1_ROOT}/cyl_e2e_lowres/train.log"
PIN_DONE_RE="Epoch: \\[1\\] Total time"
CYL_DONE_RE="Epoch: \\[1\\] Total time"

echo "[switcher] Watching Phase1 logs under ${PHASE1_ROOT}"

while true; do
  if rg -q "${PIN_DONE_RE}" "${PIN_LOG}" && rg -q "${CYL_DONE_RE}" "${CYL_LOG}"; then
    break
  fi
  sleep 120
done

echo "[switcher] Phase1 complete. Switching to split pipelines."

tmux kill-session -t "${TRAIN_SESS}" || true
tmux kill-session -t "${MON_SESS}" || true

NEW_ROOT="${REPO_ROOT}/experiments/long_runs/$(date +%Y%m%d_%H%M%S)_accel_split"
PIN_CKPT="${PHASE1_ROOT}/pinhole_e2e_lowres/checkpoint-last.pth"
CYL_CKPT="${PHASE1_ROOT}/cyl_e2e_lowres/checkpoint-last.pth"

PIN_SESS="mapany_pinhole_split_$(date +%Y%m%d_%H%M%S)"
CYL_SESS="mapany_cyl_split_$(date +%Y%m%d_%H%M%S)"
MON_NEW_SESS="mapany_monitor_split_$(date +%Y%m%d_%H%M%S)"

STARTED_ANY=0
if [[ "${SKIP_PINHOLE:-0}" != "1" ]]; then
  tmux new -d -s "${PIN_SESS}" "cd ${REPO_ROOT} && bash bash_scripts/train/accel_pinhole_pipeline_from_lowres.sh \"${NEW_ROOT}\" \"${PIN_CKPT}\""
  echo "[switcher] Pinhole session: ${PIN_SESS}"
  STARTED_ANY=1
fi
if [[ "${SKIP_CYL:-0}" != "1" ]]; then
  tmux new -d -s "${CYL_SESS}" "cd ${REPO_ROOT} && bash bash_scripts/train/accel_cyl_pipeline_from_lowres.sh \"${NEW_ROOT}\" \"${CYL_CKPT}\""
  echo "[switcher] Cyl session: ${CYL_SESS}"
  STARTED_ANY=1
fi
if [[ "${STARTED_ANY}" == "1" ]]; then
  tmux new -d -s "${MON_NEW_SESS}" "cd ${REPO_ROOT} && bash bash_scripts/monitor/monitor_long_run.sh \"${NEW_ROOT}\" 120"
  echo "[switcher] New root: ${NEW_ROOT}"
  echo "[switcher] Monitor session: ${MON_NEW_SESS}"
fi

if [[ "${STARTED_ANY}" == "0" ]]; then
  echo "[switcher] SKIP_PINHOLE=1 and SKIP_CYL=1 set; no new pipelines started."
fi
