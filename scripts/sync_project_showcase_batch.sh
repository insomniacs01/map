#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MAP_ROOT}/.." && pwd)"
WS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
SYNC_PY="${SCRIPT_DIR}/sync_benchmark_project_v2.py"
CLASH_ENV="${WS_ROOT}/yijinxiong/clash/env.sh"
MANIFEST_DIR="${WS_ROOT}/tmp/inbox/20260306_project_showcase_manifests"

PROJECT_ORG="coopVGGT"
PROJECT_NUMBER=2
TOKEN_ENV="GITHUB_TOKEN"
OWNER="MassimoQu"
REPO_NAME="map-anything"
BRANCH="lantu_A800"
RATE_KB=20
LIVE=0

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--dry-run] [--live] [--rate-kb N] [--project-org ORG] [--project-number N] [--token-env ENV]

Defaults:
  --dry-run          local manifest generation only
  --rate-kb 20       conservative bandwidth cap for live sync
  --project-org coopVGGT
  --project-number 2
  --token-env GITHUB_TOKEN
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      LIVE=0; shift ;;
    --live)
      LIVE=1; shift ;;
    --rate-kb)
      RATE_KB="${2:-}"; shift 2 ;;
    --project-org)
      PROJECT_ORG="${2:-}"; shift 2 ;;
    --project-number)
      PROJECT_NUMBER="${2:-}"; shift 2 ;;
    --token-env)
      TOKEN_ENV="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

mkdir -p "${MANIFEST_DIR}"

if (( LIVE )); then
  [[ -n "${!TOKEN_ENV:-}" ]] || { echo "Missing token env: ${TOKEN_ENV}" >&2; exit 1; }
  if [[ -f "${CLASH_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${CLASH_ENV}"
  fi
fi

run_one() {
  local title="$1"
  local run_dir="$2"
  local result="$3"
  local parent_epic="$4"
  local tier="$5"
  local priority="$6"
  local pack="$7"
  local gate_override="$8"
  local status_note="$9"
  local next_action="${10}"

  local summary_json="${MAP_ROOT}/eval_runs/${run_dir}/summary_test.json"
  local manifest_out="${MANIFEST_DIR}/${run_dir}.manifest.json"

  [[ -f "${summary_json}" ]] || { echo "Missing summary: ${summary_json}" >&2; return 1; }

  local -a cmd=(
    python "${SYNC_PY}"
    --summary_json "${summary_json}"
    --item_title "${title}"
    --result "${result}"
    --item_type "Run"
    --parent_epic "${parent_epic}"
    --priority "${priority}"
    --owner "${OWNER}"
    --curation_tier "${tier}"
    --repo_name "${REPO_NAME}"
    --branch "${BRANCH}"
    --artifact_link "map-anything/eval_runs/${run_dir}"
    --status_note "${status_note}"
    --next_action "${next_action}"
  )

  if [[ -n "${pack}" ]]; then
    cmd+=(--pack "${pack}")
  fi
  if [[ -n "${gate_override}" ]]; then
    cmd+=(--gate_status_override "${gate_override}")
  fi

  if (( LIVE )); then
    cmd+=(
      --project_org "${PROJECT_ORG}"
      --project_number "${PROJECT_NUMBER}"
      --github_token_env "${TOKEN_ENV}"
    )
    if command -v trickle >/dev/null 2>&1; then
      trickle -u "${RATE_KB}" -d "${RATE_KB}" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  else
    cmd+=(--dry_run --manifest_out "${manifest_out}")
    "${cmd[@]}"
  fi
}

run_one \
  "Run: Geometry baseline — sfix_direct (Test500 calibrated coop)" \
  "geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop" \
  "Baseline" \
  "Geometry Base Models" \
  "Reference" \
  "P2" \
  "report500" \
  "" \
  "Historical geometry baseline under the fixed Test500 / calibrated_sfm / coop-first contract." \
  "Use as the baseline reference for promoted and blocked geometry runs."

run_one \
  "Run: Geometry promoted — nm4_scaleonly_w02 (Test500 calibrated coop)" \
  "geom_recheck_test500_20260223_nm4_scaleonly_w02_calibrated" \
  "Promoted" \
  "Geometry Base Models" \
  "Key" \
  "P1" \
  "report500" \
  "" \
  "Current promoted geometry starting point under the canonical deploy contract." \
  "Compare with blocked references and decide the next smallest pose-focused validation."

run_one \
  "Run: Geometry reference — remote_nm_w015_r1 (best pre-promotion)" \
  "geom_ablation_test500_20260222_remote_nm_w015_r1" \
  "Reference" \
  "Geometry Base Models" \
  "Reference" \
  "P2" \
  "report500" \
  "" \
  "Best pre-promotion geometry reference from the fixed-contract sweep; useful for explaining the path into the promoted branch." \
  "Keep as historical reference when comparing promoted and near-miss outcomes."

run_one \
  "Run: Geometry near-miss — local_full_w02_r1_detach" \
  "geom_ablation_test500_20260222_local_full_w02_r1_detach" \
  "NearMiss" \
  "Geometry Base Models" \
  "Reference" \
  "P2" \
  "report500" \
  "" \
  "Near-miss geometry run: coop scale approached target but single-agent pose guardrail still failed." \
  "Only revive if a new pose-preserving strategy is explicitly chosen."

run_one \
  "Run: Detection baseline — e2e_v5 single (Test50 pinhole)" \
  "det_e2e_v5_eval_t005" \
  "Baseline" \
  "Detection" \
  "Key" \
  "P1" \
  "" \
  "skip" \
  "Canonical single-agent pinhole detection baseline on the shared Test50 frames contract." \
  "Use with the coop baseline to explain the current cooperative detection gap."

run_one \
  "Run: Detection baseline — e2e_v5 coop (Test50 pinhole)" \
  "det_e2e_v5_coop_eval_t005" \
  "Baseline" \
  "Detection" \
  "Key" \
  "P1" \
  "" \
  "skip" \
  "Canonical cooperative pinhole detection baseline on the same Test50 frames contract." \
  "Pair with the single baseline when reviewing whether geometry alignment is good enough for coop gains."
