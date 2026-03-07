#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"

echo "Starting OPV2V VGGT finetunes (single then coop)."
echo "Timestamp: ${timestamp}"

bash scripts/run_opv2v_vggt_finetune_single.sh "${timestamp}_single"
bash scripts/run_opv2v_vggt_finetune_coop.sh "${timestamp}_coop"

