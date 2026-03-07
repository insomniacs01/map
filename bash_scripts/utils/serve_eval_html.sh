#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-map-anything/eval_runs/}"
PORT="${2:-8383}"

if [[ ! -d "${DIR}" ]]; then
  echo "ERROR: directory not found: ${DIR}" >&2
  exit 1
fi

PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

echo "Serving: ${DIR}"
echo "URL: http://127.0.0.1:${PORT}/ (use VS Code 'Ports' to forward ${PORT})"
exec "${PYTHON_BIN}" -m http.server "${PORT}" --bind 0.0.0.0 --directory "${DIR}"

