#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${DADGAP_VENV_PATH:-${HOME}/venvs/dadgap}"
CONFIG_PATH="${1:-config/user_inputs.local.yaml}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dadgap-pycache}"
export RUFF_CACHE_DIR="${RUFF_CACHE_DIR:-/tmp/dadgap-ruff-cache}"
export MYPY_CACHE_DIR="${MYPY_CACHE_DIR:-/tmp/dadgap-mypy-cache}"

"${VENV_PATH}/bin/python" -m father_longrun print-questions
"${VENV_PATH}/bin/python" -m father_longrun check-config --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun build-phase0 --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun build-merge-contract --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun build-backbone-scaffold --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun build-reviewed-layers --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun build-refresh-spec --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun inspect-nlsy --config "${CONFIG_PATH}"
"${VENV_PATH}/bin/python" -m father_longrun build-nlsy-pilot --config "${CONFIG_PATH}"
echo "NLSY pilot build complete."
