#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${DADGAP_VENV_PATH:-${HOME}/venvs/dadgap}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dadgap-pycache}"
export RUFF_CACHE_DIR="${RUFF_CACHE_DIR:-/tmp/dadgap-ruff-cache}"
export MYPY_CACHE_DIR="${MYPY_CACHE_DIR:-/tmp/dadgap-mypy-cache}"

python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/pip" install -e ".[dev,analysis]"

echo "Virtual environment created at ${VENV_PATH}."
echo "Next steps:"
echo "  cp config/user_inputs.example.yaml config/user_inputs.local.yaml"
echo "  ${VENV_PATH}/bin/father-longrun print-questions"
echo "  ${VENV_PATH}/bin/father-longrun check-config --config config/user_inputs.local.yaml"
echo "  ${VENV_PATH}/bin/father-longrun inspect-nlsy --config config/user_inputs.local.yaml"
echo "  ${VENV_PATH}/bin/father-longrun build-nlsy-pilot --config config/user_inputs.local.yaml"
