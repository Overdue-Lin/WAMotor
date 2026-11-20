#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
UV_BIN="$(wamotor_find_uv)"
LIBERO_SOURCE_DIR="${WAMOTOR_ROOT}/third_party/LIBERO"
LIBERO_ENV="${WAMOTOR_ROOT}/.venvs/libero"
wamotor_clone_pinned \
  "https://github.com/Lifelong-Robot-Learning/LIBERO.git" \
  "8f1084e" \
  "${LIBERO_SOURCE_DIR}"
wamotor_prepare_benchmark_env \
  "${WAMOTOR_ROOT}/environments/libero" "${LIBERO_ENV}" "${UV_BIN}"
"${UV_BIN}" pip install --python "${LIBERO_ENV}/bin/python" \
  --no-deps -e "${LIBERO_SOURCE_DIR}"
"${LIBERO_ENV}/bin/python" "${WAMOTOR_ROOT}/scripts/setup/configure_libero.py" \
  --source-root "${LIBERO_SOURCE_DIR}" \
  --config-dir "${WAMOTOR_ROOT}/.runtime/libero"
echo "LIBERO_PYTHON=${LIBERO_ENV}/bin/python"
echo "LIBERO_SOURCE_DIR=${LIBERO_SOURCE_DIR}"
