#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${TASK_NAME:?Set TASK_NAME to a Hydra task config, e.g. libero_fasterwam_2cam224}"
: "${CKPT_PATH:?Set CKPT_PATH to the selected model checkpoint}"
: "${DATASET_STATS_PATH:?Set DATASET_STATS_PATH to dataset_stats.json}"

MODEL="${MODEL:-}"
if [ -z "${MODEL}" ]; then
    case "${TASK_NAME}" in
    *fasterwam*) MODEL=fasterwam ;;
    *fastwam*) MODEL=fastwam ;;
    *jointwam*) MODEL=jointwam ;;
    *) MODEL="${TASK_NAME}" ;;
    esac
fi
OUT="${OUT:-${REPO_ROOT}/evaluate_results/latency}"
SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
BENCHMARK_ITERS="${BENCHMARK_ITERS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-30}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
ACTION_INFER_MODE="${ACTION_INFER_MODE:-auto}"
SPEC_W="${SPEC_W:-10}"
SPEC_EPSILON="${SPEC_EPSILON:-5e-2}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_NAME="${RUN_NAME:-}"

PYTHON_BIN="${REPO_ROOT}/.venvs/libero/bin/python"
LIBERO_SOURCE_DIR="${REPO_ROOT}/third_party/LIBERO"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "LIBERO Python is not executable: ${PYTHON_BIN}" >&2
    echo "Run: bash scripts/setup/install_libero.sh" >&2
    exit 1
fi
if [ ! -d "${LIBERO_SOURCE_DIR}" ]; then
    echo "LIBERO source is missing: ${LIBERO_SOURCE_DIR}" >&2
    exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.runtime/numba/libero-latency}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.runtime/matplotlib/libero-latency}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.runtime/libero}"
export PYTHONPATH="${LIBERO_SOURCE_DIR}:${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

EXTRA_ARGS=()
SPECULATIVE=false
while [ "$#" -gt 0 ]; do
    case "$1" in
    --speculative)
        SPECULATIVE=true
        shift
        ;;
    *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
done

"${PYTHON_BIN}" scripts/setup/configure_libero.py \
    --source-root "${LIBERO_SOURCE_DIR}" \
    --config-dir "${LIBERO_CONFIG_PATH}"

ARGS=(
    --model "${MODEL}"
    --task "${TASK_NAME}"
    --checkpoint "${CKPT_PATH}"
    --dataset-stats "${DATASET_STATS_PATH}"
    --output "${OUT}"
    --suite "${SUITE}"
    --task-id "${TASK_ID}"
    --warmup-iters "${WARMUP_ITERS}"
    --benchmark-iters "${BENCHMARK_ITERS}"
    --num-steps-wait "${NUM_STEPS_WAIT}"
    --inference-steps "${INFERENCE_STEPS}"
    --action-horizon "${ACTION_HORIZON}"
    --action-infer-mode "${ACTION_INFER_MODE}"
    --spec-w "${SPEC_W}"
    --spec-epsilon "${SPEC_EPSILON}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
)
if [ "${SPECULATIVE}" = true ]; then
    ARGS+=(--speculative)
fi
if [ -n "${RUN_NAME}" ]; then
    ARGS+=(--run-name "${RUN_NAME}")
fi
ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

"${PYTHON_BIN}" experiments/latency/benchmark_latency.py "${ARGS[@]}"

printf 'Latency experiment finished. Results: %s/%s\n' "${OUT}" "${MODEL}"
