#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${CKPT_PATH:?Set CKPT_PATH to the FasterWAM LIBERO checkpoint.}"
: "${DATASET_STATS_PATH:?Set DATASET_STATS_PATH to dataset_stats.json.}"

PYTHON_BIN="${REPO_ROOT}/.venvs/libero-plus/bin/python"
LIBERO_PLUS_SOURCE_DIR="${REPO_ROOT}/third_party/LIBERO-plus"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "LIBERO-Plus Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [ ! -d "${LIBERO_PLUS_SOURCE_DIR}" ]; then
    echo "LIBERO-Plus source is missing: ${LIBERO_PLUS_SOURCE_DIR}" >&2
    exit 1
fi
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
TASK_NAME="${TASK_NAME:-libero_fasterwam_2cam224}"

EXTRA_ARGS=()
SPECULATIVE=false
RESUME_DIR=""
while [ "$#" -gt 0 ]; do
    case "$1" in
    --speculative)
        SPECULATIVE=true
        shift
        ;;
    --resume)
        if [ "$#" -lt 2 ] || [ -z "$2" ]; then
            echo "--resume requires an existing evaluation directory" >&2
            exit 1
        fi
        RESUME_DIR="$2"
        shift 2
        ;;
    *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
done
if [ -n "${RESUME_DIR}" ] && [ ! -d "${RESUME_DIR}" ]; then
    echo "Resume directory does not exist: ${RESUME_DIR}" >&2
    exit 1
fi

export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-${REPO_ROOT}/.runtime/libero-plus}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.runtime/numba/libero-plus}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.runtime/matplotlib/libero-plus}"
export PYTHONPATH="${LIBERO_PLUS_SOURCE_DIR}:${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

"${PYTHON_BIN}" scripts/setup/configure_libero.py \
    --source-root "${LIBERO_PLUS_SOURCE_DIR}" \
    --config-dir "${LIBERO_CONFIG_PATH}"

MANAGER_ARGS=(
    --config-name sim_libero_plus
    "task=${TASK_NAME}"
    "ckpt=${CKPT_PATH}"
    "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
    "MULTIRUN.num_gpus=${NUM_GPUS}"
    "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
    "EVALUATION.speculative=${SPECULATIVE}"
)
if [ -n "${RESUME_DIR}" ]; then
    MANAGER_ARGS+=(
        "EVALUATION.output_dir=${RESUME_DIR}"
        "EVALUATION.resume=true"
    )
fi
MANAGER_ARGS+=("${EXTRA_ARGS[@]}")

"${PYTHON_BIN}" experiments/libero/run_libero_manager.py "${MANAGER_ARGS[@]}"

if [ -n "${RESUME_DIR}" ]; then
    SUMMARY_INPUT_DIR="${RESUME_DIR}"
else
    SUMMARY_INPUT_DIR="$(find "${REPO_ROOT}/evaluate_results/libero-plus/${TASK_NAME}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi
if [ -z "${SUMMARY_INPUT_DIR}" ] || [ ! -d "${SUMMARY_INPUT_DIR}" ]; then
    echo "Could not find LIBERO-Plus evaluation output directory for summary." >&2
    exit 1
fi
"${PYTHON_BIN}" scripts/summarize_libero_plus_categories.py --input "${SUMMARY_INPUT_DIR}"
