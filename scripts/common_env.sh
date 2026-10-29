#!/bin/bash
# ==============================================================================
# common_env.sh - Common Environment, CLI Parsing, and Cluster Dispatcher
# For Physical Review B Data Regeneration Suite
# ==============================================================================

set -e

# Default Physical & Numerical Constants
export ALPHA="${ALPHA:-0.27}"
export OMEGA="${OMEGA:-1.0}"
export EPS="${EPS:-0.0005}"       # Standardized default initial perturbation
export SEED="${SEED:-42}"
export DT="${DT:-0.0196349540849362}"

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"

# Default Execution Mode and Hardware
if [ -n "$SLURM_JOB_ID" ]; then
    # Running inside a SLURM job
    IS_SLURM_JOB=true
    if [ -n "$SLURM_JOB_PARTITION" ]; then
        PARTITION="$SLURM_JOB_PARTITION"
    fi
else
    IS_SLURM_JOB=false
fi

# Locate binary by name ($1: binary name, $2: requested backend cpu/gpu)
find_executable() {
    local bin_name="$1"
    local backend="${2:-$BACKEND}"

    local candidates=()
    if [ "$backend" = "gpu" ]; then
        candidates=(
            "${PROJECT_DIR}/build/bin/${bin_name}"
            "${PROJECT_DIR}/build_slurm/bin/${bin_name}"
            "${PROJECT_DIR}/build_local/bin/${bin_name}"
        )
    else
        candidates=(
            "${PROJECT_DIR}/build_local/bin/${bin_name}"
            "${PROJECT_DIR}/build/bin/${bin_name}"
            "${PROJECT_DIR}/build_slurm/bin/${bin_name}"
        )
    fi

    for cand in "${candidates[@]}"; do
        if [ -x "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done

    # Fallback to PATH
    if command -v "$bin_name" >/dev/null 2>&1; then
        command -v "$bin_name"
        return 0
    fi

    echo ""
    return 1
}

# Cluster environment setup
setup_cluster_modules() {
    if command -v module >/dev/null 2>&1; then
        module purge 2>/dev/null || true
        module load cmake/3.24.2 2>/dev/null || module load cmake 2>/dev/null || true
        module load gcc/8.5.0-5sg556d 2>/dev/null || module load gcc/8.2.0-cytb66h 2>/dev/null || module load gcc 2>/dev/null || true
        if [ "$BACKEND" = "gpu" ] || [ "$PARTITION" = "gpu" ]; then
            module load cuda/11.1.0-zj6lnzj 2>/dev/null || module load cuda 2>/dev/null || true
            export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:$LD_LIBRARY_PATH"
        fi
    fi

    # OpenMP configuration for CPU execution
    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    else
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
    fi
    export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
    export OMP_PLACES="${OMP_PLACES:-threads}"
}

# Parse common flags from command line arguments
# Usage: parse_common_args "$@"
parse_common_args() {
    SUBMIT_SLURM=false
    CLI_PASSTHROUGH=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --backend)
                BACKEND="$2"
                shift 2
                ;;
            --partition)
                PARTITION="$2"
                shift 2
                ;;
            --slurm)
                SUBMIT_SLURM=true
                shift
                ;;
            --eps)
                EPS="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --size)
                SIZE="$2"
                shift 2
                ;;
            --L)
                L="$2"
                shift 2
                ;;
            --out-dir)
                OUT_DIR="$2"
                shift 2
                ;;
            --h0)
                H0="$2"
                shift 2
                ;;
            --ha)
                HA="$2"
                shift 2
                ;;
            --omega)
                OMEGA="$2"
                shift 2
                ;;
            --alpha)
                ALPHA="$2"
                shift 2
                ;;
            --dt)
                DT="$2"
                shift 2
                ;;
            --h0-min)
                H0_MIN="$2"
                shift 2
                ;;
            --h0-max)
                H0_MAX="$2"
                shift 2
                ;;
            --num-h0)
                NUM_H0="$2"
                H0_STEPS="$2"
                shift 2
                ;;
            --h0-steps)
                H0_STEPS="$2"
                NUM_H0="$2"
                shift 2
                ;;
            --ha-min)
                HA_MIN="$2"
                shift 2
                ;;
            --ha-max)
                HA_MAX="$2"
                shift 2
                ;;
            --ha-steps)
                HA_STEPS="$2"
                shift 2
                ;;
            --k-min)
                K_MIN="$2"
                shift 2
                ;;
            --k-max)
                K_MAX="$2"
                shift 2
                ;;
            --num-k)
                NUM_K="$2"
                shift 2
                ;;
            --transient)
                TRANSIENT="$2"
                TRANSIENT_TIME="$2"
                shift 2
                ;;
            --eval)
                EVAL="$2"
                AVERAGING_TIME="$2"
                shift 2
                ;;
            --max-power)
                MAX_POWER="$2"
                shift 2
                ;;
            --exp-min)
                EXP_MIN="$2"
                shift 2
                ;;
            --exp-max)
                EXP_MAX="$2"
                shift 2
                ;;
            --help|-h)
                SHOW_HELP=true
                shift
                ;;
            *)
                CLI_PASSTHROUGH+=("$1")
                shift
                ;;
        esac
    done

    # Default backend and partition selection
    if [ -z "$BACKEND" ]; then
        if [ "$PARTITION" = "knl_tacc" ]; then
            BACKEND="cpu"
        elif [ "$PARTITION" = "gpu" ]; then
            BACKEND="gpu"
        elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            BACKEND="gpu"
        else
            BACKEND="cpu"
        fi
    fi

    if [ -z "$PARTITION" ]; then
        if [ "$BACKEND" = "gpu" ]; then
            PARTITION="gpu"
        else
            PARTITION="knl_tacc"
        fi
    fi
}

# Dispatcher: Submit to SLURM if --slurm was requested outside of SLURM
check_slurm_dispatch() {
    local script_file="$1"
    shift
    local script_args=("$@")

    if [ "$SUBMIT_SLURM" = true ] && [ "$IS_SLURM_JOB" = false ]; then
        if ! command -v sbatch >/dev/null 2>&1; then
            echo "Error: sbatch command not found. Cannot submit to SLURM on this host."
            exit 1
        fi

        local sbatch_opts=("--partition=${PARTITION}")
        if [ "$PARTITION" = "gpu" ]; then
            sbatch_opts+=("--gres=gpu:1")
        fi

        echo "Submitting ${script_file} to SLURM partition '${PARTITION}'..."
        sbatch "${sbatch_opts[@]}" "$script_file" "${script_args[@]}"
        exit 0
    fi
}
