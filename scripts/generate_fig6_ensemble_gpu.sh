#!/bin/bash
#SBATCH --job-name=fig6_ensemble_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/fig6_ens_%A_%a.out
#SBATCH --error=logs/fig6_ens_%A_%a.err

# ==============================================================================
# generate_fig6_ensemble_gpu.sh
#
# Runs an ensemble of 10 independent realizations (seeds) on the GPU partition
# for Figure 6 (Edwards-Wilkinson Universal Scaling Collapse of the NR Phase).
#
# When submitted via SLURM:
#   Dispatches 10 parallel GPU tasks (--array=1-10).
#   Each task computes a distinct seed: SEED = SEED_BASE + SLURM_ARRAY_TASK_ID
#
# When run locally:
#   Loops sequentially over 1..NUM_SAMPLES.
#
# Parameters:
#   h0 = 5.0, ha = 3.0, alpha = 0.27, omega = 1.0, dt = 2*pi/(128*omega)
#   L = 8192.0, SIZE = 16384 (dx = 0.5), eps = 0.0005
# ==============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/scripts/common_env.sh" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
    elif [ -f "${SLURM_SUBMIT_DIR}/common_env.sh" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common_env.sh"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
SEED_BASE="${SEED_BASE:-100}"

parse_common_args "$@"

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --samples <int>          Number of ensemble seeds (default: 10)"
    echo "  --seed-base <int>        Base offset for seeds (default: 100)"
    echo "  --backend <cpu|gpu>      Compute backend (default: gpu)"
    echo "  --partition <gpu|knl>    SLURM partition (default: gpu)"
    echo "  --slurm                  Submit job array via sbatch instead of running locally"
    echo "  --size <int>             Grid points (default: 16384)"
    echo "  --L <float>              Domain length L/L0 (default: 8192.0)"
    echo "  --max-power <int>        Max power of two steps (default: 26 -> t ~ 3.3e6)"
    echo "  --eps <float>            Initial noise perturbation (default: 0.0005)"
    echo "  --out-dir <path>         Ensemble base output directory"
    exit 0
fi

# Set standard physics & grid parameters
H0="${H0:-5.0}"
HA="${HA:-3.0}"
DT="${DT:-0.049087385212340519}" # 2*pi / (128*omega)
SIZE="${SIZE:-16384}"
L="${L:-8192.0}"
MAX_POWER="${MAX_POWER:-26}"
EPS="${EPS:-0.0005}"
BACKEND="${BACKEND:-gpu}"
PARTITION="${PARTITION:-gpu}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/fig6_ensemble_gpu}"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"

# Dispatch via sbatch if requested outside SLURM
if [ "$SUBMIT_SLURM" = true ] && [ "$IS_SLURM_JOB" = false ]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "Error: sbatch not found."
        exit 1
    fi
    echo "Submitting 10-sample ensemble to SLURM GPU partition..."
    sbatch --partition="${PARTITION}" --gres=gpu:1 --array="1-${NUM_SAMPLES}" "$0" "$@"
    exit 0
fi

setup_cluster_modules

EXEC=$(find_executable "time_evo" "$BACKEND")
if [ -z "$EXEC" ]; then
    echo "Error: 'time_evo' executable not found."
    exit 1
fi

run_single_seed() {
    local task_num="$1"
    local seed_val=$((SEED_BASE + task_num))
    local seed_str=$(printf "%03d" "$task_num")
    local seed_dir="${OUT_DIR}/seed_${seed_str}"
    mkdir -p "$seed_dir"
    local prefix="${seed_dir}/time_data"

    echo "=========================================================="
    echo "Task #${task_num}: Running Sample SEED=${seed_val} (Sample ${task_num}/${NUM_SAMPLES})"
    echo "Backend: $BACKEND | Grid: SIZE=$SIZE, L=$L, dx=$(awk "BEGIN {print $L/$SIZE}")"
    echo "Physics: H0=$H0, HA=$HA, OMEGA=$OMEGA, ALPHA=$ALPHA"
    echo "Timestep: dt=$DT | max_power=$MAX_POWER | EPS=$EPS"
    echo "Saving outputs to: $seed_dir"
    echo "=========================================================="

    time "$EXEC" "$SIZE" "$L" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" "$MAX_POWER" "$prefix" "$EPS" "$seed_val" "$BACKEND"

    cat <<EOF > "${seed_dir}/parameters.txt"
Simulation: time_evo NC Edwards-Wilkinson (Ensemble Sample)
Sample_Index: $task_num
Date: $(date -Iseconds)
Backend: $BACKEND
SIZE: $SIZE
L: $L
ALPHA: $ALPHA
H0: $H0
HA: $HA
OMEGA: $OMEGA
DT: $DT
MAX_POWER: $MAX_POWER
EPS: $EPS
SEED: $seed_val
PREFIX: $prefix
EOF
    echo "Sample ${task_num} completed."
}

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # SLURM array execution: each node/GPU processes one sample in parallel
    run_single_seed "$SLURM_ARRAY_TASK_ID"
else
    # Sequential local execution across all requested samples
    echo "Running ${NUM_SAMPLES} samples sequentially on ${BACKEND}..."
    for i in $(seq 1 "$NUM_SAMPLES"); do
        run_single_seed "$i"
    done

    # Run ensemble averaging
    if [ -f "${PROJECT_DIR}/python/average_nc_ew_ensemble.py" ]; then
        echo "Averaging ensemble results..."
        python3 "${PROJECT_DIR}/python/average_nc_ew_ensemble.py" "$OUT_DIR"
    fi
fi
