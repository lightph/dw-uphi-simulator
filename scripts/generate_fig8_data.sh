#!/bin/bash
#SBATCH --job-name=gen_fig8_data
#SBATCH --output=logs/fig8_%A_%a.out
#SBATCH --error=logs/fig8_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

# ==============================================================================
# generate_fig8_data.sh - Regenerate Simulation Data for Figure 8
# Finite-size scaling analysis in the SP phase: roughness and peak mode vs L/L0
# System sizes: L = 2^exp, Size N = 2^(exp+1) (dx = 0.5) for exp in [0, 18]
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

parse_common_args "$@"

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --backend <cpu|gpu>      Compute backend (default: auto-detect)"
    echo "  --partition <gpu|knl_tacc> SLURM partition (default: gpu for GPU, knl_tacc for CPU)"
    echo "  --slurm                  Submit job via sbatch instead of running locally"
    echo "  --exp-min <int>          Minimum power of two exponent for L (default: 0 -> L=1)"
    echo "  --exp-max <int>          Maximum power of two exponent for L (default: 18 -> L=262144)"
    echo "  --h0 <float>             DC magnetic field (default: 4.0)"
    echo "  --ha <float>             AC magnetic field amplitude (default: 3.0)"
    echo "  --eps <float>            Perturbation scale (default: 0.0005)"
    echo "  --seed <int>             Random seed (default: 42)"
    echo "  --out-dir <path>         Output directory (default: output/scaling_data_recomputed)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

# Standardized Physical Defaults
H0="${H0:-4.0}"
HA="${HA:-3.0}"
BATCH="${BATCH:-1}"
LABEL="${LABEL:-SC_scaling}"

DT="${DT:-0.049087385212340519}"
TRANSIENT_TIME="${TRANSIENT_TIME:-102943.7080728303448}"
AVERAGING_TIME="${AVERAGING_TIME:-102943.7080728303448}"
WINDOW_TIME="${WINDOW_TIME:-1000.0}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-16}"

EXP_MIN="${EXP_MIN:-0}"
EXP_MAX="${EXP_MAX:-18}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/scaling_data_recomputed}"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"

EXEC=$(find_executable "main" "$BACKEND")
if [ -z "$EXEC" ]; then
    echo "Error: 'main' executable not found."
    exit 1
fi

echo "=========================================================="
echo "Regenerating Data for Figure 8 (SP Phase Finite-Size Scaling)"
echo "Backend: $BACKEND | Partition: $PARTITION | Executable: $EXEC"
echo "Physics: H0=$H0 | HA=$HA | OMEGA=$OMEGA | ALPHA=$ALPHA"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

run_single_size() {
    local exp="$1"
    local l_int=$((1 << exp))
    local l_val="${l_int}.0"
    local size_val=$((1 << (exp + 1)))

    local sub_dir="${OUT_DIR}/main_${LABEL}_${l_val}_${H0}_${HA}_${OMEGA}"
    mkdir -p "$sub_dir"
    local prefix="${sub_dir}/time_data"

    echo "Running L=$l_val (SIZE=$size_val, dx=0.5)..."
    "$EXEC" "$size_val" "$BATCH" "$l_val" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" \
           "$TRANSIENT_TIME" "$AVERAGING_TIME" "$WINDOW_TIME" "$SUBSAMPLE_RATE" \
           "$prefix" "$EPS" "$SEED" "$BACKEND"

    cat <<EOF > "${sub_dir}/parameters.txt"
Job: Figure 8 Scaling
Size: $size_val
Batch: $BATCH
L: $l_val
Alpha: $ALPHA
H0: $H0
Ha: $HA
Omega: $OMEGA
dt: $DT
Transient Time: $TRANSIENT_TIME
Averaging Time: $AVERAGING_TIME
Window Time: $WINDOW_TIME
Subsample Rate: $SUBSAMPLE_RATE
Eps: $EPS
Seed: $SEED
Backend: $BACKEND
EOF
}

# Check if running as SLURM array task
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Running SLURM Array Task $SLURM_ARRAY_TASK_ID..."
    run_single_size "$SLURM_ARRAY_TASK_ID"
else
    # Local execution: Loop over powers of 2
    echo "Running system size scaling from 2^$EXP_MIN to 2^$EXP_MAX..."
    for ((e=EXP_MIN; e<=EXP_MAX; e++)); do
        run_single_size "$e"
    done
fi

echo "Done! Figure 8 data generated in $OUT_DIR."

