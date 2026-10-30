#!/bin/bash
#SBATCH --job-name=gen_fig6_data
#SBATCH --output=logs/fig6_%j.out
#SBATCH --error=logs/fig6_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

# ==============================================================================
# generate_fig6_data.sh - Regenerate Simulation Data for Figure 6
# Universal 1D Edwards-Wilkinson Scaling Collapse of the NR Phase
# Parameters: h0 = 5.0, ha = 3.0, alpha = 0.27, omega = 1.0
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
    echo "  --size <int>             Grid points (default: 2048 for local, 65536 for cluster)"
    echo "  --L <float>              Domain length L/L0 (default: 1024.0 for local, 8192.0 for cluster)"
    echo "  --max-power <int>        Max power of two steps 2^p (default: 24 for local, 28 for cluster)"
    echo "  --eps <float>            Initial noise perturbation (default: 0.0005)"
    echo "  --seed <int>             Random seed (default: 42)"
    echo "  --out-dir <path>         Output directory (default: output/reference_nc_ew_data_recomputed)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

# Standardized Physical & Grid Defaults
H0="${H0:-5.0}"
HA="${HA:-3.0}"
DT="${DT:-0.049087385212340519}" # 2*pi / (128*omega)

if [ "$IS_SLURM_JOB" = true ] || [ "$BACKEND" = "gpu" ]; then
    # Full high-resolution parameters
    SIZE="${SIZE:-65536}"
    L="${L:-8192.0}"
    MAX_POWER="${MAX_POWER:-28}"
else
    # Standard local parameters
    SIZE="${SIZE:-2048}"
    L="${L:-1024.0}"
    MAX_POWER="${MAX_POWER:-24}"
fi

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/reference_nc_ew_data_recomputed}"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"
PREFIX="${OUT_DIR}/time_data"

EXEC=$(find_executable "time_evo" "$BACKEND")
if [ -z "$EXEC" ]; then
    echo "Error: 'time_evo' executable not found."
    exit 1
fi

echo "=========================================================="
echo "Regenerating Data for Figure 6 (Edwards-Wilkinson Scaling Collapse)"
echo "Backend: $BACKEND | Partition: $PARTITION | Executable: $EXEC"
echo "Grid: SIZE=$SIZE | L=$L (dx=$(awk "BEGIN {print $L/$SIZE}"))"
echo "Physics: H0=$H0 | HA=$HA | OMEGA=$OMEGA | ALPHA=$ALPHA"
echo "Timestep: dt=$DT | max_power=$MAX_POWER (total steps: 2^$MAX_POWER)"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

time "$EXEC" "$SIZE" "$L" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" "$MAX_POWER" "$PREFIX" "$EPS" "$SEED" "$BACKEND"

cat <<EOF > "${OUT_DIR}/parameters.txt"
Simulation: time_evo NC Edwards-Wilkinson
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
SEED: $SEED
PREFIX: $PREFIX
EOF

echo "Done! Figure 6 data generated in $OUT_DIR."

