#!/bin/bash
#SBATCH --job-name=time_evo_NC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/time_evo_NC_%j.out
#SBATCH --error=logs/time_evo_NC_%j.err

# ==============================================================================
# Simulation Data Generation for Figure 6 (NC Phase Edwards-Wilkinson Scaling)
# Physical parameters:
#   h0 = 5.0, ha = 3.0, omega = 1.0, alpha = 0.27
# System size:
#   size = 65536 (2^16 points), L = 8192.0 (dx = 0.125)
# Time integration:
#   dt = 0.049087385212340519 (2*pi / (128*omega))
#   max_power = 28 (2^28 steps => t_max ~ 1.31e7)
# ==============================================================================

set -eo pipefail

# 1. Clean environment and load cluster modules
module purge 2>/dev/null || true
module load gcc/8.2.0-cytb66h 2>/dev/null || true
module load cuda/11.1.0-zj6lnzj 2>/dev/null || true
module load cmake/3.24.2 2>/dev/null || true
module load mpfr/3.1.6-ubdztn6 gmp/6.3.0-blw24vo mpc/1.1.0-44wbkc3 2>/dev/null || true

# Jump to submit directory or project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

# Determine project directory dynamically
if [ -d "$PWD/build_slurm" ] || [ -d "$PWD/build" ]; then
    PROJECT_DIR="$PWD"
elif [ -d "$HOME/inv1/proyecto-v3" ]; then
    PROJECT_DIR="$HOME/inv1/proyecto-v3"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Locate executable
if [ -x "${PROJECT_DIR}/build_slurm/bin/time_evo" ]; then
    EXEC="${PROJECT_DIR}/build_slurm/bin/time_evo"
elif [ -x "${PROJECT_DIR}/build/bin/time_evo" ]; then
    EXEC="${PROJECT_DIR}/build/bin/time_evo"
else
    echo "Error: Executable time_evo not found in ${PROJECT_DIR}/build_slurm/bin or ${PROJECT_DIR}/build/bin"
    exit 1
fi

# Export CUDA and local libraries if present
export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:${PROJECT_DIR}/fftw_local/lib:$LD_LIBRARY_PATH"

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================================="
echo "Starting Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on Node: $(hostname)"
echo "Project Directory: $PROJECT_DIR"
echo "Executable: $EXEC"
echo "--- GPU Hardware Info ---"
nvidia-smi -L || true
echo "=========================================================="

# --- System & Physics Parameters for NC Phase ---
SIZE=65536                  # Grid points: 2^16
L=8192.0                    # Domain length: L/L0 = 8192 (dx = 0.125)
ALPHA=0.27                  # Gilbert damping
H0=5.0                      # DC field component
HA=3.0                      # AC field amplitude
OMEGA=1.0                   # AC angular frequency
DT=0.049087385212340519     # Timestep (T_drive / 128)
MAX_POWER=28                # Total steps = 2^28 (~2.68e8 steps, t_max ~ 1.31e7)
EPS=0.05                    # Initial condition noise amplitude
SEED=42                     # Random seed
BACKEND="gpu"

# Output directory naming convention
BASE_OUT_DIR="${PROJECT_DIR}/output/time_evo_NC_${L}_${H0}_${HA}_${OMEGA}"
OUT_DIR="$BASE_OUT_DIR"

count=1
while [ -d "$OUT_DIR" ]; do
    OUT_DIR="${BASE_OUT_DIR}(${count})"
    ((count++))
done

mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/time_data"

# Save parameters metadata file for provenance
cat <<EOF > "${OUT_DIR}/parameters.txt"
Simulation: time_evo NC Edwards-Wilkinson
Date: $(date -Iseconds)
SLURM_JOB_ID: ${SLURM_JOB_ID:-local}
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
BACKEND: $BACKEND
PREFIX: $PREFIX
EOF

echo "Physics Parameters: H0=$H0 | HA=$HA | OMEGA=$OMEGA | ALPHA=$ALPHA"
echo "Grid Parameters   : SIZE=$SIZE | L=$L | dx=$(echo "scale=4; $L/$SIZE" | bc -l 2>/dev/null || echo "0.125")"
echo "Time Parameters   : dt=$DT | max_power=$MAX_POWER (t_max ~ 1.3e7)"
echo "Output Directory  : $OUT_DIR"
echo "=========================================================="

# Launch simulation
time "$EXEC" "$SIZE" "$L" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" "$MAX_POWER" "$PREFIX" "$EPS" "$SEED" "$BACKEND"

echo "=========================================================="
echo "Simulation completed successfully!"
echo "Data written to: $OUT_DIR"
echo "You can now run: python python/plot_figure6_nc_ew.py \"$OUT_DIR\""
echo "=========================================================="

