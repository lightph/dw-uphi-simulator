#!/bin/bash
#SBATCH --job-name=dw_sweep_ha3
#SBATCH --output=logs/sweep_knl_%A_%a.out
#SBATCH --error=logs/sweep_knl_%A_%a.err
#SBATCH --partition=knl_tacc
#SBATCH --exclude=node122,node124,node139,node140,node142
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --array=0-199

# ==============================================================================
# SLURM Sweep Script for KNL CPU Partition (ha = 3.0, h0 in [0, 8], 200 points)
# Extended wall sizes: L = 8 (N = 16) and L = 1024 (N = 2048), dx = 0.5
# Physics: alpha = 0.27, omega = 1.0, ha = 3.0
# ==============================================================================

# 1. Environment & Modules
module purge 2>/dev/null || true
module load cmake/3.24.2 2>/dev/null || module load cmake 2>/dev/null || true
module load gcc/8.5.0-5sg556d 2>/dev/null || module load gcc 2>/dev/null || true

export CC="${CC:-/share/apps/spack/instalaciones/linux-westmere/gcc-8.5.0-5sg556d/bin/gcc}"
export CXX="${CXX:-/share/apps/spack/instalaciones/linux-westmere/gcc-8.5.0-5sg556d/bin/g++}"

# OpenMP multi-threading settings for KNL CPU backend
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# 2. Directory setup
cd "$SLURM_SUBMIT_DIR"
PROJECT_DIR="${PROJECT_DIR:-$HOME/inv1/proyecto-v3}"

# Auto-detect executable location (cluster build_slurm or local build)
if [ -f "${PROJECT_DIR}/build_slurm/bin/sweep" ]; then
    EXEC="${PROJECT_DIR}/build_slurm/bin/sweep"
elif [ -f "${PROJECT_DIR}/build/bin/sweep" ]; then
    EXEC="${PROJECT_DIR}/build/bin/sweep"
elif [ -f "$SLURM_SUBMIT_DIR/../build/bin/sweep" ]; then
    PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
    EXEC="${PROJECT_DIR}/build/bin/sweep"
else
    EXEC="${PROJECT_DIR}/build_slurm/bin/sweep"
fi

# Ensure executable exists
if [ ! -f "$EXEC" ]; then
    echo "Error: Executable not found at '$EXEC'"
    exit 1
fi

# --- System & Physics Parameters ---
ALPHA=0.27
OMEGA=1.0
DT=0.0196349540849362

# Spatial resolution dx = 0.5
# Size L = 8 -> N = 8 / 0.5 = 16
L_SMALL=8.0
SIZE_SMALL=16

# Size L = 1024 -> N = 1024 / 0.5 = 2048
L_BIG=1024.0
SIZE_BIG=2048

# Simulation Durations
TRANSIENT=131072
# EVAL: Exact multiple of 4*pi/omega (twice driving period, 1000 cycles)
EVAL=$(awk -v omega=$OMEGA -v cycles=1000 'BEGIN { printf "%.10f", cycles * 4.0 * 3.14159265358979323846 / omega }')

# Driving Field Grid Parameters
# ha locked to 3.0
HA_MIN=3.0
HA_MAX=3.0
HA_STEPS=1

# h0 swept from 0.0 to 8.0 over 200 points (indices 0 to 199 -> 199 intervals)
H0_MIN=0.0
H0_MAX=8.0
H0_POINTS=200
H0_STEPS=$((H0_POINTS - 1))

# Calculate h0 for this specific SLURM task
H0_VAL=$(awk -v min=$H0_MIN -v max=$H0_MAX -v steps=$H0_STEPS -v task=$SLURM_ARRAY_TASK_ID \
         'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')

# Output Directory Structure
MASTER_DIR="${PROJECT_DIR}/output/sweep_ha3_L8_L1024_${SLURM_ARRAY_JOB_ID:-manual}"
DIR_SMALL="${MASTER_DIR}/L_8"
DIR_BIG="${MASTER_DIR}/L_1024"

mkdir -p "$DIR_SMALL" "$DIR_BIG" "$SLURM_SUBMIT_DIR/logs"

# Save Run Parameters metadata once (via Task 0)
if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
    PARAMS_FILE="${MASTER_DIR}/run_parameters.txt"
    cat <<EOF > "$PARAMS_FILE"
==========================================================
Run Parameters for Sweep Job: ${SLURM_ARRAY_JOB_ID:-manual}
==========================================================
Timestamp:      $(date)
Executable:     $EXEC
Output Dir:     $MASTER_DIR
Partition:      knl_tacc (CPU)
Threads/Task:   $OMP_NUM_THREADS
----------------------------------------------------------
System Sizes:
  - Small:      L = $L_SMALL, Size (N) = $SIZE_SMALL (dx = 0.5)
  - Big:        L = $L_BIG, Size (N) = $SIZE_BIG (dx = 0.5)
Physics:
  - Alpha:      $ALPHA
  - Omega:      $OMEGA
  - ha:         $HA_MIN (Locked)
  - h0 Range:   [$H0_MIN, $H0_MAX] (Total: $H0_POINTS points, Task: 0..$H0_STEPS)
Numerics:
  - dt:         $DT
  - Transient:  $TRANSIENT
  - Eval:       $EVAL
==========================================================
EOF
fi

OUT_FILE_SMALL="${DIR_SMALL}/results_h0_${H0_VAL}.txt"
OUT_FILE_BIG="${DIR_BIG}/results_h0_${H0_VAL}.txt"

echo "=========================================================="
echo "Sweep Task ${SLURM_ARRAY_TASK_ID:-0} on Node: $(hostname)"
echo "Backend: CPU (partition knl_tacc, OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo "H0 = $H0_VAL | HA = $HA_MIN"
echo "Eval Time = $EVAL"
echo "Outputs -> $MASTER_DIR"
echo "=========================================================="

# 1. Run small system size (L = 8, N = 16)
echo "[1/2] Simulating L = $L_SMALL (N = $SIZE_SMALL) on CPU..."
time $EXEC $SIZE_SMALL $L_SMALL $ALPHA $OMEGA $DT $TRANSIENT $EVAL $H0_VAL $HA_MIN $HA_MAX $HA_STEPS "$OUT_FILE_SMALL" cpu

# 2. Run large system size (L = 1024, N = 2048)
echo "[2/2] Simulating L = $L_BIG (N = $SIZE_BIG) on CPU..."
time $EXEC $SIZE_BIG $L_BIG $ALPHA $OMEGA $DT $TRANSIENT $EVAL $H0_VAL $HA_MIN $HA_MAX $HA_STEPS "$OUT_FILE_BIG" cpu

echo "=========================================================="
echo "Task ${SLURM_ARRAY_TASK_ID:-0} complete for both sizes."
echo "=========================================================="