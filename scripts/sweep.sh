#!/bin/bash
#SBATCH --job-name=dw_sweep_full
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-100
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# 1. Load environment
module purge
module load cmake/3.24.2
module load cuda/11.1.0-zj6lnzj

export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:$LD_LIBRARY_PATH"
export CC="/share/apps/spack/instalaciones/linux-westmere/gcc-8.5.0-5sg556d/bin/gcc"
export CXX="/share/apps/spack/instalaciones/linux-westmere/gcc-8.5.0-5sg556d/bin/g++"

cd "$SLURM_SUBMIT_DIR"
PROJECT_DIR="$HOME/inv1/proyecto-v3"
EXEC="${PROJECT_DIR}/build_slurm/bin/sweep"

# --- System & Physics Parameters ---
SIZE=512
L=256.0
ALPHA=0.27
OMEGA=1.0
DT=0.196349540849362

# Simulation Durations
TRANSIENT=131072
# Define EVAL as an exact multiple of 4*pi/omega (twice the driving period)
EVAL=$(awk -v omega=$OMEGA -v cycles=1000 'BEGIN { printf "%.10f", cycles * 4.0 * 3.14159265358979323846 / omega }')

# Define h0 grid parameters (0 to 10, dh0=0.1 -> 100 intervals, array goes 0 to 100)
H0_MIN=0.0
H0_MAX=10.0
H0_STEPS=100 

# Define ha grid parameters (0 to 20, dh=0.1 -> 200 intervals + 1 endpoint = 201 points)
HA_MIN=0.0
HA_MAX=20.0
HA_STEPS=201  

# Calculate h0 for this specific SLURM task using awk
H0_VAL=$(awk -v min=$H0_MIN -v max=$H0_MAX -v steps=$H0_STEPS -v task=$SLURM_ARRAY_TASK_ID \
         'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')

# --- Tidy Output & Log Directory Management ---
MASTER_DIR="${PROJECT_DIR}/output/sweep_full_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$MASTER_DIR" "$SLURM_SUBMIT_DIR/logs"

# --- Save Run Parameters (Only once, via Task 0) ---
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    PARAMS_FILE="${MASTER_DIR}/run_parameters.txt"
    cat <<EOF > "$PARAMS_FILE"
==========================================================
Run Parameters for Sweep Job: $SLURM_ARRAY_JOB_ID
==========================================================
Timestamp:      $(date)
Executable:     $EXEC
Output Dir:     $MASTER_DIR
----------------------------------------------------------
System Size:    $SIZE
Length (L):     $L
Alpha:          $ALPHA
Omega:          $OMEGA
dt:             $DT
Transient Time: $TRANSIENT
Eval Time:      $EVAL
h0 Range:       [$H0_MIN, $H0_MAX] with $H0_STEPS intervals
ha Range:       [$HA_MIN, $HA_MAX] with $HA_STEPS intervals
==========================================================
EOF
fi

OUTPUT_FILE="${MASTER_DIR}/results_h0_${H0_VAL}.txt"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

echo "=========================================================="
echo "Sweep Task $SLURM_ARRAY_TASK_ID starting on Node: $(hostname)"
echo "Assigned GPU: $GPU_NAME"
echo "H0 = $H0_VAL | HA range = [$HA_MIN, $HA_MAX] with $HA_STEPS steps"
echo "Eval Time = $EVAL (Aligned to 4*pi/omega)"
echo "Outputs -> $MASTER_DIR"
echo "=========================================================="

time $EXEC $SIZE $L $ALPHA $OMEGA $DT $TRANSIENT $EVAL $H0_VAL $HA_MIN $HA_MAX $HA_STEPS $OUTPUT_FILE

echo "Task $SLURM_ARRAY_TASK_ID complete."
