#!/bin/bash
#SBATCH --job-name=sweep_c3
#SBATCH --output=logs/sweep_c3_%A_%a.out
#SBATCH --error=logs/sweep_c3_%A_%a.err
#SBATCH --array=0-200%1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ==============================================================================
# Phase Diagram Combo 3: alpha = 0.27, omega = 2.0 (Double Frequency Drive)
# Strictly limited to 1 GPU node at a time via --array=0-200%1
# ==============================================================================

set -eo pipefail

module purge 2>/dev/null || true
module load cmake/3.24.2 2>/dev/null || true
module load cuda/11.1.0-zj6lnzj 2>/dev/null || module load cuda 2>/dev/null || true

export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:$LD_LIBRARY_PATH"
export CC="/share/apps/spack/instalaciones/linux-westmere/gcc-8.5.0-5sg556d/bin/gcc"
export CXX="/share/apps/spack/instalaciones/linux-westmere/gcc-8.5.0-5sg556d/bin/g++"

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/inv1/proyecto-v3}"
cd "$PROJECT_DIR"

EXEC="${PROJECT_DIR}/build_slurm/bin/sweep"
if [ ! -x "$EXEC" ]; then
    EXEC=$(find "${PROJECT_DIR}" -name sweep -type f -perm /111 | head -n 1)
fi

# Physics Parameters
SIZE=512
L=256.0
ALPHA="0.27"
OMEGA="2.0"

DT=$(awk -v omega=$OMEGA 'BEGIN { printf "%.16f", 2.0 * 3.14159265358979323846 / (320.0 * omega) }')
TRANSIENT=131072
EVAL=$(awk -v omega=$OMEGA -v cycles=1000 'BEGIN { printf "%.10f", cycles * 4.0 * 3.14159265358979323846 / omega }')

H0_MIN=0.0
H0_MAX=10.0
H0_STEPS=200

HA_MIN=0.0
HA_MAX=20.0
HA_STEPS=401

H0_VAL=$(awk -v min=$H0_MIN -v max=$H0_MAX -v steps=$H0_STEPS -v task=$SLURM_ARRAY_TASK_ID \
         'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')

MASTER_DIR="${PROJECT_DIR}/output/sweep_phase_diagram_alpha_${ALPHA}_omega_${OMEGA}"
mkdir -p "$MASTER_DIR" "${PROJECT_DIR}/logs"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    PARAMS_FILE="${MASTER_DIR}/run_parameters.txt"
    cat <<EOF > "$PARAMS_FILE"
==========================================================
Run Parameters for Phase Diagram Combo 3: alpha=$ALPHA, omega=$OMEGA
SLURM Job ID:   $SLURM_ARRAY_JOB_ID
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
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo "GPU")

echo "=========================================================="
echo "Sweep Combo 3 [Task $SLURM_ARRAY_TASK_ID] Node: $(hostname)"
echo "GPU: $GPU_NAME"
echo "Parameters: ALPHA=$ALPHA, OMEGA=$OMEGA"
echo "H0 = $H0_VAL | HA range = [$HA_MIN, $HA_MAX] ($HA_STEPS steps)"
echo "Eval Time = $EVAL | dt = $DT"
echo "Output -> $OUTPUT_FILE"
echo "=========================================================="

time "$EXEC" "$SIZE" "$L" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA_MIN" "$HA_MAX" "$HA_STEPS" "$OUTPUT_FILE" "gpu"

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
