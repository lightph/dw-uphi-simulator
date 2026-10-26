#!/bin/bash
#SBATCH --job-name=time_evo_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
#SBATCH --output=time_evo_%A_%a.out
#SBATCH --error=time_evo_%A_%a.err

# 1. Clean the environment and load dependencies
module purge
module load gcc/8.2.0-cytb66h
module load cuda/11.1.0-zj6lnzj
module load cmake/3.24.2
module load mpfr/3.1.6-ubdztn6 gmp/6.3.0-blw24vo mpc/1.1.0-44wbkc3

# Jump to the directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Define project directory
PROJECT_DIR="$HOME/inv1/proyecto-v2"

# Explicitly export CUDA and local FFTW library paths for runtime execution
export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:${PROJECT_DIR}/fftw_local/lib:$LD_LIBRARY_PATH"

echo "Starting Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on Node: $HOSTNAME"
echo "--- GPU Hardware Info ---"
nvidia-smi -L
echo "-------------------------"

# Path to the new pure-GPU executable
EXEC="${PROJECT_DIR}/build_slurm/bin/time_evo"

# --- Shared System Parameters ---
SIZE=65536          # 2^17
L=8192.0
ALPHA=0.27
OMEGA=1.0
DT=0.049087385212340519
MAX_POWER=30        

# --- Shared Hardware and Output Parameters ---
EPS=0.05
SEED=42
PRECISION="double"

# --- Varying Parameters based on Array Task ID ---
case $SLURM_ARRAY_TASK_ID in
    1)
        H0=4
        HA=0.0
        LABEL="DC"
        ;;
    2)
        H0=4.0
        HA=3.0
        LABEL="SC"
        ;;
    3)
        H0=5.0
        HA=3.0
        LABEL="NC"
        ;;
    *)
        echo "Error: Invalid task ID"
        exit 1
        ;;
esac

# Base output directory including the specific label (DC, SC, or NC)
BASE_OUT_DIR="${PROJECT_DIR}/output/time_evo_${LABEL}_${L}_${H0}_${HA}_${OMEGA}"
OUT_DIR="$BASE_OUT_DIR"

# Incremental copy suffix logic
count=1
while [ -d "$OUT_DIR" ]; do
    OUT_DIR="${BASE_OUT_DIR}(${count})"
    ((count++))
done

# Create the unique output directory
mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/time_data"

echo "=========================================================="
echo "Running configuration: $LABEL (Array Task $SLURM_ARRAY_TASK_ID)"
echo "Physics: H0=$H0 | HA=$HA | OMEGA=$OMEGA | L=$L | ALPHA=$ALPHA"
echo "Num    : dt=$DT | max_power=$MAX_POWER | size=$SIZE"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

# Execute (Note: $BACKEND argument is removed because the new binary is pure GPU)
time $EXEC $SIZE $L $ALPHA $H0 $HA $OMEGA $DT $MAX_POWER $PREFIX $EPS $SEED $PRECISION

echo "Time evolution complete for $LABEL."