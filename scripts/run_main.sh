#!/bin/bash
#SBATCH --job-name=domain_wall_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-2
#SBATCH --output=main_%A_%a.out
#SBATCH --error=main_%A_%a.err

# 1. Clean the environment and load dependencies
module purge
module load gcc/8.2.0-cytb66h
module load cuda/11.1.0-zj6lnzj
module load cmake/3.24.2
module load mpfr/3.1.6-ubdztn6 gmp/6.3.0-blw24vo mpc/1.1.0-44wbkc3

# Jump to the directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Define project directory
PROJECT_DIR="$HOME/inv1/proyecto-v3"

# Explicitly export CUDA and local FFTW library paths for runtime execution
export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:${PROJECT_DIR}/fftw_local/lib:$LD_LIBRARY_PATH"

echo "Starting Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on Node: $HOSTNAME"
echo "--- GPU Hardware Info ---"
nvidia-smi -L
echo "-------------------------"

# Path to the new pure-GPU executable
EXEC="${PROJECT_DIR}/build_slurm/bin/main"

# --- Shared System Parameters ---
SIZE=16384
BATCH=1
L=8192.0
ALPHA=0.27
OMEGA=1.0
DT=0.049087385212340519

# --- New Temporal Parameters ---
TRANSIENT_TIME=2000000
AVERAGING_TIME=1647099.32916528550563020
WINDOW_TIME=1000
SUBSAMPLE_RATE=16

# --- Shared Hardware and Output Parameters ---
EPS=0.05
SEED=42
BACKEND="gpu"
LABEL="NC"

# --- Varying Parameters based on Array Task ID ---
case $SLURM_ARRAY_TASK_ID in
    1)
        H0=5.0
        HA=3.0
        ;;
    2)
        H0=10.0
        HA=15.0
        ;;
    *)
        echo "Error: Invalid task ID"
        exit 1
        ;;
esac

# Base output directory including the specific label
BASE_OUT_DIR="${PROJECT_DIR}/output/main_${LABEL}_${L}_${H0}_${HA}_${OMEGA}"
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
PARAM_FILE="${OUT_DIR}/parameters.txt"

# Save parameters to a text file
cat <<EOF > "$PARAM_FILE"
Job Name: $SLURM_JOB_NAME
Job ID: $SLURM_JOB_ID
Array Task: $SLURM_ARRAY_TASK_ID
Label: $LABEL
Size: $SIZE
Batch: $BATCH
L: $L
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

echo "=========================================================="
echo "Running configuration: $LABEL (Task $SLURM_ARRAY_TASK_ID)"
echo "Physics: H0=$H0 | HA=$HA | OMEGA=$OMEGA | L=$L | ALPHA=$ALPHA"
echo "Time   : transient=$TRANSIENT_TIME | avg=$AVERAGING_TIME | window=$WINDOW_TIME | subs=$SUBSAMPLE_RATE | dt=$DT"
echo "Grid   : size=$SIZE | batch=$BATCH"
echo "Saving outputs to: $OUT_DIR"
echo "Parameters saved to: $PARAM_FILE"
echo "=========================================================="

# Execute
time $EXEC $SIZE $BATCH $L $ALPHA $H0 $HA $OMEGA $DT $TRANSIENT_TIME $AVERAGING_TIME $WINDOW_TIME $SUBSAMPLE_RATE "$PREFIX" $EPS $SEED $BACKEND

echo "Time evolution complete for $LABEL (Task $SLURM_ARRAY_TASK_ID)."
