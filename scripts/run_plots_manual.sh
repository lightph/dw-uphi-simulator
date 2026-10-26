#!/bin/bash
#SBATCH --job-name=plot_manual_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=knl_tacc
#SBATCH --output=logs/plot_manual_%A_%a.out
#SBATCH --error=logs/plot_manual_%A_%a.err

# 1. Clean the environment and load dependencies
module purge
module load gcc/8.2.0-cytb66h
module load cmake/3.24.2

# Jump to the directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Since this script lives in proyecto-v3/scripts, the project root is one level up
PROJECT_DIR="$(dirname "$PWD")"

# 2. Activate the python virtual environment
source "${PROJECT_DIR}/myenv/bin/activate"

echo "Starting Manual Plotting Task: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on Node: $HOSTNAME"
echo "Project Directory: $PROJECT_DIR"
echo "=========================================================="

OUTPUT_DIR="${PROJECT_DIR}/output"
PLOT_SCRIPT="${PROJECT_DIR}/python/plot_time_evo.py"

# --- MANUALLY DEFINE YOUR DATASET FOLDERS HERE ---
datasets=(
    "time_evo_DC_1024.0_4_0.0_1.0"
    "time_evo_SC_1024.0_4.0_3.0_1.0"
    "time_evo_NC_1024.0_5.0_4.0_1.0"
)

total_dirs=${#datasets[@]}

if [ $total_dirs -eq 0 ]; then
    echo "Error: No datasets specified in the manual list."
    exit 1
fi

# Map SLURM_ARRAY_TASK_ID (1-indexed) to array index (0-indexed)
index=$((SLURM_ARRAY_TASK_ID - 1))

if [ $index -ge $total_dirs ]; then
    echo "Error: Task ID $SLURM_ARRAY_TASK_ID exceeds total defined datasets ($total_dirs)."
    exit 1
fi

target_folder="${datasets[$index]}"
target_dir="${OUTPUT_DIR}/${target_folder}"
BASE_PREFIX="${target_dir}/time_data"

if [ ! -d "$target_dir" ]; then
    echo "Error: Target directory $target_dir does not exist."
    exit 1
fi

echo "Processing dataset: $target_folder"
echo "Target prefix: $BASE_PREFIX"
echo "--------------------------------------------------"

# Run python script with default settings
python "$PLOT_SCRIPT" "$BASE_PREFIX"

echo "=========================================================="
echo "Finished processing $target_folder"
echo "=========================================================="
