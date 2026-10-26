#!/bin/bash
#SBATCH --job-name=plot_evo_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=knl_tacc
#SBATCH --output=logs/plot_%A_%a.out
#SBATCH --error=logs/plot_%A_%a.err

# 1. Clean the environment and load dependencies (matching your simulation script style)
module purge
module load gcc/8.2.0-cytb66h
module load cmake/3.24.2
module load python

# Jump to the directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Since this script lives in proyecto-v2/scripts, the project root is one level up
PROJECT_DIR="$(dirname "$PWD")"

# 2. Activate the python virtual environment
source "${PROJECT_DIR}/myenv/bin/activate"

echo "Starting Plotting Task: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on Node: $HOSTNAME"
echo "Project Directory: $PROJECT_DIR"
echo "=========================================================="

OUTPUT_DIR="${PROJECT_DIR}/output"
PLOT_SCRIPT="${PROJECT_DIR}/python/plot_time_evo.py"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory $OUTPUT_DIR does not exist."
    exit 1
fi

# Automatically find and sort all output subdirectories
mapfile -t dirs < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
total_dirs=${#dirs[@]}

if [ $total_dirs -eq 0 ]; then
    echo "Error: No output directories found in $OUTPUT_DIR."
    exit 1
fi

# Map SLURM_ARRAY_TASK_ID (1-indexed) to array index (0-indexed)
index=$((SLURM_ARRAY_TASK_ID - 1))

if [ $index -ge $total_dirs ]; then
    echo "Error: Task ID $SLURM_ARRAY_TASK_ID exceeds total directories ($total_dirs)."
    exit 1
fi

target_dir="${dirs[$index]}"
BASE_PREFIX="${target_dir}/time_data"

echo "Processing dataset: $(basename "$target_dir")"
echo "Target prefix: $BASE_PREFIX"
echo "--------------------------------------------------"

# Run python script with default settings
python "$PLOT_SCRIPT" "$BASE_PREFIX"

echo "=========================================================="
echo "Finished processing $(basename "$target_dir")"
echo "=========================================================="