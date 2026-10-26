#!/bin/bash
#SBATCH --job-name=plot_main_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --partition=knl_tacc
#SBATCH --array=1-2
#SBATCH --output=logs/plot_main_%A_%a.out
#SBATCH --error=logs/plot_main_%A_%a.err

# Clean the environment and load dependencies
module purge
module load gcc/8.2.0-cytb66h
module load cmake/3.24.2

# Jump to the directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Ensure the logs directory exists so SLURM doesn't fail silently
mkdir -p logs

# Since this script lives in proyecto-v2/scripts, the project root is one level up
PROJECT_DIR="$(dirname "$PWD")"

# Activate the python virtual environment
source "${PROJECT_DIR}/myenv/bin/activate"

echo "Starting Manual Plotting Task: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on Node: $HOSTNAME"
echo "Project Directory: $PROJECT_DIR"
echo "=========================================================="

OUTPUT_DIR="${PROJECT_DIR}/output"
PLOT_SCRIPT="${PROJECT_DIR}/python/plot_main.py"

# Datasets updated to match the array tasks from the simulation script
datasets=(
    "main_NC_8192.0_5.0_3.0_1.0"
    "main_NC_8192.0_10.0_15.0_1.0"
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

if [ ! -d "$target_dir" ]; then
    echo "Error: Target directory $target_dir does not exist."
    exit 1
fi

echo "Processing dataset: $target_folder"
echo "Target directory: $target_dir"
echo "--------------------------------------------------"

# Run python script with the target directory. Saving figures inside the specific dataset folder.
python "$PLOT_SCRIPT" "$target_dir" --output_dir "${target_dir}/figures"

echo "=========================================================="
echo "Finished processing $target_folder"
echo "=========================================================="