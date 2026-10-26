#!/bin/bash
#SBATCH --job-name=plot_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=knl_tacc
#SBATCH --output=logs/plot_scaling_%j.out
#SBATCH --error=logs/plot_scaling_%j.err

# Clean the environment and load dependencies
module purge
module load gcc/8.2.0-cytb66h
module load cmake/3.24.2

# Jump to the directory where the script was submitted from
cd "$SLURM_SUBMIT_DIR"

# Ensure the logs directory exists so SLURM doesn't fail silently
mkdir -p logs

# Assuming this script lives in proyecto-v2/scripts, the project root is one level up
PROJECT_DIR="$(dirname "$PWD")"

# Activate the python virtual environment
source "${PROJECT_DIR}/myenv/bin/activate"

echo "Starting Scaling Plotting Task: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on Node: $HOSTNAME"
echo "Project Directory: $PROJECT_DIR"
echo "=========================================================="

# Define the base scaling directory and the plotting script
SCALING_DIR="${PROJECT_DIR}/output/scaling"
PLOT_SCRIPT="${PROJECT_DIR}/python/plot_scaling.py"

if [ ! -d "$SCALING_DIR" ]; then
    echo "Error: Scaling directory $SCALING_DIR does not exist."
    exit 1
fi

echo "Processing scaling datasets in: $SCALING_DIR"
echo "--------------------------------------------------"

# Run the Python script
python "$PLOT_SCRIPT" "$SCALING_DIR"

echo "=========================================================="
echo "Finished plotting scaling analysis."
echo "=========================================================="