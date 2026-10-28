#!/bin/bash
#SBATCH --job-name=plot_fig6_nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=knl_tacc
#SBATCH --output=logs/plot_fig6_%j.out
#SBATCH --error=logs/plot_fig6_%j.err

# ==============================================================================
# SLURM Plotting Script for Figure 6 (NC Phase Universal EW Scaling)
# Generates publication-ready PDF:
#   paper/figuras/NC_EW.pdf (with scaling collapse inset)
#   paper/figuras/NC_EW_side_by_side.pdf
# ==============================================================================

set -eo pipefail

# 1. Clean environment and load modules
module purge 2>/dev/null || true
module load gcc/8.2.0-cytb66h 2>/dev/null || true
module load cmake/3.24.2 2>/dev/null || true
module load python 2>/dev/null || true

# Jump to submit directory or determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

if [ -d "$PWD/python" ]; then
    PROJECT_DIR="$PWD"
elif [ -d "$HOME/inv1/proyecto-v3" ]; then
    PROJECT_DIR="$HOME/inv1/proyecto-v3"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$PROJECT_DIR"

# 2. Activate Python environment
if [ -f "${PROJECT_DIR}/myenv/bin/activate" ]; then
    source "${PROJECT_DIR}/myenv/bin/activate"
elif [ -f "$HOME/miniconda/bin/activate" ]; then
    source "$HOME/miniconda/bin/activate"
    conda activate cuda-build 2>/dev/null || conda activate base 2>/dev/null || true
fi

# Ensure writable matplotlib cache dir
export MPLCONFIGDIR="/tmp/mpl_cache_${USER:-user}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================================="
echo "Starting Plotting Task: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on Node: $(hostname)"
echo "Project Directory: $PROJECT_DIR"
echo "Python: $(which python3 || which python)"
echo "=========================================================="

# 3. Locate dataset
# Can pass custom dataset folder as first argument to sbatch, e.g.:
#   sbatch scripts/slurm_plot_figure6.sh output/time_evo_NC_8192.0_5.0_3.0_1.0
DATASET_ARG="${1:-}"

PLOT_SCRIPT="${PROJECT_DIR}/python/plot_figure6_nc_ew.py"

if [ -n "$DATASET_ARG" ]; then
    echo "Running plot on specified dataset: $DATASET_ARG"
    python3 "$PLOT_SCRIPT" "$DATASET_ARG" --layout all
else
    echo "Auto-detecting latest simulation dataset..."
    python3 "$PLOT_SCRIPT" --layout all
fi

echo "=========================================================="
echo "Plotting completed successfully!"
echo "Generated PDFs:"
echo "  - ${PROJECT_DIR}/paper/figuras/NC_EW.pdf (Standard Inset Layout)"
echo "  - ${PROJECT_DIR}/paper/figuras/NC_EW_side_by_side.pdf (Side-by-Side Layout)"
echo "  - ${PROJECT_DIR}/paper/figuras/NC_EW_stacked.pdf (Stacked 3-Plot Layout)"
echo "  - ${PROJECT_DIR}/paper/figuras/NC_EW_wide_row.pdf (Bonus 2-Column Wide Layout)"
echo "=========================================================="

