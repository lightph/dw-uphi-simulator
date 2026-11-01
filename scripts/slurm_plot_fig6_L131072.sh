#!/bin/bash
#SBATCH --job-name=plot_fig6_L131072_knl
#SBATCH --partition=knl_tacc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/plot_fig6_L131072_%j.out
#SBATCH --error=logs/plot_fig6_L131072_%j.err

set -eo pipefail

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/inv1/proyecto-v3}"
cd "$PROJECT_DIR"

# Clean environment and activate python environment
module purge 2>/dev/null || true
module load python 2>/dev/null || true

if [ -f "$HOME/inv1/proyecto-v2/myenv/bin/activate" ]; then
    source "$HOME/inv1/proyecto-v2/myenv/bin/activate"
elif [ -f "${PROJECT_DIR}/myenv/bin/activate" ]; then
    source "${PROJECT_DIR}/myenv/bin/activate"
fi

PYTHON_BIN="$(which python3 || which python)"
echo "Using Python binary: $PYTHON_BIN"
$PYTHON_BIN -c "import numpy, matplotlib; print('Modules verified: numpy and matplotlib available')"

OUTDIR="${PROJECT_DIR}/figures_L131072"
mkdir -p "$OUTDIR"

echo "Running plot_figure6_nc_ew.py for giant system L=2^17 on compute node: $(hostname)"
$PYTHON_BIN "${PROJECT_DIR}/python/plot_figure6_nc_ew.py" "${PROJECT_DIR}/output/fig6_L131072_gpu" --layout all --outdir "$OUTDIR"

echo "Figure 6 plots for L=2^17 generated successfully in $OUTDIR"
