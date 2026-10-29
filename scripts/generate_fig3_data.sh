#!/bin/bash
#SBATCH --job-name=gen_fig3_data
#SBATCH --output=logs/fig3_%j.out
#SBATCH --error=logs/fig3_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# ==============================================================================
# generate_fig3_data.sh - Regenerate Simulation Data for Figure 3
# Linear Floquet stability diagrams for ha = 2.0 and ha = 3.0
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

parse_common_args "$@"

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --backend <cpu|gpu>      Compute backend (default: auto-detect)"
    echo "  --partition <gpu|knl_tacc> SLURM partition (default: gpu for GPU, knl_tacc for CPU)"
    echo "  --slurm                  Submit job via sbatch instead of running locally"
    echo "  --h0-min <float>         Minimum DC field (default: 0.0)"
    echo "  --h0-max <float>         Maximum DC field (default: 8.0)"
    echo "  --num-h0 <int>           Number of DC field steps (default: 300)"
    echo "  --k-min <float>          Minimum wavenumber kappa (default: 0.7)"
    echo "  --k-max <float>          Maximum wavenumber kappa (default: 1.2)"
    echo "  --num-k <int>            Number of kappa steps (default: 200)"
    echo "  --eps <float>            Initial perturbation scale (default: 0.0005)"
    echo "  --out-dir <path>         Output directory (default: output/fig3_data)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

H0_MIN="${H0_MIN:-0.0}"
H0_MAX="${H0_MAX:-8.0}"
NUM_H0="${NUM_H0:-300}"
K_MIN="${K_MIN:-0.7}"
K_MAX="${K_MAX:-1.2}"
NUM_K="${NUM_K:-200}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/fig3_data}"
CACHE_DIR="${PROJECT_DIR}/output/.cache"
mkdir -p "$OUT_DIR" "$CACHE_DIR" "${PROJECT_DIR}/logs"

echo "=========================================================="
echo "Regenerating Data for Figure 3 (Floquet Stability Diagrams)"
echo "Backend: $BACKEND | Partition: $PARTITION"
echo "Grid: h0 in [$H0_MIN, $H0_MAX] ($NUM_H0 pts) | kappa in [$K_MIN, $K_MAX] ($NUM_K pts)"
echo "Physics: ha in {2.0, 3.0} | Alpha=$ALPHA | Omega=$OMEGA"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR (and cache: $CACHE_DIR)"
echo "=========================================================="

# 1. Run via C++ OpenMP / GPU binary if available
EXEC_CPU=$(find_executable "floquet_sweep_cpu" "cpu")
EXEC_GPU=$(find_executable "floquet_sweep_gpu" "gpu")

if [ "$BACKEND" = "gpu" ] && [ -n "$EXEC_GPU" ]; then
    echo "Running CUDA GPU Floquet Solver: $EXEC_GPU"
    "$EXEC_GPU" "$NUM_H0" 2 "$NUM_K" "${OUT_DIR}/floquet_grid_ha_2.00.bin"
    "$EXEC_GPU" "$NUM_H0" 3 "$NUM_K" "${OUT_DIR}/floquet_grid_ha_3.00.bin"
elif [ -n "$EXEC_CPU" ]; then
    echo "Running OpenMP CPU Floquet Solver: $EXEC_CPU"
    "$EXEC_CPU" --alpha "$ALPHA" --omega "$OMEGA" --h0-min "$H0_MIN" --h0-max "$H0_MAX" --num-h0 "$NUM_H0" \
                --ha 2.0 --k-min "$K_MIN" --k-max "$K_MAX" --num-k "$NUM_K" -o "${OUT_DIR}/floquet_grid_ha_2.00.txt"
    "$EXEC_CPU" --alpha "$ALPHA" --omega "$OMEGA" --h0-min "$H0_MIN" --h0-max "$H0_MAX" --num-h0 "$NUM_H0" \
                --ha 3.0 --k-min "$K_MIN" --k-max "$K_MAX" --num-k "$NUM_K" -o "${OUT_DIR}/floquet_grid_ha_3.00.txt"
fi

# 2. Also regenerate the Python NPZ cache used by publication plotting
echo "Regenerating Python NPZ cache for publication plotter..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '${PROJECT_DIR}/floquet/python')
from plot_figure3 import get_cached_grid
import numpy as np

h0_vals = np.linspace($H0_MIN, $H0_MAX, $NUM_H0)
kappa_vals = np.linspace($K_MIN, $K_MAX, $NUM_K)
cache_dir = Path('${CACHE_DIR}')

print('Generating ha = 2.0 cache...')
get_cached_grid(2.0, h0_vals, kappa_vals, cache_dir, alpha=$ALPHA, omega=$OMEGA, recompute=True)
print('Generating ha = 3.0 cache...')
get_cached_grid(3.0, h0_vals, kappa_vals, cache_dir, alpha=$ALPHA, omega=$OMEGA, recompute=True)
print('Floquet NPZ cache generation complete.')
"

cat <<EOF > "${OUT_DIR}/run_parameters.txt"
Figure: 3 (Floquet Stability Diagrams)
Date: $(date -Iseconds)
Backend: $BACKEND
Alpha: $ALPHA
Omega: $OMEGA
h0_min: $H0_MIN
h0_max: $H0_MAX
num_h0: $NUM_H0
k_min: $K_MIN
k_max: $K_MAX
num_k: $NUM_K
ha_values: [2.0, 3.0]
Eps: $EPS
Seed: $SEED
Output: $OUT_DIR
EOF

echo "Done! Figure 3 data generated in $OUT_DIR and $CACHE_DIR."

