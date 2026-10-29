#!/bin/bash
#SBATCH --job-name=gen_fig2_data
#SBATCH --output=logs/fig2_%j.out
#SBATCH --error=logs/fig2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# ==============================================================================
# generate_fig2_data.sh - Regenerate Simulation Data for Figure 2
# Rigid wall Shapiro steps and average velocities vs h0 for ha=0 and ha=5
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
    echo "  --h0-min <float>         Minimum DC magnetic field (default: 0.0)"
    echo "  --h0-max <float>         Maximum DC magnetic field (default: 10.0)"
    echo "  --num-h0 <int>           Number of DC field steps (default: 500)"
    echo "  --eps <float>            Perturbation scale (default: 0.0005)"
    echo "  --out-dir <path>         Output directory (default: output/fig2_data)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

H0_MIN="${H0_MIN:-0.0}"
H0_MAX="${H0_MAX:-10.0}"
NUM_H0="${NUM_H0:-500}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/fig2_data}"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"

echo "=========================================================="
echo "Regenerating Data for Figure 2 (Rigid Wall Shapiro Steps)"
echo "Backend: $BACKEND | Partition: $PARTITION"
echo "Physics: h0 in [$H0_MIN, $H0_MAX] ($NUM_H0 steps) | ha in {0, 5} | Alpha=$ALPHA | Omega=$OMEGA"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

if [ "$BACKEND" = "gpu" ]; then
    EXEC_GPU=$(find_executable "rigid_wall_sweep_gpu" "gpu")
    if [ -n "$EXEC_GPU" ]; then
        echo "Running CUDA GPU Rigid Wall Solver: $EXEC_GPU"
        "$EXEC_GPU" "$H0_MIN" "$H0_MAX" "$NUM_H0" 0.0 0.0 1 "${OUT_DIR}/rigid_wall_ha0.txt"
        "$EXEC_GPU" "$H0_MIN" "$H0_MAX" "$NUM_H0" 5.0 5.0 1 "${OUT_DIR}/rigid_wall_ha5.txt"
    else
        echo "CUDA binary not found. Falling back to OpenMP CPU solver..."
        BACKEND="cpu"
    fi
fi

if [ "$BACKEND" = "cpu" ]; then
    EXEC_CPU=$(find_executable "rigid_wall_sweep_cpu" "cpu")
    if [ -n "$EXEC_CPU" ]; then
        echo "Running OpenMP CPU Rigid Wall Solver: $EXEC_CPU"
        "$EXEC_CPU" --alpha "$ALPHA" --omega "$OMEGA" --h0-min "$H0_MIN" --h0-max "$H0_MAX" --num-h0 "$NUM_H0" --ha-min 0.0 --ha-max 0.0 --num-ha 1 -o "${OUT_DIR}/rigid_wall_ha0.txt"
        "$EXEC_CPU" --alpha "$ALPHA" --omega "$OMEGA" --h0-min "$H0_MIN" --h0-max "$H0_MAX" --num-h0 "$NUM_H0" --ha-min 5.0 --ha-max 5.0 --num-ha 1 -o "${OUT_DIR}/rigid_wall_ha5.txt"
    else
        echo "C++ binary not found. Running vectorized Python solver..."
        python3 "${PROJECT_DIR}/rigid_wall/python/rigid_wall.py"
    fi
fi

cat <<EOF > "${OUT_DIR}/run_parameters.txt"
Figure: 2 (Rigid Wall Shapiro Steps)
Date: $(date -Iseconds)
Backend: $BACKEND
Alpha: $ALPHA
Omega: $OMEGA
h0_min: $H0_MIN
h0_max: $H0_MAX
num_h0: $NUM_H0
ha_values: [0.0, 5.0]
Eps: $EPS
Seed: $SEED
Output: $OUT_DIR
EOF

echo "Done! Figure 2 data generated in $OUT_DIR."

