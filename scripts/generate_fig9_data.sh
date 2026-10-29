#!/bin/bash
#SBATCH --job-name=gen_fig9_data
#SBATCH --output=logs/fig9_%A_%a.out
#SBATCH --error=logs/fig9_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# ==============================================================================
# generate_fig9_data.sh - Regenerate Simulation Data for Figure 9
# Small-System Phase Diagram in the (h0, ha) plane for L/L0 = 8
# Standardized size: L = 8.0, Size N = 16 (dx = 0.5)
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
    echo "  --size <int>             Number of spatial grid points (default: 16)"
    echo "  --L <float>              Domain length L/L0 (default: 8.0, dx=0.5)"
    echo "  --h0-min <float>         Minimum DC field (default: 0.0)"
    echo "  --h0-max <float>         Maximum DC field (default: 10.0)"
    echo "  --h0-steps <int>         Number of h0 intervals (default: 100)"
    echo "  --ha-min <float>         Minimum AC field (default: 0.0)"
    echo "  --ha-max <float>         Maximum AC field (default: 20.0)"
    echo "  --ha-steps <int>         Number of ha intervals (default: 201)"
    echo "  --eps <float>            Perturbation scale (default: 0.0005)"
    echo "  --out-dir <path>         Output directory (default: output/sweep_full_L8_recomputed)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

SIZE="${SIZE:-16}"
L="${L:-8.0}"

H0_MIN="${H0_MIN:-0.0}"
H0_MAX="${H0_MAX:-10.0}"
H0_STEPS="${H0_STEPS:-100}"

HA_MIN="${HA_MIN:-0.0}"
HA_MAX="${HA_MAX:-20.0}"
HA_STEPS="${HA_STEPS:-201}"

TRANSIENT="${TRANSIENT:-131072}"
EVAL="${EVAL:-12566.3706143592}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/sweep_full_L8_recomputed}"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"

EXEC_SWEEP=$(find_executable "sweep" "$BACKEND")
if [ -z "$EXEC_SWEEP" ]; then
    echo "Error: 'sweep' executable not found."
    exit 1
fi

echo "=========================================================="
echo "Regenerating Data for Figure 9 (Small-System Phase Diagram L=8)"
echo "Backend: $BACKEND | Partition: $PARTITION | Executable: $EXEC_SWEEP"
echo "Grid: SIZE=$SIZE | L=$L (dx=$(awk "BEGIN {print $L/$SIZE}"))"
echo "Physics: h0 in [$H0_MIN, $H0_MAX] ($H0_STEPS steps) | ha in [$HA_MIN, $HA_MAX] ($HA_STEPS steps)"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

# 1. Parameter sweep execution
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    H0_VAL=$(awk -v min="$H0_MIN" -v max="$H0_MAX" -v steps="$H0_STEPS" -v task="$SLURM_ARRAY_TASK_ID" \
             'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')
    echo "Running SLURM Array Task $SLURM_ARRAY_TASK_ID for h0 = $H0_VAL..."
    OUT_FILE="${OUT_DIR}/results_h0_${H0_VAL}.txt"
    "$EXEC_SWEEP" "$SIZE" "$L" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA_MIN" "$HA_MAX" "$HA_STEPS" "$OUT_FILE" "$BACKEND" "$EPS" "$SEED"
else
    TOTAL_H0=$((H0_STEPS + 1))
    echo "Running 2D sweep across $TOTAL_H0 h0 values..."
    for ((task=0; task<=H0_STEPS; task++)); do
        H0_VAL=$(awk -v min="$H0_MIN" -v max="$H0_MAX" -v steps="$H0_STEPS" -v task="$task" \
                 'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')
        OUT_FILE="${OUT_DIR}/results_h0_${H0_VAL}.txt"
        if [ ! -s "$OUT_FILE" ]; then
            "$EXEC_SWEEP" "$SIZE" "$L" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA_MIN" "$HA_MAX" "$HA_STEPS" "$OUT_FILE" "$BACKEND" "$EPS" "$SEED"
        fi
        if (( (task + 1) % 10 == 0 || task == H0_STEPS )); then
            echo "  Progress: [$((task + 1))/$TOTAL_H0] h0 rows completed (h0 = $H0_VAL)"
        fi
    done
fi

# 2. Floquet stability dataset for L = 8
FLOQUET_OUT="${PROJECT_DIR}/output/floquet_transient_omega_1.000_L_8.bin"
if [ ! -s "$FLOQUET_OUT" ]; then
    echo "Computing Floquet stability dataset for L = $L..."
    EXEC_FLOQ=$(find_executable "floquet_sweep_cpu" "cpu")
    if [ -n "$EXEC_FLOQ" ]; then
        "$EXEC_FLOQ" --alpha "$ALPHA" --omega "$OMEGA" -L "$L" \
                     --h0-min "$H0_MIN" --h0-max "$H0_MAX" --num-h0 500 \
                     --ha-min "$HA_MIN" --ha-max "$HA_MAX" --num-ha 500 \
                     -o "$FLOQUET_OUT"
    fi
fi

cat <<EOF > "${OUT_DIR}/run_parameters.txt"
==========================================================
Run Parameters for Sweep Job: Figure 9 (L=8)
==========================================================
Timestamp:      $(date)
Backend:        $BACKEND
System Size:    $SIZE
Length (L):     $L
Alpha:          $ALPHA
Omega:          $OMEGA
dt:             $DT
Transient Time: $TRANSIENT
Eval Time:      $EVAL
h0 Range:       [$H0_MIN, $H0_MAX] with $H0_STEPS intervals
ha Range:       [$HA_MIN, $HA_MAX] with $HA_STEPS intervals
Eps:            $EPS
Seed:           $SEED
==========================================================
EOF

echo "Done! Figure 9 data generated in $OUT_DIR."

