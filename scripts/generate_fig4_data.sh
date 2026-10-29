#!/bin/bash
#SBATCH --job-name=gen_fig4_data
#SBATCH --output=logs/fig4_%A_%a.out
#SBATCH --error=logs/fig4_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# ==============================================================================
# generate_fig4_data.sh - Regenerate Simulation Data for Figure 4
# Flexible extended DW velocities and roughness vs h0 at ha = 3.0
# Standardized sizes: Small (L=8, N=16) and Big (L=1024, N=2048)
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
    echo "  --num-h0 <int>           Number of DC field points (default: 200)"
    echo "  --ha <float>             AC amplitude (default: 3.0)"
    echo "  --eps <float>            Perturbation scale (default: 0.0005)"
    echo "  --seed <int>             Random seed (default: 42)"
    echo "  --out-dir <path>         Output directory (default: output/sweep_ha3_L8_L1024_recomputed)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

# Check passthrough arguments for figure 4 specific options
for ((arg_i=0; arg_i<${#CLI_PASSTHROUGH[@]}; arg_i++)); do
    case "${CLI_PASSTHROUGH[arg_i]}" in
        --size-small)
            SIZE_SMALL="${CLI_PASSTHROUGH[arg_i+1]}"
            ;;
        --size-big)
            SIZE_BIG="${CLI_PASSTHROUGH[arg_i+1]}"
            ;;
        --L-small)
            L_SMALL="${CLI_PASSTHROUGH[arg_i+1]}"
            ;;
        --L-big)
            L_BIG="${CLI_PASSTHROUGH[arg_i+1]}"
            ;;
    esac
done

# Standardized Parameters
HA="${HA:-3.0}"
H0_MIN="${H0_MIN:-0.0}"
H0_MAX="${H0_MAX:-8.0}"
NUM_H0="${NUM_H0:-200}"

L_SMALL="${L_SMALL:-8.0}"
SIZE_SMALL="${SIZE_SMALL:-16}"

L_BIG="${L_BIG:-1024.0}"
SIZE_BIG="${SIZE_BIG:-2048}"

TRANSIENT="${TRANSIENT:-131072}"
EVAL="${EVAL:-12566.3706143592}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/sweep_ha3_L8_L1024_recomputed}"
DIR_SMALL="${OUT_DIR}/L_8"
DIR_BIG="${OUT_DIR}/L_1024"
mkdir -p "$DIR_SMALL" "$DIR_BIG" "${PROJECT_DIR}/logs"

EXEC=$(find_executable "sweep" "$BACKEND")
if [ -z "$EXEC" ]; then
    echo "Error: 'sweep' executable not found."
    exit 1
fi

echo "=========================================================="
echo "Regenerating Data for Figure 4 (DW Metrics vs h0 at ha=3)"
echo "Backend: $BACKEND | Partition: $PARTITION | Executable: $EXEC"
echo "Sizes: Small L=$L_SMALL (N=$SIZE_SMALL) | Big L=$L_BIG (N=$SIZE_BIG) | dx=0.5"
echo "Physics: ha=$HA | h0 in [$H0_MIN, $H0_MAX] ($NUM_H0 points)"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

# Check if running as part of a SLURM array job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    TASK_STEPS=$((NUM_H0 > 1 ? NUM_H0 - 1 : 1))
    H0_VAL=$(awk -v min="$H0_MIN" -v max="$H0_MAX" -v steps="$TASK_STEPS" -v task="$SLURM_ARRAY_TASK_ID" \
             'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')
    echo "Running SLURM Array Task $SLURM_ARRAY_TASK_ID for h0 = $H0_VAL..."

    OUT_FILE_SMALL="${DIR_SMALL}/results_h0_${H0_VAL}.txt"
    OUT_FILE_BIG="${DIR_BIG}/results_h0_${H0_VAL}.txt"

    "$EXEC" "$SIZE_SMALL" "$L_SMALL" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA" "$HA" 1 "$OUT_FILE_SMALL" "$BACKEND" "$EPS" "$SEED"
    "$EXEC" "$SIZE_BIG" "$L_BIG" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA" "$HA" 1 "$OUT_FILE_BIG" "$BACKEND" "$EPS" "$SEED"
else
    # Sequential / Local batch loop over h0
    echo "Running full sweep over $NUM_H0 h0 points..."
    for ((i=0; i<NUM_H0; i++)); do
        TASK_STEPS=$((NUM_H0 > 1 ? NUM_H0 - 1 : 1))
        H0_VAL=$(awk -v min="$H0_MIN" -v max="$H0_MAX" -v steps="$TASK_STEPS" -v task="$i" \
                 'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')

        OUT_FILE_SMALL="${DIR_SMALL}/results_h0_${H0_VAL}.txt"
        OUT_FILE_BIG="${DIR_BIG}/results_h0_${H0_VAL}.txt"

        if [ ! -s "$OUT_FILE_SMALL" ]; then
            "$EXEC" "$SIZE_SMALL" "$L_SMALL" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA" "$HA" 1 "$OUT_FILE_SMALL" "$BACKEND" "$EPS" "$SEED"
        fi
        if [ ! -s "$OUT_FILE_BIG" ]; then
            "$EXEC" "$SIZE_BIG" "$L_BIG" "$ALPHA" "$OMEGA" "$DT" "$TRANSIENT" "$EVAL" "$H0_VAL" "$HA" "$HA" 1 "$OUT_FILE_BIG" "$BACKEND" "$EPS" "$SEED"
        fi

        if (( (i + 1) % 20 == 0 || i + 1 == NUM_H0 )); then
            echo "  Progress: [$((i + 1))/$NUM_H0] points computed (h0 = $H0_VAL)"
        fi
    done
fi

cat <<EOF > "${OUT_DIR}/run_parameters.txt"
Figure: 4 (Wall Metrics vs h0 at ha=3)
Date: $(date -Iseconds)
Backend: $BACKEND
Small_L: $L_SMALL
Small_Size: $SIZE_SMALL
Big_L: $L_BIG
Big_Size: $SIZE_BIG
Alpha: $ALPHA
Omega: $OMEGA
ha: $HA
h0_min: $H0_MIN
h0_max: $H0_MAX
num_h0: $NUM_H0
dt: $DT
Transient: $TRANSIENT
Eval: $EVAL
Eps: $EPS
Seed: $SEED
Output: $OUT_DIR
EOF

echo "Done! Figure 4 data generated in $OUT_DIR."
