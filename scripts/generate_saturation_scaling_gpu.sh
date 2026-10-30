#!/bin/bash
#SBATCH --job-name=sat_scaling_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --array=3-13
#SBATCH --output=logs/sat_scaling_%A_%a.out
#SBATCH --error=logs/sat_scaling_%A_%a.err

# ==============================================================================
# generate_saturation_scaling_gpu.sh
#
# Runs properly saturated finite-size simulations on the GPU partition across
# sizes L/L0 = 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (Array 3-13).
#
# Key physics requirement:
#   For Edwards-Wilkinson / KPZ scaling, saturation time scales as t_sat ~ L^z = L^2.
#   To observe TRUE stationary roughness w^2_sat for large systems, the transient
#   time must scale quadratically with system length:
#       T_trans(L) = max(2000.0, 1.0 * L^2)
#       T_avg(L)   = max(1000.0, 0.3 * L^2)
#
# Defaults:
#   NR Phase (Edwards-Wilkinson): H0 = 5.0, HA = 3.0, ALPHA = 0.27, OMEGA = 1.0
#   Grid resolution: dx = 0.5 (SIZE = 2 * L)
#   Perturbation: EPS = 0.0005, SEED = 42
# ==============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/scripts/common_env.sh" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
    elif [ -f "${SLURM_SUBMIT_DIR}/common_env.sh" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/common_env.sh"

parse_common_args "$@"

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --slurm                  Submit job array via sbatch to cluster GPU partition"
    echo "  --partition <gpu>        SLURM partition (default: gpu)"
    echo "  --h0 <float>             DC field (default: 5.0 for NR phase, 4.0 for SP phase)"
    echo "  --ha <float>             AC field amplitude (default: 3.0)"
    echo "  --alpha <float>          Gilbert damping (default: 0.27)"
    echo "  --omega <float>          AC frequency (default: 1.0)"
    echo "  --eps <float>            Initial noise perturbation (default: 0.0005)"
    echo "  --seed <int>             Random seed (default: 42)"
    echo "  --exp <int>              Run single exponent L = 2^exp (3 to 13)"
    echo "  --out-dir <path>         Base output directory (default: output/saturation_scaling_gpu)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

# Ensure CUDA runtime library paths are loaded
if [ -d "/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64" ]; then
    export LD_LIBRARY_PATH="/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:$LD_LIBRARY_PATH"
fi
if [ -d "${PROJECT_DIR}/fftw_local/lib" ]; then
    export LD_LIBRARY_PATH="${PROJECT_DIR}/fftw_local/lib:$LD_LIBRARY_PATH"
fi

# Physics Parameters
H0="${H0:-5.0}"
HA="${HA:-3.0}"
ALPHA="${ALPHA:-0.27}"
OMEGA="${OMEGA:-1.0}"
DT="${DT:-0.049087385212340519}" # 2*pi / (128*omega)
EPS="${EPS:-0.0005}"
SEED="${SEED:-42}"
BATCH=1
BACKEND="gpu"

# Determine system size exponent:
# In SLURM array, EXP is SLURM_ARRAY_TASK_ID (range 3 to 13)
# In manual / local run, EXP can be passed via --exp <int> or defaults to 6 (L=64)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    EXP="$SLURM_ARRAY_TASK_ID"
elif [ -n "$EXP_PARAM" ]; then
    EXP="$EXP_PARAM"
elif [ -n "$L" ]; then
    # Calculate exponent from L
    EXP=$(awk "BEGIN {print int(log($L)/log(2) + 0.5)}")
else
    EXP=6
fi

L_INT=$((1 << EXP))
L="${L_INT}.0"
SIZE=$((1 << (EXP + 1))) # dx = 0.5

# Properly scaled saturation timescales: t_sat ~ L^2
# For small L, enforce a minimum equilibration time of 2000.0
# For large L, scale transient time as 1.0 * L^2, and averaging time as 0.3 * L^2
TRANSIENT_TIME=$(awk "BEGIN {t = 1.0 * $L_INT * $L_INT; if (t < 2000.0) t = 2000.0; printf \"%.6f\", t}")
AVERAGING_TIME=$(awk "BEGIN {t = 0.3 * $L_INT * $L_INT; if (t < 1000.0) t = 1000.0; printf \"%.6f\", t}")
WINDOW_TIME=$(awk "BEGIN {w = 0.05 * $TRANSIENT_TIME; if (w > 2000.0) w = 2000.0; if (w < 200.0) w = 200.0; printf \"%.6f\", w}")

# Subsampling rate for transient metrics logging
if [ "$L_INT" -le 128 ]; then
    SUBSAMPLE_RATE=16
elif [ "$L_INT" -le 1024 ]; then
    SUBSAMPLE_RATE=64
elif [ "$L_INT" -le 4096 ]; then
    SUBSAMPLE_RATE=256
else
    SUBSAMPLE_RATE=1024
fi

LABEL="sat_L${L_INT}"
BASE_OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/saturation_scaling_gpu}/main_NC_sat_${L}_${H0}_${HA}_${OMEGA}"
OUT_DIR="$BASE_OUT_DIR"

mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"
PREFIX="${OUT_DIR}/time_data"
PARAM_FILE="${OUT_DIR}/parameters.txt"

EXEC=$(find_executable "main" "gpu")
if [ -z "$EXEC" ]; then
    echo "Error: GPU 'main' executable not found."
    exit 1
fi

cat <<EOF > "$PARAM_FILE"
Job Name: ${SLURM_JOB_NAME:-saturation_scaling_gpu}
Job ID: ${SLURM_JOB_ID:-manual}
Array Task: ${SLURM_ARRAY_TASK_ID:-$EXP}
Label: $LABEL
Size: $SIZE
Batch: $BATCH
L: $L
Alpha: $ALPHA
H0: $H0
Ha: $HA
Omega: $OMEGA
dt: $DT
Transient Time: $TRANSIENT_TIME
Averaging Time: $AVERAGING_TIME
Window Time: $WINDOW_TIME
Subsample Rate: $SUBSAMPLE_RATE
Eps: $EPS
Seed: $SEED
Backend: $BACKEND
EOF

echo "=========================================================="
echo "Saturated Roughness Scaling Run (GPU Task $EXP)"
echo "Executable: $EXEC"
echo "Grid: L=$L | Size=$SIZE (dx=0.5)"
echo "Physics: H0=$H0 | HA=$HA | ALPHA=$ALPHA | OMEGA=$OMEGA"
echo "Saturation Scaling Timescales:"
echo "  Transient Time: $TRANSIENT_TIME (~ $EXP L^2 steps)"
echo "  Averaging Time: $AVERAGING_TIME"
echo "  Window Time   : $WINDOW_TIME"
echo "  Subsample Rate: $SUBSAMPLE_RATE"
echo "  Perturbation  : EPS=$EPS | SEED=$SEED"
echo "Saving to: $OUT_DIR"
echo "=========================================================="

time "$EXEC" "$SIZE" "$BATCH" "$L" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" "$TRANSIENT_TIME" "$AVERAGING_TIME" "$WINDOW_TIME" "$SUBSAMPLE_RATE" "$PREFIX" "$EPS" "$SEED" "$BACKEND"

echo "Task complete for L=$L."
