#!/bin/bash
#SBATCH --job-name=fig6_L131072
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/fig6_L131072_%j.out
#SBATCH --error=logs/fig6_L131072_%j.err

# ==============================================================================
# Slurm Job: Figure 6 Edwards-Wilkinson Analysis for L/L0 = 2^17 (131072)
# Grid: Size = 262144 (dx = 0.5), L = 131072.0
# Parameters: H0 = 5.0, HA = 3.0, ALPHA = 0.27, OMEGA = 1.0, EPS = 0.0005
# ==============================================================================

set -eo pipefail

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/inv1/proyecto-v3}"
cd "$PROJECT_DIR"

source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc/8.5.0-5sg556d 2>/dev/null || module load gcc/8.2.0-cytb66h 2>/dev/null || module load gcc 2>/dev/null || true
module load cuda/11.1.0-zj6lnzj 2>/dev/null || module load cuda 2>/dev/null || true

export LD_LIBRARY_PATH="/home/luz.perez/miniconda/envs/cuda-build/lib:/share/apps/spack/instalaciones/linux-westmere/cuda-11.1.0-z/lib64:$LD_LIBRARY_PATH"

EXEC="${PROJECT_DIR}/build_slurm/bin/time_evo"
if [ ! -x "$EXEC" ]; then
    echo "Executable $EXEC not found! Trying fallback search..."
    EXEC=$(find "${PROJECT_DIR}" -name time_evo -type f -perm /111 | head -n 1)
fi

echo "=========================================================="
echo "Executable: $EXEC"
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "GPU Device: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "=========================================================="

# Physics and Grid Parameters
L="131072.0"
SIZE="262144" # dx = 0.5
H0="5.0"
HA="3.0"
ALPHA="0.27"
OMEGA="1.0"
DT="0.0196349540849362" # 2*pi / (320*omega)
MAX_POWER="28"          # 2^28 steps (~5.27e6 time)
EPS="0.0005"
SEED="42"
BACKEND="gpu"

OUT_DIR="${PROJECT_DIR}/output/fig6_L131072_gpu"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"
PREFIX="${OUT_DIR}/time_data"

cat <<EOF > "${OUT_DIR}/parameters.txt"
Simulation: time_evo NC Edwards-Wilkinson (Giant System L=2^17)
Job ID: ${SLURM_JOB_ID:-manual}
Date: $(date -Iseconds)
Backend: $BACKEND
SIZE: $SIZE
L: $L
ALPHA: $ALPHA
H0: $H0
HA: $HA
OMEGA: $OMEGA
DT: $DT
MAX_POWER: $MAX_POWER
EPS: $EPS
SEED: $SEED
PREFIX: $PREFIX
EOF

echo "Starting Figure 6 Analysis for Giant System L/L0 = 2^17 ($L)"
echo "Grid Size: SIZE=$SIZE (dx=$(awk "BEGIN {print $L/$SIZE}"))"
echo "Physics: H0=$H0, HA=$HA, OMEGA=$OMEGA, ALPHA=$ALPHA"
echo "Time: dt=$DT, max_power=$MAX_POWER (total steps: 2^$MAX_POWER)"
echo "Output Directory: $OUT_DIR"
echo "=========================================================="

time "$EXEC" "$SIZE" "$L" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" "$MAX_POWER" "$PREFIX" "$EPS" "$SEED" "$BACKEND"

echo "=========================================================="
echo "Simulation completed successfully!"
echo "Outputs stored in: $OUT_DIR"
echo "=========================================================="

