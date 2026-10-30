#!/bin/bash
#SBATCH --job-name=gen_fig7_data
#SBATCH --output=logs/fig7_%j.out
#SBATCH --error=logs/fig7_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# ==============================================================================
# generate_fig7_data.sh - Regenerate Simulation Data for Figure 7
# Spatiotemporal profile cascade snapshots of the SP phase (h0=4, ha=3)
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
    echo "  --backend <cpu|gpu>      Compute backend (default: auto-detect)"
    echo "  --partition <gpu|knl_tacc> SLURM partition (default: gpu for GPU, knl_tacc for CPU)"
    echo "  --slurm                  Submit job via sbatch instead of running locally"
    echo "  --size <int>             Number of spatial grid points (default: 1024)"
    echo "  --L <float>              Domain length L/L0 (default: 512.0, dx=0.5)"
    echo "  --h0 <float>             DC magnetic field (default: 4.0)"
    echo "  --ha <float>             AC magnetic field amplitude (default: 3.0)"
    echo "  --eps <float>            Perturbation scale (default: 0.0005)"
    echo "  --seed <int>             Random seed (default: 42)"
    echo "  --out-dir <path>         Output directory (default: output/sp_snapshots)"
    exit 0
fi

check_slurm_dispatch "$0" "$@"
setup_cluster_modules

# Standardized Physical & Grid Defaults
SIZE="${SIZE:-1024}"
L="${L:-512.0}"
H0="${H0:-4.0}"
HA="${HA:-3.0}"
BATCH="${BATCH:-1}"
TRANSIENT="${TRANSIENT:-10000.0}"
SNAPSHOT_TIME="${SNAPSHOT_TIME:-12.566370614}"  # 4*pi / omega (2 complete drive cycles)
WINDOW="${WINDOW:-2.0}"
RATE="${RATE:-1}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/sp_snapshots}"
mkdir -p "$OUT_DIR" "${PROJECT_DIR}/logs"
PREFIX="${OUT_DIR}/sp_snap"

EXEC=$(find_executable "snapshots" "$BACKEND")
if [ -z "$EXEC" ]; then
    echo "Error: 'snapshots' executable not found."
    exit 1
fi

echo "=========================================================="
echo "Regenerating Data for Figure 7 (SP Phase Spatiotemporal Cascade)"
echo "Backend: $BACKEND | Partition: $PARTITION | Executable: $EXEC"
echo "Grid: SIZE=$SIZE | L=$L (dx=$(awk "BEGIN {print $L/$SIZE}"))"
echo "Physics: H0=$H0 | HA=$HA | OMEGA=$OMEGA | ALPHA=$ALPHA"
echo "Perturbation EPS=$EPS | SEED=$SEED"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

time "$EXEC" "$SIZE" "$BATCH" "$L" "$ALPHA" "$H0" "$HA" "$OMEGA" "$DT" \
     "$TRANSIENT" "$SNAPSHOT_TIME" "$WINDOW" "$RATE" "$PREFIX" "$EPS" "$SEED" "$BACKEND"

cat <<EOF > "${OUT_DIR}/run_parameters.txt"
Figure: 7 (SP Cascade Snapshots)
Date: $(date -Iseconds)
Backend: $BACKEND
Size: $SIZE
L: $L
Alpha: $ALPHA
Omega: $OMEGA
h0: $H0
ha: $HA
dt: $DT
Transient: $TRANSIENT
SnapshotTime: $SNAPSHOT_TIME
Eps: $EPS
Seed: $SEED
Prefix: $PREFIX
EOF

echo "Done! Figure 7 data generated in $OUT_DIR."

