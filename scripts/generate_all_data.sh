#!/bin/bash
# ==============================================================================
# generate_all_data.sh - Master Data Generation Suite for Publication Figures
# Physical Review B: "Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls"
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

parse_common_args "$@"

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --backend <cpu|gpu>        Compute backend (default: auto-detect)"
    echo "  --partition <gpu|knl_tacc> SLURM partition (default: gpu for GPU, knl_tacc for CPU)"
    echo "  --slurm                    Submit jobs via sbatch to SLURM cluster"
    echo "  --figures <list>           Specific figure numbers to regenerate (e.g. --figures 1 2 4)"
    echo "  --eps <float>              Initial perturbation amplitude (default: 0.0005)"
    echo "  --seed <int>               Random seed (default: 42)"
    exit 0
fi

FIGURES_TO_RUN=()
while [ $# -gt 0 ]; do
    case "$1" in
        --figures)
            shift
            while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
                FIGURES_TO_RUN+=("$1")
                shift
            done
            ;;
        *)
            shift
            ;;
    esac
done

if [ ${#FIGURES_TO_RUN[@]} -eq 0 ]; then
    FIGURES_TO_RUN=(1 2 3 4 5 6 7 8 9)
fi

echo "=================================================================="
echo "PHYSICAL REVIEW B DATA REGENERATION MASTER SUITE"
echo "=================================================================="
echo "Backend:    $BACKEND"
echo "Partition:  $PARTITION"
echo "SLURM Mode: $SUBMIT_SLURM"
echo "Default EPS: $EPS"
echo "Selected Figures: ${FIGURES_TO_RUN[*]}"
echo "=================================================================="

for fig in "${FIGURES_TO_RUN[@]}"; do
    script="${SCRIPT_DIR}/generate_fig${fig}_data.sh"
    if [ ! -f "$script" ]; then
        echo "Warning: Script '$script' not found. Skipping Figure $fig."
        continue
    fi

    echo ""
    echo "------------------------------------------------------------------"
    echo "Executing Data Generation for Figure $fig: $(basename "$script")"
    echo "------------------------------------------------------------------"

    args=("--backend" "$BACKEND" "--partition" "$PARTITION" "--eps" "$EPS")
    if [ "$SUBMIT_SLURM" = true ]; then
        args+=("--slurm")
    fi

    "$script" "${args[@]}"
done

echo ""
echo "=================================================================="
echo "Data generation sequence triggered/completed successfully."
echo "=================================================================="
