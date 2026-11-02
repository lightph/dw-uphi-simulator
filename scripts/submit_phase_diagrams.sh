#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo "Submitting 6 Phase Diagram Sweep Job Arrays"
echo "Constraint: Strictly 1 GPU node occupied at any time"
echo "Chaining mechanism: Slurm --dependency=afterany and %1"
echo "=========================================================="

JOB1=$(sbatch scripts/sweep_combo1_a027_w05.sh | awk '{print $4}')
echo "Submitted Combo 1 (alpha=0.27, omega=0.5): Job ID $JOB1"

JOB2=$(sbatch --dependency=afterany:$JOB1 scripts/sweep_combo2_a027_w15.sh | awk '{print $4}')
echo "Submitted Combo 2 (alpha=0.27, omega=1.5): Job ID $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --dependency=afterany:$JOB2 scripts/sweep_combo3_a027_w20.sh | awk '{print $4}')
echo "Submitted Combo 3 (alpha=0.27, omega=2.0): Job ID $JOB3 (depends on $JOB2)"

JOB4=$(sbatch --dependency=afterany:$JOB3 scripts/sweep_combo4_a010_w10.sh | awk '{print $4}')
echo "Submitted Combo 4 (alpha=0.10, omega=1.0): Job ID $JOB4 (depends on $JOB3)"

JOB5=$(sbatch --dependency=afterany:$JOB4 scripts/sweep_combo5_a010_w05.sh | awk '{print $4}')
echo "Submitted Combo 5 (alpha=0.10, omega=0.5): Job ID $JOB5 (depends on $JOB4)"

JOB6=$(sbatch --dependency=afterany:$JOB5 scripts/sweep_combo6_a050_w10.sh | awk '{print $4}')
echo "Submitted Combo 6 (alpha=0.50, omega=1.0): Job ID $JOB6 (depends on $JOB5)"

echo "=========================================================="
echo "All 6 phase diagram pipelines submitted successfully!"
echo "=========================================================="
squeue -u "$USER"
