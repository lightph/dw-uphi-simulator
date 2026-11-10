#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo "Submitting Remaining Phase Diagram Sweep Job Arrays"
echo "Constraint: Strictly 1 GPU node occupied at any time"
echo "Chaining mechanism: Slurm --dependency=afterany and %1"
echo "=========================================================="

# Combo 1 and Combo 2 already completed (results saved in output/).
# Submitting Combo 3 (will automatically skip already completed tasks 0-8)
JOB3=$(sbatch scripts/sweep_combo3_a027_w20.sh | awk '{print $4}')
echo "Submitted Combo 3 (alpha=0.27, omega=2.0): Job ID $JOB3"

JOB4=$(sbatch --dependency=afterany:$JOB3 scripts/sweep_combo4_a010_w10.sh | awk '{print $4}')
echo "Submitted Combo 4 (alpha=0.10, omega=1.0): Job ID $JOB4 (depends on $JOB3)"

JOB5=$(sbatch --dependency=afterany:$JOB4 scripts/sweep_combo5_a010_w05.sh | awk '{print $4}')
echo "Submitted Combo 5 (alpha=0.10, omega=0.5, dt-fixed): Job ID $JOB5 (depends on $JOB4)"

JOB6=$(sbatch --dependency=afterany:$JOB5 scripts/sweep_combo6_a050_w10.sh | awk '{print $4}')
echo "Submitted Combo 6 (alpha=0.50, omega=1.0): Job ID $JOB6 (depends on $JOB5)"

JOB1_RERUN=$(sbatch --dependency=afterany:$JOB6 scripts/sweep_combo1_a027_w05.sh | awk '{print $4}')
echo "Submitted Combo 1 Rerun (alpha=0.27, omega=0.5, dt-fixed): Job ID $JOB1_RERUN (depends on $JOB6)"

echo "=========================================================="
echo "All remaining phase diagram pipelines submitted successfully!"
echo "=========================================================="
squeue -u "$USER"
