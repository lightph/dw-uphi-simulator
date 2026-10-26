#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

set -eo pipefail

# Jump to the project root directory (assuming script is run from scripts/)
cd "$SLURM_SUBMIT_DIR/.."

echo "========================================"
echo "Testing GPU Backend (Double Precision)..."
echo "========================================"
BUILD_DIR="build_slurm"

# Arguments: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <averaging_time> <window_time> <subsample_rate> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/main 256 2 50.0 0.27 4.0 3.0 1.0 0.05 10.0 10.0 2.0 5 test_gpu 0.05 42 gpu

echo ""
echo "========================================"
echo "Success! GPU backend ran as intended."
echo "========================================"

# Clean up test output files
rm -f test_gpu*
