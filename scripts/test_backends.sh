#!/usr/bin/env bash
set -eo pipefail

# Move to the project root directory (assuming script is run from scripts/)
cd "$(dirname "$0")/.."

echo "========================================"
echo "Setting up environment..."
echo "========================================"
# 1. Purge system modules to prevent conflicts with the Conda environment
module purge

# 2. Activate the Conda environment containing your modern toolchain
# Temporarily disable unbound variable checking to bypass the Conda script bug
set +u
source "$HOME/miniconda/bin/activate"
conda activate cuda-build
set -u

echo "========================================"
echo "Building the project (CPU + GPU)..."
echo "========================================"
BUILD_DIR="build_slurm"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Clean previous CMake cache artifacts
rm -rf CMakeCache.txt CMakeFiles/

# Configure and build
# 86 targets Ampere (RTX 3080, A10)
# 120 targets Blackwell (RTX 5070, 5070 Ti)
cmake .. \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86-real;86-virtual;120-real;120-virtual"

make -j"$(nproc)"
cd ..

echo ""
echo "========================================"
echo "Testing GPU Backend (Double Precision)..."
echo "========================================"
# Arguments: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <averaging_time> <window_time> <subsample_rate> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/main 256 2 50.0 0.27 4.0 3.0 1.0 0.05 10.0 10.0 2.0 5 test_gpu 0.05 42 gpu

echo ""
echo "========================================"
echo "Testing CPU Backend (Double Precision)..."
echo "========================================"
./${BUILD_DIR}/bin/main 256 2 50.0 0.27 4.0 3.0 1.0 0.05 10.0 10.0 2.0 5 test_cpu 0.05 42 cpu

echo ""
echo "========================================"
echo "Success! Both backends ran as intended."
echo "========================================"

# Clean up test output files
rm -f test_gpu* test_cpu*
