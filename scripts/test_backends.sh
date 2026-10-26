#!/bin/bash
set -e

# Move to the project root directory (assuming script is run from scripts/)
cd "$(dirname "$0")/.."

echo "========================================"
echo "Building the project (CPU + GPU)..."
echo "========================================"
mkdir -p build
cd build
cmake -DUSE_CUDA=ON ..
make -j
cd ..

echo ""
echo "========================================"
echo "Testing GPU Backend (Double Precision)..."
echo "========================================"
# Arguments: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <averaging_time> <window_time> <subsample_rate> <output_prefix> [eps] [seed] [backend]
./build/bin/main 256 2 50.0 0.27 4.0 3.0 1.0 0.05 10.0 10.0 2.0 5 test_gpu 0.05 42 gpu

echo ""
echo "========================================"
echo "Testing CPU Backend (Double Precision)..."
echo "========================================"
./build/bin/main 256 2 50.0 0.27 4.0 3.0 1.0 0.05 10.0 10.0 2.0 5 test_cpu 0.05 42 cpu

echo ""
echo "========================================"
echo "Success! Both backends ran as intended."
echo "========================================"

# Clean up test output files
rm -f test_gpu* test_cpu*
