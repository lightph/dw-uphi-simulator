#!/usr/bin/env bash
set -eo pipefail

# Jump to the project root directory (assuming script is run from scripts/)
cd "$(dirname "$0")/.."

echo "========================================"
echo "Testing CPU Backend Locally (Double Precision)..."
echo "========================================"
BUILD_DIR="build_local"

if [ ! -f "${BUILD_DIR}/bin/main" ]; then
    echo "Error: Executable not found. Please run ./compile_local.sh first."
    exit 1
fi

# Arguments: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <averaging_time> <window_time> <subsample_rate> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/main 256 2 50.0 0.27 4.0 3.0 1.0 0.05 10.0 10.0 2.0 5 test_local_cpu 0.05 42 cpu

echo ""
echo "========================================"
echo "Success! Local CPU backend ran as intended."
echo "========================================"

# Clean up test output files
rm -f test_local_cpu*
