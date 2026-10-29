#!/usr/bin/env bash
set -eo pipefail

echo "========================================"
echo "Compiling project locally (CPU only)..."
echo "========================================"

export CCACHE_DIR="${TMPDIR:-/tmp}/ccache_${USER}"
export CCACHE_DISABLE=1
if [ -z "${CC}" ]; then export CC="/usr/bin/gcc"; fi
if [ -z "${CXX}" ]; then export CXX="/usr/bin/g++"; fi

BUILD_DIR="build_local"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure and build without CUDA (Incremental)
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"

echo "========================================"
echo "Compilation complete. Executables are in ${BUILD_DIR}/bin/"
echo "========================================"
