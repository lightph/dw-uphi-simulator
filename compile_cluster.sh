#!/usr/bin/env bash

set -eo pipefail

# 1. Purge system modules to prevent conflicts with the Conda environment
# Added '|| true' to prevent failure if the module command is not available in this shell
module purge 2>/dev/null || true

# 2. Activate the Conda environment containing your modern toolchain
# Temporarily disable unbound variable checking to bypass Conda script bugs
set +u
source "$HOME/miniconda/bin/activate"
conda activate cuda-build

# 3. Auto-detect and download missing dependencies
echo "-- Checking environment dependencies..."
# Changed fftw3f to fftw3 since we moved to double precision exclusively
if ! command -v pkg-config | grep -q "miniconda" || ! pkg-config --exists fftw3 &>/dev/null || ! command -v ccache | grep -q "miniconda"; then
  echo "-- Missing Conda dependencies detected. Auto-downloading 'fftw', 'pkg-config', and 'ccache'..."
  conda install -y -c conda-forge pkg-config fftw ccache
else
  echo "-- Dependencies found."
fi

# Re-enable unbound variable checking only AFTER all Conda interactions are done
set -u

# 4. Setup the build directory
BUILD_DIR="build_slurm"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 5. Configure and build (Incremental)
cmake .. \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86-real;86-virtual;120-real;120-virtual"

make -j"$(nproc)"
