#!/usr/bin/env bash
#SBATCH --job-name=compile_dw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=compile_%j.out
#SBATCH --error=compile_%j.err

set -eo pipefail

# 1. Purge system modules to prevent conflicts with the Conda environment
module purge

# 2. Activate the Conda environment containing your modern toolchain
# Temporarily disable unbound variable checking to bypass the Conda script bug
set +u
source "$HOME/miniconda/bin/activate"
conda activate cuda-build
set -u

# Jump to the directory where the script was submitted from
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

# 3. Setup the build directory
BUILD_DIR="build_slurm"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 4. Clean previous CMake cache artifacts
rm -rf CMakeCache.txt CMakeFiles/

# 5. Configure and build
# 86 targets Ampere (RTX 3080, A10)
# 120 targets Blackwell (RTX 5070, 5070 Ti)
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES="86-real;86-virtual;120-real;120-virtual"

make -j"$(nproc)"
