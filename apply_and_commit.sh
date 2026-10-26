#!/bin/bash
set -e

echo "Creating Commit 1: Build system and ignore rules..."
git add .gitignore CMakeLists.txt
git commit -m "build: update CMake for pure CUDA targets and add .aider to gitignore"

echo "Creating Commit 2: Core GPU backend and FFT batching enhancements..."
git add include/core/fft-handlers/cufft_handlers.cuh include/core/gpu_backend.cuh
git commit -m "core: enhance GPU backend with batched operations, streams, and central moments"

echo "Creating Commit 3: Refactor simulation core to CUDA with batching support..."
# Remove old .hpp files if they exist in git
git rm -qf include/domain_wall.hpp include/state.hpp include/stepper.hpp 2>/dev/null || true
git add include/domain_wall.cuh include/state.cuh include/stepper.cuh \
        include/simulation_observer.hpp include/sweep_observer.hpp
git commit -m "refactor: migrate simulation core to CUDA headers and introduce batching"

echo "Creating Commit 4: Migrate executables to pure CUDA..."
# Remove old .cpp files if they exist in git
git rm -qf src/sweep.cpp src/time_evo.cpp 2>/dev/null || true
git add src/sweep_main.cu src/time_evo.cu src/main.cu src/moments.cu \
        src/movies.cu src/saturation_check.cu src/snapshots.cu
git commit -m "feat: migrate and expand executables to pure CUDA"

echo "Creating Commit 5: Python plotting scripts enhancements and additions..."
git add python/plot_sweep.py python/plot_time_evo.py python/plot_main.py python/plot_scaling.py
git commit -m "feat(python): enhance plotting scripts for PRB style and add new analysis tools"

echo "Creating Commit 6: SLURM job scripts and cluster sync utilities..."
git add scripts/sge_run_time_evo.sh scripts/run_main.sh scripts/run_plot_main.sh \
        scripts/run_plot_scaling.sh scripts/run_plots.sh scripts/run_plots_manual.sh \
        scripts/scaling_analysis.sh scripts/slurm_time_evo.sh scripts/sweep.sh \
        sync_cluster.sh
git commit -m "chore: add SLURM submission scripts and cluster sync utility"

echo "All commits created successfully!"
