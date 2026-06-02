# Domain Wall Simulator

This repository contains the C++ and CUDA simulation code used to model domain wall dynamics for the associated physics paper. 

The project features an optimized core utilizing pseudospectral methods to solve partial differential equations. It supports both CPU-only execution (via OpenMP) and highly parallel GPU-accelerated execution (via CUDA).

## Project Structure
* `domain_wall_simulator/`: Core physics simulator interfaces and implementations.
* `fft-handlers/`: Fast Fourier Transform wrappers for CPU (FFTW3) and GPU (cuFFT).
* `include/`: Shared project headers and utilities (e.g., file saving routines).
* `programs/`: Executable entry points for the active experiments (`h_sweep`, `complete_h_sweep`, `structure_analysis`).
* `python/`: Post-processing, data analysis, and plotting scripts.

## Dependencies
To compile this project, the following dependencies are required:
* **CMake** (3.15 or higher)
* **C++17** compatible compiler (GCC/Clang)
* **OpenMP**
* **FFTW3** (must include OpenMP support: `fftw3_omp` / `fftw3f_omp`)
* **CUDA Toolkit** (Optional but recommended; supports architectures 52 through 86)
* **ccache** (Optional, speeds up recompilation)

## Build Instructions

By default, the build system compiles with CUDA support and double precision. All executables and libraries are automatically routed to `bin/` and `lib/` subdirectories within the build folder.

1. Create a build directory and run CMake:
   ```bash
   mkdir build
   cd build
   cmake .. -DUSE_CUDA=ON -DUSE_DOUBLE_PRECISION=ON
   ```
   *Configuration Notes:* * Set `-DUSE_CUDA=OFF` if you do not have a compatible NVIDIA GPU. 
   * Set `-DUSE_DOUBLE_PRECISION=OFF` to compile in single precision for improved performance (not recommended).

2. Compile the project:
   ```bash
   make -j$(nproc)
   ```

## Usage
Once compiled, the resulting binaries are located in the `build/bin/` directory. They will have a suffix corresponding to the precision used during the build (`_dp` for double precision, `_sp` for single precision).

Available targets:
* `h_sweep_dp`
* `complete_h_sweep_dp`
* `structure_analysis_dp`
