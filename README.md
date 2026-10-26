# Domain Wall Simulator

A high-performance, batched pseudospectral simulator for 1D domain wall dynamics. This project is designed to study the non-linear dynamics, stability, and phase transitions of domain walls under external driving fields. 

The simulator features a unified API that seamlessly compiles for both **GPU (CUDA + cuFFT)** and **CPU (OpenMP + FFTW3)** backends, allowing for massive parallelization of parameter sweeps and time evolution analysis. All simulations are performed in double precision.

## Features

- **Dual Backend Support:** Run on NVIDIA GPUs for maximum throughput or fallback to multi-core CPUs without changing the simulation code. Both backends can be compiled simultaneously.
- **Batched Execution:** Efficiently simulates multiple independent systems simultaneously using batched FFTs.
- **Pseudospectral Stepper:** Highly accurate spatial derivatives and linear propagation in Fourier space.
- **Advanced Observables:** On-the-fly computation of spatial variance (rugosity), spectral entropy, power spectra, histograms, central moments, and cumulants.
- **Publication-Ready Plotting:** Python scripts configured for Physical Review B (PRB) style plots, including Floquet boundary overlays and Family-Vicsek scaling collapse.

## Prerequisites

### C++ Build Requirements
- **CMake** (>= 3.19)
- **C++17** compatible compiler (GCC, Clang, or MSVC)
- **CUDA Toolkit** (>= 11.0, recommended 12.x) - *Required for GPU backend*
- **FFTW3** (with OpenMP support: `fftw3`, `fftw3_omp`) - *Always required*
- **OpenMP** - *Always required*
- **Doxygen** (Optional) - *Required for generating code documentation*

### Python Plotting Requirements
- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`

You can install the Python dependencies via pip:
```bash
pip install numpy pandas matplotlib
```

## Build Instructions

The project uses CMake to manage the build process. By default, it will compile both the CPU and GPU backends into the executables if CUDA is found. You can disable the GPU backend using the `USE_CUDA` CMake option.

### Option A: Build for GPU and CPU (Default)
This will compile the project using `nvcc` and link against the CUDA Runtime, cuFFT, FFTW3, and OpenMP libraries.

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Option B: Build for CPU-Only
This will compile the project using your standard C++ compiler and link against FFTW3 and OpenMP, treating `.cu` files as standard C++ files.

```bash
mkdir -p build-cpu
cd build-cpu
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Generating Documentation
The codebase is fully documented using Doxygen. To generate the HTML and LaTeX documentation:

```bash
doxygen Doxyfile
```
You can then open `docs/html/index.html` in your web browser to view the API documentation.

## Usage Instructions

After building, the executables will be available in the `build/bin/` directory. All executables accept an optional `[backend]` argument at the end (`gpu` or `cpu`, defaults to `gpu`).

### Running Simulations

- **`main`**: Tracks the time evolution of a specific parameter set, outputting transient variance, instantaneous power spectra, and real-space state snapshots. Supports batched execution.
  ```bash
  ./bin/main <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <averaging_time> <window_time> <subsample_rate> <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
  ```

- **`time_evo`**: Tracks the time evolution using logarithmic time steps (powers of 2) for long-term scaling analysis.
  ```bash
  ./bin/time_evo <size> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
  ```

- **`sweep`**: Performs a 1D or 2D parameter sweep over the driving field parameters ($h_0$, $h_a$). Computes time-averaged observables to generate phase diagrams.
  ```bash
  ./bin/sweep <size> <L> <alpha> <omega> <dt> <transient_time> <eval_time> <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> [backend=gpu]
  ```

- **`moments`**: Computes higher-order statistical moments and cumulants of the domain wall profile.
  ```bash
  ./bin/moments <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <total_time> <start_time> <num_checkpoints> <num_sq_checkpoints> <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
  ```

- **`snapshots`**: Generates raw data for visualizing the domain wall profile over time.
  ```bash
  ./bin/snapshots <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <snapshot_time> <window_time> <snapshot_rate> <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
  ```

- **`movies`**: Generates binary snapshot data for high-framerate animations.
  ```bash
  ./bin/movies <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <t_max> <window_time> <movie_start> <movie_end> <movie_rate> <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
  ```

- **`sat_check`**: Utility to check saturation and convergence of the system.
  ```bash
  ./bin/sat_check <size> <L> <alpha> <dt> [backend=gpu]
  ```

### Data Analysis and Plotting

The `python/` directory contains scripts to process the raw `.txt` outputs from the C++ executables and generate high-quality plots.

**1. Plotting Parameter Sweeps**
Generates phase diagrams, spectral entropy maps, and variance plots.
```bash
python python/plot_sweep.py <path_to_input_directory> <path_to_output_directory>
```
*Note: The input directory should contain the `results_h0_*.txt` files generated by the `sweep` executable.*

**2. Plotting Time Evolution**
Generates time-series plots, power spectra, histograms, and animated `.mp4` videos of the domain wall dynamics.
```bash
python python/plot_time_evo.py <path_to_base_prefix> [k_cutoff] [--override] [--outdir <path>]
```
- `<path_to_base_prefix>`: The prefix used when running `time_evo` or `main` (e.g., `output/time_data`).
- `[k_cutoff]`: (Optional) Maximum wavenumber to plot in spatial power spectra.
- `--override`: Force regeneration of all plots and videos.

**3. Plotting Scaling Analysis**
Generates Family-Vicsek scaling collapse and system size dependence plots.
```bash
python python/plot_scaling.py <path_to_scaling_base_directory>
```

## Project Structure

```text
.
├── CMakeLists.txt              # CMake build configuration
├── include/
│   ├── core/                   # Core backend logic (CPU/GPU abstractions)
│   │   ├── cpu_backend.hpp     # OpenMP + FFTW implementation
│   │   ├── gpu_backend.cuh     # CUDA + cuFFT implementation
│   │   └── fft-handlers/       # Wrappers for cuFFT and FFTW
│   ├── domain_wall.cuh         # Main simulation class
│   ├── simulation_observer.hpp # Handles data collection and I/O
│   ├── state.cuh               # Data structures for the physical state
│   └── stepper.cuh             # Pseudospectral time integration
├── src/                        # C++ Executable entry points
├── python/                     # Data analysis and plotting scripts
└── scripts/                    # Bash scripts for cluster submission (SGE/Slurm)
```
