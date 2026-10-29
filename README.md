# Domain Wall Simulator: Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.19%2B-orange.svg)](https://cmake.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official C++ / CUDA implementation, high-performance simulation engine, and publication artifact package for:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, and A. B. Kolton  
> *Instituto Balseiro (UNCUYO) & Centro Atómico Bariloche (CNEA-CONICET), San Carlos de Bariloche, Río Negro, Argentina.*  
> GitHub Repository: [https://github.com/lightph/dw-uphi-simulator](https://github.com/lightph/dw-uphi-simulator)

---

## Table of Contents

- [Overview & Physical Model](#overview--physical-model)
  - [The Slonczewski ($u-\phi$) Model](#the-slonczewski-u--phi-model)
  - [Conversion to Physical Units (Co/Pt Multilayers)](#conversion-to-physical-units-copt-multilayers)
  - [The Five Dynamical Phases](#the-five-dynamical-phases)
- [Repository Tour](#repository-tour)
- [Paper Figure Reproduction Guide](#paper-figure-reproduction-guide)
  - [Figure 2: Shapiro Steps & Average Velocities in Rigid DWs](#figure-2-shapiro-steps--average-velocities-in-rigid-dws)
  - [Figure 3: Floquet Linear Stability Diagrams](#figure-3-floquet-linear-stability-diagrams)
  - [Figure 4: Flexible Extended DW Dynamics vs $h_0$ ($L/L_0=8$ and $1024$)](#figure-4-flexible-extended-dw-dynamics-vs-h_0-ll_08-and-1024)
  - [Figure 5: Full Dynamical Phase Diagram in the $(h_0, h_a)$ Plane](#figure-5-full-dynamical-phase-diagram-in-the-h_0-h_a-plane)
  - [Figure 6: Universal Edwards-Wilkinson Scaling Collapse of the NR Phase](#figure-6-universal-edwards-wilkinson-scaling-collapse-of-the-nr-phase)
  - [Figure 7: Spatiotemporal Cascade of the SP Phase](#figure-7-spatiotemporal-cascade-of-the-sp-phase)
  - [Figure 8: Finite-Size Scaling of the SP Phase ($w^2$ and $\kappa_{\mathrm{peak}}$ vs $L/L_0$)](#figure-8-finite-size-scaling-of-the-sp-phase-w2-and-kappa_mathrmpeak-vs-ll_0)
  - [Figure 9 (Appendix): Finite-Size Phase Diagram for $L/L_0=8$](#figure-9-appendix-finite-size-phase-diagram-for-ll_08)
- [Prerequisites & Dependencies](#prerequisites--dependencies)
- [Compilation & Build](#compilation--build)
  - [Quick Automated Build Scripts](#quick-automated-build-scripts)
  - [Manual CMake Build (GPU + CPU)](#manual-cmake-build-gpu--cpu)
  - [Manual CMake Build (CPU Only)](#manual-cmake-build-cpu-only)
  - [Verifying the Build Locally](#verifying-the-build-locally)
- [Simulation Executables Reference](#simulation-executables-reference)
  - [1. `main` (Standard Time Evolution & Observables)](#1-main-standard-time-evolution--observables)
  - [2. `time_evo` (Logarithmic Sampling & Surface Growth)](#2-time_evo-logarithmic-sampling--surface-growth)
  - [3. `sweep` (1D / 2D Parameter Sweeps & Order Parameters)](#3-sweep-1d--2d-parameter-sweeps--order-parameters)
  - [4. `moments` (Higher-Order Moments & Cumulants)](#4-moments-higher-order-moments--cumulants)
  - [5. `snapshots` (Real-Space Profile Snapshots)](#5-snapshots-real-space-profile-snapshots)
  - [6. `movies` (High-Framerate Binary Animations)](#6-movies-high-framerate-binary-animations)
  - [7. `sat_check` (Saturation & Convergence Utility)](#7-sat_check-saturation--convergence-utility)
- [Python Analysis & Plotting Suite](#python-analysis--plotting-suite)
- [HPC & Cluster Workflow](#hpc--cluster-workflow)
- [Directory Structure](#directory-structure)
- [Citation](#citation)

---

## Overview & Physical Model

This repository contains a high-performance, batched pseudospectral simulation framework developed to study the non-linear dynamics, phase transitions, and synchronization phenomena of extended magnetic domain walls (DWs) driven by combined DC and AC magnetic fields.

### The Slonczewski ($u-\phi$) Model

In ferromagnetic thin films with perpendicular magnetic anisotropy (PMA), a quasi-1D domain wall can be described by its local position $u(x,t)$ and internal magnetization angle $\phi(x,t)$. In dimensionless units, the coupled equations of motion are given by:

$$\dot{u} = \frac{1}{2} \left[ \alpha^2 h(t) + \sin(2\phi) + \alpha \partial_x^2 u - \partial_x^2 \phi \right]$$

$$\dot{\phi} = \frac{1}{2} \left[ \alpha h(t) - \alpha \sin(2\phi) + \partial_x^2 u + \alpha \partial_x^2 \phi \right]$$

where:
- $\alpha$ is the dimensionless Gilbert damping parameter.
- $h(t) = h_0 + h_a \cos(\Omega t)$ is the external perpendicular magnetic field consisting of a DC component $h_0$ and an AC amplitude $h_a$ oscillating at frequency $\Omega$.

Introducing the complex coordinate $z(x,t) = u(x,t) - i\phi(x,t)$, this system maps compactly to:

$$\dot{z} = \frac{\alpha - i}{2} \left[ \alpha h(t) + i\sin\left(2\,\mathrm{Im}(z)\right) + \partial_x^2 z \right]$$

The spatial linear term $\mathcal{L}(z) = \frac{\alpha-i}{2}\partial_x^2 z$ is integrated in Fourier space using the unconditionally stable Crank-Nicolson method, while the non-linear term $\mathcal{N}(z) = \frac{\alpha-i}{2}[\alpha h(t) + i\sin(2\,\mathrm{Im}(z))]$ is stepped explicitly. All spatial derivatives and linear propagation are evaluated with spectral accuracy in double precision.

### Conversion to Physical Units (Co/Pt Multilayers)

Model variables are nondimensionalized using intrinsic material scales. For a typical thin Co/Pt multilayer film ($\alpha = 0.27$, $A_{\text{ex}} = 1.4 \times 10^{-11}\text{ J/m}$, $M_s = 9.1 \times 10^5\text{ A/m}$, $K_u = 8.4 \times 10^5\text{ J/m}^3$, $N_n = 0.016$):

| Scale / Parameter | Dimensionless Symbol | Physical Relation | Co/Pt Value |
|:---|:---:|:---|:---:|
| **DW Width** | $u_0$ | $\Delta = \sqrt{A_{\text{ex}} / (K_u - \mu_0 M_s^2 / 2)}$ | $\approx 6.6\text{ nm}$ |
| **In-Plane Length Scale** | $L_0$ | $\Delta \sqrt{2\tilde{K} / N_n}$ | $\approx 58.0\text{ nm}$ |
| **Timescale** | $T_0$ | $(1+\alpha^2)/(\gamma M_s N_n)$ | $\approx 0.33\text{ ns}$ ($1/T_0 \approx 3.03\text{ GHz}$) |
| **Walker Breakdown Field** | $H_W$ | $\alpha M_s N_n / 2$ | $\approx 2.48\text{ mT}$ |
| **Applied Drive Frequency** | $\Omega = 1.0$ | $f = \Omega / (2\pi T_0)$ | $\approx 480\text{ MHz}$ |
| **System Length ($L/L_0 = 8$)** | $L = 8$ | $8 \times L_0$ | $\approx 464\text{ nm}$ |
| **System Length ($L/L_0 = 1000$)** | $L = 1000$ | $1000 \times L_0$ | $\approx 58.0\text{ }\mu\text{m}$ (thermodynamic limit) |

### The Five Dynamical Phases

By combining spatial order/disorder of the interface position $u$ and temporal synchronization/asynchrony of the precessional angle $\phi$, the paper identifies **five distinct macroscopic dynamical regimes**:

| Acronym | Phase Name | Spatial Structure ($u$) | Precession Dynamics ($\phi$) | Coherence & Characteristics |
|:---:|:---|:---:|:---:|:---|
| **PF** | **Pinned Flat** | Flat | Pinned ($\langle\dot\phi\rangle = 0$) | Below effective Walker breakdown $h_W(h_a) \approx \|J_0(\alpha h_a / \Omega)\|$ |
| **SF** | **Synchronized Flat** | Flat | Frequency-Locked ($\langle\dot\phi\rangle = n\Omega/2$) | Coherent precession; locks to Shapiro steps; suppressed roughness |
| **SP** | **Synchronized Periodic** | Periodic Corrugation | Frequency-Locked ($\langle\dot\phi\rangle = n\Omega/2$) | Coherent precession; spatial periodicity selected by Floquet mode $\kappa^*$ |
| **NF** | **Non-synchronized Flat** | Flat | Unlocked ($\langle\dot\phi\rangle \neq n\Omega/2$) | Coherent precession; dynamically stabilized flat interface |
| **NR** | **Non-synchronized Rough** | Self-Affine Rough | Incoherent / Chaotic | Deterministic spatiotemporal chaos; 1D Edwards-Wilkinson universality |

---

## Repository Tour

The repository is structured to separate simulation kernels, reference datasets, HPC scripts, and visualization:

- `include/`: Header-only C++17 / CUDA core simulator.
  - `core/`: Backend abstractions (`cpu_backend.hpp`, `gpu_backend.cuh`) and FFT handlers (`fftw_handlers.hpp`, `cufft_handlers.cuh`).
  - `domain_wall.cuh`: Main simulation container managing state vectors, batched execution, and time stepping.
  - `stepper.cuh`: Pseudospectral Crank-Nicolson integration routines.
  - `state.cuh`: Real-space and Fourier-space grid buffers.
  - `simulation_observer.hpp` & `sweep_observer.hpp`: Real-time observable accumulation and disk I/O.
- `src/`: Executable entry points for both CUDA and CPU backends (`main.cu`, `time_evo.cu`, `sweep_main.cu`, `moments.cu`, `snapshots.cu`, `movies.cu`, `saturation_check.cu`).
- `output/`: Reference datasets produced on the CAB-CNEA cluster matching all main publication figures:
  - `output/reference_nc_ew_data/`: High-resolution time-evolution dataset for Figure 6 (Edwards-Wilkinson scaling).
  - `output/sweep_ha3_L8_L1024_979329/`: Converged 1D sweep data at $h_a=3$ for Figure 4 ($L/L_0=8$ and $1024$).
  - `output/scaling_data/`: Converged multi-batch system-size scaling dataset for Figure 8 ($L/L_0 = 1$ to $262144$).
  - `output/sweep_full_973971/`: High-resolution 2D grid sweep dataset for Figure 5 (Phase Diagram).
  - `output/floquet_transient_omega_1.000_L_1024.bin`: Binary Floquet monodromy stability data.
- `python/`: Publication-grade plotting scripts designed for Physical Review B (PRB) style.
- `scripts/`: Production SLURM array submission and local benchmarking shell scripts.
- `paper/`: Complete LaTeX source (`main.tex`, `refs.bib`) and figures for the manuscript.

---

## Paper Figure Reproduction Guide

All publication figures from the paper can be generated directly using the scripts in `python/` and the pre-computed reference datasets in `output/`, or re-simulated from scratch.

### Figure 2: Shapiro Steps & Average Velocities in Rigid DWs

- **Physical Description**: Steady-state average internal angular velocity $\langle \dot{\psi}_1 \rangle_\tau$ and translational velocity $\langle \dot{q}_1 \rangle_\tau$ vs. DC field $h_0$ for $h_a=0$ and $h_a=5$. Compares numerical 4th-order Runge-Kutta integration of the rigid Walker ODE against 1st- and 2nd-order analytical perturbation theory (Bessel function expansions).
- **Reproduction Command**:
  ```bash
  python3 python/plot_figure4_wall_metrics.py --data-dir output/sweep_ha3_L8_L1024_979329
  ```
  *(The rigid-wall Runge-Kutta solver is embedded into `plot_figure4_wall_metrics.py` and `plot_paper_figure.py`, generating both numerical curves and perturbation boundaries).*

### Figure 3: Floquet Linear Stability Diagrams

- **Physical Description**: Linear stability spectrum of flexural spatial modes $\kappa$ around the synchronized flat DW for $h_a=2$ and $h_a=3$ ($\alpha=0.27$, $\Omega=1$). Shows the largest Floquet multiplier modulus $|\mu_\kappa| > 1$ calculated by integrating the monodromy matrix over one period.
- **Reproduction Command**:
  ```bash
  python3 python/plot_prb_publication.py \
    --data-dir output/sweep_full_973971 \
    --floquet-bin output/floquet_transient_omega_1.000_L_1024.bin
  ```

### Figure 4: Flexible Extended DW Dynamics vs $h_0$ ($L/L_0=8$ and $1024$)

- **Physical Description**: Mean DW velocity $\langle\dot{u}\rangle_{x,t}$, precessional velocity $\langle\dot{\phi}\rangle_{x,t}$, and steady-state roughness $\langle w^2\rangle_t$ across $h_0$ at fixed $h_a=3$, comparing a small system ($L/L_0=8$), an extended thermodynamic system ($L/L_0=1024$), and the zero-dimensional rigid wall.
- **Reference Dataset**: `output/sweep_ha3_L8_L1024_979329/`
- **Reproduction Command**:
  ```bash
  python3 python/plot_figure4_wall_metrics.py \
    --data-dir output/sweep_ha3_L8_L1024_979329 \
    --out-dir figures
  ```
  Outputs `figures/sweep_ha_3.003_caption_only.pdf` and `figures/sweep_ha_3.003_with_legend.pdf`.
- **Re-running the Simulation**:
  ```bash
  ./build/bin/sweep 512 256.0 0.27 1.0 0.019634954 131072 12566.370614 4.0 3.0 3.0 1 "results.txt" gpu
  ```
  *(Or execute `scripts/sweep_knl_ha3.sh` for cluster submission).*

### Figure 5: Full Dynamical Phase Diagram in the $(h_0, h_a)$ Plane

- **Physical Description**: The 5-phase macroscopic dynamical phase diagram (PF, SF, SP, NF, NR) over $0 \le h_0 \le 10$ and $0 \le h_a \le 20$ in the thermodynamic limit ($L/L_0 = 1000$), with solid contiguous phase regions, analytical Shapiro step boundaries, and Floquet instability overlays.
- **Reference Dataset**: `output/sweep_full_973971/` and `output/floquet_transient_omega_1.000_L_1024.bin`
- **Reproduction Command**:
  ```bash
  python3 python/plot_prb_publication.py \
    --data-dir output/sweep_full_973971 \
    --floquet-bin output/floquet_transient_omega_1.000_L_1024.bin \
    -o figures/ \
    --export-paper
  ```
  Generates `figures/prb_phase_diagram.pdf`, `figures/prb_rugosity.pdf`, and `figures/prb_strobo_variance.pdf`.
- **Interactive Inspection with `plot_sweep_local.py`**:
  ```bash
  # View phase diagram and rugosity directly
  python3 python/plot_sweep_local.py output/sweep_full_973971 -m phase_diagram rugosity

  # Extract 1D slice at ha = 3.0
  python3 python/plot_sweep_local.py output/sweep_full_973971 --slice-ha 3.0
  ```

### Figure 6: Universal Edwards-Wilkinson Scaling Collapse of the NR Phase

- **Physical Description**: Non-equilibrium roughening dynamics in the chaotic NR regime ($h_0=5, h_a=3, \alpha=0.27, \Omega=1, L/L_0=1000$). Proves that the spatiotemporally chaotic DW belongs to the 1D Edwards-Wilkinson universality class:
  - Structure factor power-law decay $S_\kappa \propto \kappa^{-2}$.
  - Interface roughness growth $w^2 \propto t^{1/2}$.
  - Master scaling curve collapse: $S_\kappa t^{-1}$ vs. $\kappa t^{1/2}$ ($\zeta = 1/2$, $z = 2$).
- **Reference Dataset**: `output/reference_nc_ew_data/`
- **Reproduction Command**:
  ```bash
  python3 python/plot_figure6_nc_ew.py output/reference_nc_ew_data --layout inset
  ```
  Additional layouts available via `--layout`: `side_by_side`, `stacked`, `compensated`, `wide_row`, or `all`.
- **Re-running the Simulation**:
  ```bash
  # High-throughput cluster run via SLURM:
  sbatch scripts/slurm_generate_nc_data.sh

  # Or standalone via time_evo:
  ./build/bin/time_evo 65536 8192.0 0.27 5.0 3.0 1.0 0.049087385 28 "output/time_data" 0.05 42 gpu
  ```

### Figure 7: Spatiotemporal Cascade of the SP Phase

- **Physical Description**: Real-space profile snapshots of a domain wall section in the steady-state SP phase ($h_0=4, h_a=3, \alpha=0.27, \Omega=1$) over two driving periods ($4\pi/\Omega$), showing stable breathing and peak/valley inversion.
- **Simulation Command**:
  ```bash
  ./build/bin/snapshots 1024 1 512.0 0.27 4.0 3.0 1.0 0.019634954 10000.0 12.566 2.0 1 "output/sp_snap" 0.05 42 gpu
  ```

### Figure 8: Finite-Size Scaling of the SP Phase ($w^2$ and $\kappa_{\mathrm{peak}}$ vs $L/L_0$)

- **Physical Description**: Finite-size scaling analysis demonstrating that in the SP phase, steady-state interface roughness $\langle w^2 \rangle_t$ and the dominant wavenumber $\kappa_{\mathrm{peak}}$ converge to system-size independent constants for large $L/L_0$. In the thermodynamic limit, $\kappa_{\mathrm{peak}}$ saturates at the maximum Floquet growth mode $\kappa^* \approx 0.833$. For small systems ($L/L_0 \le 64$), lack of discrete modes in the instability band keeps the wall flat.
- **Reference Dataset**: `output/scaling_data/`
- **Reproduction Command**:
  ```bash
  python3 python/plot_figure8_sc_scaling.py --data-dir output/scaling_data
  ```
  Generates `paper/figuras/SC_scaling.pdf` and `figures/SC_scaling.pdf`.
- **Re-running the Simulation**:
  ```bash
  # Execute array job across system sizes L = 2^0 to 2^18:
  sbatch scripts/scaling_analysis.sh
  ```

### Figure 9 (Appendix): Finite-Size Phase Diagram for $L/L_0=8$

- **Physical Description**: 5-phase diagram for a small system ($L/L_0 = 8$) illustrating how discrete spatial mode quantization suppresses the chaotic NR phase and shifts the critical field $h_c$ from $\approx 7$ down to $\approx 3.8$.
- **Reproduction Command**:
  ```bash
  python3 python/plot_prb_publication.py \
    --data-dir output/sweep_ha3_L8_L1024_979329/L_8 \
    -o figures/
  ```

---

## Prerequisites & Dependencies

### C++ Compiler & Libraries
- **CMake** $\ge 3.19$
- **C++17** compliant compiler (GCC $\ge 8.2$, Clang $\ge 11$, or MSVC)
- **CUDA Toolkit** $\ge 11.0$ (12.x fully supported) - *Required for GPU acceleration*
- **FFTW3** with OpenMP support (`libfftw3`, `libfftw3_omp`) - *Always required*
- **OpenMP** runtime library - *Always required*
- **ccache** (Optional, speeds up recompilation)
- **Doxygen** (Optional, for generating code documentation)

### Ubuntu / Debian Package Installation
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libfftw3-dev libfftw3-omp-dev libomp-dev ccache
```

### Fedora / RHEL / Rocky Linux Package Installation
```bash
sudo dnf install -y cmake gcc-c++ fftw-devel fftw-libs-openmp libgomp ccache
```

### Python Environment
- Python 3.8+
- Required packages:
  ```bash
  pip install numpy matplotlib pandas
  ```

---

## Compilation & Build

### Quick Automated Build Scripts

Two ready-to-run helper scripts are provided:

```bash
# 1. Build locally on CPU (treats .cu as C++17 with OpenMP + FFTW3):
./compile_local.sh

# 2. Build for GPU cluster (NVCC + cuFFT + FFTW3):
./compile_cluster.sh
```

### Manual CMake Build (GPU + CPU)

Compiles both backends with maximum optimization (`-O3`, `--use_fast_math`, interprocedural optimization):

```bash
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Executables are placed in `build/bin/`.

### Manual CMake Build (CPU Only)

For systems without an NVIDIA GPU:

```bash
mkdir -p build_local && cd build_local
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Executables are placed in `build_local/bin/`.

### Verifying the Build Locally

You can verify that all 6 executables compile and run properly using the automated CPU test suite:

```bash
./scripts/test_local_cpu.sh
```
This executes short test instances of `main`, `moments`, `snapshots`, `movies`, `sweep`, and `sat_check`.

---

## Simulation Executables Reference

All binaries are located in `build/bin/` (or `build_local/bin/`). Every executable accepts an optional `[backend]` parameter as its final argument: `gpu` (default) or `cpu`.

### 1. `main` (Standard Time Evolution & Observables)

Simulates time evolution and writes transient metrics, time series, and spatial height histograms. Supports batched execution.

```bash
./bin/main <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
           <transient_time> <averaging_time> <window_time> <subsample_rate> \
           <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

| Argument | Description | Default / Example |
|:---|:---|:---:|
| `size` | Number of spatial grid points ($N = 2^k$) | `1024` |
| `batch` | Number of concurrent independent systems | `1` |
| `L` | Physical domain length ($L/L_0$) | `512.0` |
| `alpha` | Gilbert damping constant $\alpha$ | `0.27` |
| `h0` | DC magnetic field $h_0$ | `4.0` |
| `ha` | AC magnetic field amplitude $h_a$ | `3.0` |
| `omega` | AC angular frequency $\Omega$ | `1.0` |
| `dt` | Time step $\delta t$ | `0.019634954` |
| `transient_time` | Initial transient duration before data collection | `10000.0` |
| `averaging_time` | Evaluation duration for time-averaged metrics | `10000.0` |
| `window_time` | Moving average window | `100.0` |
| `subsample_rate` | Stride for saving time series entries | `16` |
| `output_prefix` | Base filename prefix for output text files | `"output/run1"` |
| `eps` | Initial spatial perturbation amplitude | `0.05` |
| `seed` | RNG seed for perturbation | `42` |
| `backend` | Compute backend (`gpu` or `cpu`) | `gpu` |

**Generated Outputs**:
- `<prefix>_transient_metrics.txt`: Rolling variance $\langle w^2 \rangle_t$, velocities $\dot{u}$, $\dot{\phi}$, and spectral entropy.
- `<prefix>_time_series.txt`: Subsampled instantaneous mean position $\langle u \rangle$ and velocity.
- `<prefix>_power_spectrum.txt`: Time-averaged power spectrum $S_\kappa$ across Fourier modes.
- `<prefix>_height_distribution.txt`: Probability density function of interface fluctuations.

### 2. `time_evo` (Logarithmic Sampling & Surface Growth)

Samples system properties at powers-of-two steps ($t = 2^p \times \delta t$), ideal for tracking surface growth scaling exponents across multiple decades of time.

```bash
./bin/time_evo <size> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> \
               <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

- `max_power`: Maximum power of two ($2^{\text{max\_power}}$ total steps). Example: `28` yields $\approx 2.68 \times 10^8$ steps.
- Outputs rolling roughness `<prefix>_transient_w2.txt` and power spectra `<prefix>_ps_step_<step>.txt` at each power of two.

### 3. `sweep` (1D / 2D Parameter Sweeps & Order Parameters)

Performs sweeps over $h_a$ at a given $h_0$, computing steady-state velocities $\langle\dot{u}\rangle$, $\langle\dot{\phi}\rangle$, roughness $\langle w^2 \rangle_t$, and stroboscopic variance $\langle \sigma^2 \rangle_x$:

```bash
./bin/sweep <size> <L> <alpha> <omega> <dt> <transient_time> <eval_time> \
            <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> [backend=gpu]
```

### 4. `moments` (Higher-Order Moments & Cumulants)

Computes statistical central moments ($\mu_2, \mu_3, \mu_4$), skewness, and kurtosis of the domain wall:

```bash
./bin/moments <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <total_time> \
              <start_time> <num_checkpoints> <num_sq_checkpoints> <output_prefix> \
              [eps=0.05] [seed=42] [backend=gpu]
```

### 5. `snapshots` (Real-Space Profile Snapshots)

Outputs real-space configurations $u(x,t)$ and $\phi(x,t)$ at specified intervals:

```bash
./bin/snapshots <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                <transient_time> <snapshot_time> <window_time> <snapshot_rate> \
                <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

### 6. `movies` (High-Framerate Binary Animations)

Streams raw binary states to disk for creating smooth video animations of domain wall motion:

```bash
./bin/movies <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <t_max> \
             <window_time> <movie_start> <movie_end> <movie_rate> \
             <output_prefix> [eps=0.05] [seed=42] [backend=gpu]
```

### 7. `sat_check` (Saturation & Convergence Utility)

Tests interface growth until roughness saturates, useful for verifying finite-size crossover times:

```bash
./bin/sat_check <size> <L> <alpha> <dt> [backend=gpu]
```

---

## Python Analysis & Plotting Suite

All scripts in `python/` format figures to match the Physical Review B two-column standard (single-column width: 3.375 in, double-column: 7.0 in, STIX/Times fonts):

- `plot_figure6_nc_ew.py`: Generates Figure 6 (universal Edwards-Wilkinson scaling collapse, $S_\kappa \propto \kappa^{-2}$, $w^2 \propto t^{1/2}$, and $S_\kappa t^{-1}$ vs. $\kappa t^{1/2}$).
  ```bash
  python3 python/plot_figure6_nc_ew.py output/reference_nc_ew_data --layout inset
  ```
- `plot_figure4_wall_metrics.py`: Generates Figure 4 (flexible extended domain wall velocities and roughness vs. $h_0$ at $h_a=3$).
  ```bash
  python3 python/plot_figure4_wall_metrics.py --data-dir output/sweep_ha3_L8_L1024_979329
  ```
- `plot_figure8_sc_scaling.py`: Generates Figure 8 (finite-size scaling analysis of the SP phase, showing convergence of roughness $\langle w^2 \rangle_t$ and peak wavenumber $\kappa_{\mathrm{peak}} \to \kappa^*$).
  ```bash
  python3 python/plot_figure8_sc_scaling.py --data-dir output/scaling_data
  ```
- `plot_prb_publication.py`: Official publication plotter for 2D sweeps. Generates 5-phase diagrams, log-scale rugosity heatmaps, and stroboscopic variance with vector boundary overlays.
  ```bash
  python3 python/plot_prb_publication.py --data-dir output/sweep_full_973971 --floquet-bin output/floquet_transient_omega_1.000_L_1024.bin
  ```
- `plot_sweep_local.py`: CLI-friendly tool for slicing, filtering, zooming, and previewing 1D and 2D parameter sweeps with automatic caching:
  ```bash
  python3 python/plot_sweep_local.py output/sweep_full_973971 -m phase_diagram --slice-ha 3.0
  ```
- `plot_time_evo.py`: Generates time series, power spectra, height distributions, and animated `.mp4` videos from `time_evo` or `main` data.
  ```bash
  python3 python/plot_time_evo.py output/time_data --outdir figures
  ```
- `plot_scaling.py`: Analyzes Family-Vicsek scaling and dynamic exponents across multiple system lengths.

---

## HPC & Cluster Workflow

For large-scale simulations on high-performance computing clusters (SLURM / SGE):

- `scripts/sweep.sh`: SLURM job array script for full $(h_0, h_a)$ 2D parameter sweeps across GPU nodes.
- `scripts/scaling_analysis.sh`: SLURM job array script simulating system sizes $L = 2^0$ to $2^{18}$ (used for Figure 8).
- `scripts/slurm_generate_nc_data.sh`: 48-hour GPU job generating the high-precision Edwards-Wilkinson dataset (Figure 6).
- `scripts/slurm_plot_figure6.sh`: Batch script to plot Figure 6 on compute nodes.
- `sync_cluster.sh`: Rsync-based utility to sync code and results between local development environments and the cluster:
  ```bash
  ./sync_cluster.sh push   # Sync code to remote cluster
  ./sync_cluster.sh pull   # Pull simulation outputs from remote cluster
  ```

---

## Directory Structure

```text
.
├── CMakeLists.txt                # Unified CMake build configuration (CUDA + C++17)
├── compile_local.sh              # Quick compilation for CPU backend
├── compile_cluster.sh            # Quick compilation for GPU cluster backend
├── sync_cluster.sh               # Rsync synchronization with remote cluster
├── include/                      # C++17 / CUDA Header-only core simulator
│   ├── core/                     # Backend abstraction layer
│   │   ├── cpu_backend.hpp       # OpenMP + FFTW3 backend
│   │   ├── gpu_backend.cuh       # CUDA + cuFFT backend
│   │   ├── fft-handlers/         # Wrappers for cuFFT and FFTW
│   │   ├── cuda_precision.cuh    # Precision traits for CUDA (double)
│   │   └── fftw_precision.hpp    # Precision traits for FFTW (double)
│   ├── domain_wall.cuh           # Main simulation class with batched execution
│   ├── simulation_observer.hpp   # Real-time data accumulation and I/O
│   ├── sweep_observer.hpp        # Stroboscopic variance and order parameter I/O
│   ├── state.cuh                 # Physical state buffers (real & Fourier space)
│   ├── stepper.cuh               # Pseudospectral Crank-Nicolson stepper
│   └── sim_utils.hpp             # Helper math, RNG, and reduction kernels
├── src/                          # C++ / CUDA Executable entry points
│   ├── main.cu                   # Standard time evolution & observable analysis
│   ├── time_evo.cu               # Logarithmic time-stepping for scaling analysis
│   ├── sweep_main.cu             # 1D/2D parameter sweep engine
│   ├── moments.cu                # Central moments & cumulant calculations
│   ├── snapshots.cu              # Real-space profile snapshots
│   ├── movies.cu                 # High-framerate binary movie output
│   └── saturation_check.cu       # Roughness saturation & convergence utility
├── output/                       # Pre-computed reference datasets for publication figures
│   ├── reference_nc_ew_data/     # Figure 6: EW scaling data (L/L0 = 1000, 2^28 steps)
│   ├── sweep_ha3_L8_L1024_979329/ # Figure 4: ha=3 sweeps for L/L0=8 and 1024
│   ├── scaling_data/             # Figure 8: SP phase scaling data (L/L0 = 1 to 262144)
│   ├── sweep_full_973971/        # Figure 5: Full 2D (h0, ha) sweep data
│   └── floquet_transient_*.bin   # Binary Floquet stability monodromy data
├── python/                       # PRB publication-grade plotting suite
│   ├── plot_figure4_wall_metrics.py  # Figure 4 plotter (Flexible DW dynamics)
│   ├── plot_figure6_nc_ew.py         # Figure 6 plotter (Universal EW scaling collapse)
│   ├── plot_figure8_sc_scaling.py    # Figure 8 plotter (SP phase finite-size scaling)
│   ├── plot_prb_publication.py       # Figure 5 plotter (5-phase diagram & Floquet overlay)
│   ├── plot_sweep_local.py           # CLI sweep inspector with 1D slicing & caching
│   ├── plot_time_evo.py              # Time series, spectra, histograms & MP4 video exporter
│   └── plot_scaling.py               # Family-Vicsek scaling collapse analysis
├── scripts/                      # SLURM and local execution shell scripts
│   ├── test_local_cpu.sh         # Fast automated sanity check of all CPU binaries
│   ├── slurm_generate_nc_data.sh # SLURM job for Figure 6 simulation
│   ├── scaling_analysis.sh       # SLURM array job for Figure 8 scaling
│   ├── sweep.sh                  # SLURM array job for full 2D phase diagram sweep
│   └── sweep_local.sh            # Local 1D parameter sweep execution
└── paper/                        # Publication manuscript LaTeX source and figures
    ├── main.tex                  # RevTeX 4-1 manuscript
    ├── refs.bib                  # BibTeX bibliography
    └── figuras/                  # Manuscript figure assets (PDF / PNG)
```

---

## Citation

If you use this simulation code or find our results useful in your research, please cite our paper:

```bibtex
@article{perezherrera2026synchronization,
  title   = {Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls},
  author  = {P{\'e}rez Herrera, Luz and Pusiol, E. F. and Kolton, A. B.},
  journal = {Physical Review B},
  year    = {2026},
  note    = {Under review}
}

@misc{github_luz,
  author       = {Luz P{\'e}rez Herrera},
  title        = {{DW-uphi-simulator: High-performance pseudospectral simulator for 1D domain wall dynamics}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/lightph/dw-uphi-simulator}}
}
```

---

## Acknowledgments

Simulations were performed using the high-performance computing cluster at the *Gerencia de Física, Centro Atómico Bariloche (CAB-CNEA / SNCAD, Argentina)*. We thank T. Giamarchi for fruitful discussions.
