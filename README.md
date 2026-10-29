# Domain Wall Simulator: Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5%2B-red.svg)](https://www.openmp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.19%2B-orange.svg)](https://cmake.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official C++17 / CUDA high-performance simulation framework, analytical perturbation solvers, Floquet monodromy stability analyzer, and complete publication reproduction suite for:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, and A. B. Kolton  
> *Instituto Balseiro (UNCUYO) & Centro Atómico Bariloche (CNEA-CONICET), San Carlos de Bariloche, Río Negro, Argentina.*  
> GitHub Repository: [https://github.com/lightph/dw-uphi-simulator](https://github.com/lightph/dw-uphi-simulator)

---

## Table of Contents

- [Overview & Physical Framework](#overview--physical-framework)
  - [The Slonczewski ($u-\phi$) Equations](#the-slonczewski-u--phi-equations)
  - [Conversion to Physical Units (Co/Pt Multilayers)](#conversion-to-physical-units-copt-multilayers)
  - [The Five Macroscopic Dynamical Regimes](#the-five-macroscopic-dynamical-regimes)
- [Repository Architecture: The Three Pillars](#repository-architecture-the-three-pillars)
  - [1. Rigid Wall Dynamics (`rigid_wall/`)](#1-rigid-wall-dynamics-rigid_wall)
  - [2. Floquet Stability Analysis (`floquet/`)](#2-floquet-stability-analysis-floquet)
  - [3. Full PDE Simulations (`pde/`)](#3-full-pde-simulations-pde)
  - [4. Publication Reproduction Suite (`reproduction/`)](#4-publication-reproduction-suite-reproduction)
- [Master Paper Figure Reproduction Guide](#master-paper-figure-reproduction-guide)
  - [Summary Map of Publication Figures](#summary-map-of-publication-figures)
  - [One-Click Recreation of All Figures](#one-click-recreation-of-all-figures)
  - [Figure 1: Domain Wall Coordinates & Schematic](#figure-1-domain-wall-coordinates--schematic)
  - [Figure 2: Rigid Wall Synchronization & Shapiro Steps](#figure-2-rigid-wall-synchronization--shapiro-steps)
  - [Figure 3: Floquet Linear Stability Diagrams](#figure-3-floquet-linear-stability-diagrams)
  - [Figure 4: Flexible Extended DW Dynamics vs $h_0$ ($h_a=3$)](#figure-4-flexible-extended-dw-dynamics-vs-h_0-h_a3)
  - [Figure 5: Full Dynamical Phase Diagram ($L/L_0=256$)](#figure-5-full-dynamical-phase-diagram-ll_0256)
  - [Figure 6: Universal Edwards-Wilkinson Scaling Collapse (NR Phase)](#figure-6-universal-edwards-wilkinson-scaling-collapse-nr-phase)
  - [Figure 7: Spatiotemporal Cascade of the SP Phase](#figure-7-spatiotemporal-cascade-of-the-sp-phase)
  - [Figure 8: Finite-Size Scaling of the SP Phase](#figure-8-finite-size-scaling-of-the-sp-phase)
  - [Figure 9: Small-System Phase Diagram ($L/L_0=8$)](#figure-9-small-system-phase-diagram-ll_08)
- [Prerequisites & Build System](#prerequisites--build-system)
  - [System Dependencies](#system-dependencies)
  - [Automated Quick Builds](#automated-quick-builds)
  - [Manual CMake Build (CPU Only)](#manual-cmake-build-cpu-only)
  - [Manual CMake Build (CUDA GPU Acceleration)](#manual-cmake-build-cuda-gpu-acceleration)
- [Executables & Module Reference](#executables--module-reference)
  - [Rigid Wall Binaries](#rigid-wall-binaries)
  - [Floquet Stability Binaries](#floquet-stability-binaries)
  - [PDE Simulation Binaries](#pde-simulation-binaries)
- [Verifying and Compiling the LaTeX Manuscript](#verifying-and-compiling-the-latex-manuscript)
- [Citation](#citation)

---

## Overview & Physical Framework

This repository provides a self-contained, publication-grade computational ecosystem to investigate non-linear dynamics, frequency locking (Shapiro steps), pattern formation, and non-equilibrium roughening transitions in precessing magnetic domain walls (DWs) subjected to combined DC and AC magnetic fields:

$$h(t) = h_0 + h_a \cos(\Omega t)$$

### The Slonczewski ($u-\phi$) Equations

In thin ferromagnetic films with perpendicular magnetic anisotropy (PMA), the collective dynamics of an extended 1D domain wall are described by its local interface displacement $u(x,t)$ and internal magnetization angle $\phi(x,t)$. In dimensionless form:

$$\dot{u} = \frac{1}{2} \left[ \alpha^2 h(t) + \sin(2\phi) + \alpha \partial_x^2 u - \partial_x^2 \phi \right]$$

$$\dot{\phi} = \frac{1}{2} \left[ \alpha h(t) - \alpha \sin(2\phi) + \partial_x^2 u + \alpha \partial_x^2 \phi \right]$$

where $\alpha$ is the dimensionless Gilbert damping parameter. Introducing the complex coordinate $z(x,t) = u(x,t) - i\phi(x,t)$, the system maps compactly to:

$$\dot{z} = \frac{\alpha - i}{2} \left[ \alpha h(t) + i\sin\left(2\,\mathrm{Im}(z)\right) + \partial_x^2 z \right]$$

The spatial linear diffusion and dispersion $\mathcal{L}(z) = \frac{\alpha-i}{2}\partial_x^2 z$ is integrated in Fourier space via the unconditionally stable Crank-Nicolson method, while non-linear gyrotropic terms are integrated explicitly in real space with double-precision spectral accuracy.

### Conversion to Physical Units (Co/Pt Multilayers)

Dimensionless variables map directly to physical observables in experimental thin films (e.g. Co/Pt multilayers with $\alpha = 0.27$, $A_{\text{ex}} = 1.4 \times 10^{-11}\text{ J/m}$, $M_s = 9.1 \times 10^5\text{ A/m}$, $K_u = 8.4 \times 10^5\text{ J/m}^3$, $N_n = 0.016$):

| Scale / Parameter | Dimensionless Symbol | Physical Relation | Co/Pt Physical Value |
|:---|:---:|:---|:---:|
| **DW Width** | $u_0$ | $\Delta = \sqrt{A_{\text{ex}} / (K_u - \mu_0 M_s^2 / 2)}$ | $\approx 6.6\text{ nm}$ |
| **In-Plane Length Scale** | $L_0$ | $\Delta \sqrt{2\tilde{K} / N_n}$ | $\approx 58.0\text{ nm}$ |
| **Characteristic Timescale** | $T_0$ | $(1+\alpha^2)/(\gamma M_s N_n)$ | $\approx 0.33\text{ ns}$ ($1/T_0 \approx 3.03\text{ GHz}$) |
| **Walker Breakdown Field** | $H_W$ | $\alpha M_s N_n / 2$ | $\approx 2.48\text{ mT}$ |
| **Applied Drive Frequency** | $\Omega = 1.0$ | $f = \Omega / (2\pi T_0)$ | $\approx 480\text{ MHz}$ |
| **System Length ($L/L_0 = 8$)** | $L = 8$ | $8 \times L_0$ | $\approx 464\text{ nm}$ |
| **System Length ($L/L_0 = 256$)** | $L = 256$ | $256 \times L_0$ | $\approx 14.8\text{ }\mu\text{m}$ |
| **Thermodynamic Limit ($L/L_0 = 1000$)** | $L = 1000$ | $1000 \times L_0$ | $\approx 58.0\text{ }\mu\text{m}$ |

### The Five Macroscopic Dynamical Regimes

The coupling between temporal precession locking and spatial flexural stability produces **five distinct macroscopic dynamical phases**:

| Acronym | Phase Name | Interface Roughness ($w^2$) | Precession Dynamics ($\langle\dot\phi\rangle$) | Physical Signature |
|:---:|:---|:---:|:---:|:---|
| **PF** | **Pinned Flat** | Flat ($w^2 = 0$) | Pinned ($\langle\dot\phi\rangle = 0$) | Below dynamic Walker threshold $h_W(h_a) \approx \|J_0(\alpha h_a / \Omega)\|$ |
| **SF** | **Synchronized Flat** | Flat ($w^2 = 0$) | Frequency-Locked ($\langle\dot\phi\rangle = n\Omega/2$) | Coherent uniform translation on Shapiro steps |
| **SP** | **Synchronized Periodic** | Periodic Corrugation ($w^2 > 0$) | Frequency-Locked ($\langle\dot\phi\rangle = n\Omega/2$) | Spatiotemporal pattern formation selected by Floquet mode $\kappa^*$ |
| **NF** | **Non-synchronized Flat** | Flat ($w^2 = 0$) | Unlocked ($\langle\dot\phi\rangle \neq n\Omega/2$) | Dynamically stabilized precessing flat interface |
| **NR** | **Non-synchronized Rough** | Self-Affine Rough ($w^2 \sim t^{1/2}$) | Incoherent / Chaotic | Deterministic spatiotemporal chaos; 1D Edwards-Wilkinson universality |

---

## Repository Architecture: The Three Pillars

The codebase is organized into three clean, modular physical tiers, complemented by an automated reproduction pipeline:

```text
.
├── rigid_wall/                  # PILLAR 1: 0D Rigid Wall ODE & Perturbation Theory
│   ├── include/rigid_wall.hpp   # Header-only C++ ODE stepper, Bessel quadrature, perturbation theory
│   ├── src/                     # OpenMP CPU & CUDA GPU parameter sweep executables
│   ├── python/                  # Pure NumPy ODE integrator, Bessel quadrature, Fig. 2 plotter
│   └── README.md                # Module architecture & mathematical formulation
├── floquet/                     # PILLAR 2: Linear Stability & Floquet Monodromy Analysis
│   ├── include/floquet.hpp      # Header-only C++ Floquet monodromy integrator & eigenvalue solver
│   ├── src/                     # OpenMP CPU & CUDA GPU (h0, ha, kappa) sweep executables
│   ├── python/                  # Vectorized Python solver, Arnold tongue mapper, Fig. 3 plotter
│   └── README.md                # Floquet theory documentation & matrix derivation
├── pde/                         # PILLAR 3: Full 1D Pseudospectral PDE Simulator
│   ├── include -> ../include    # Core C++17 / CUDA simulation kernels & Crank-Nicolson stepper
│   ├── src -> ../src            # 7 production binaries (main, time_evo, sweep, snapshots, etc.)
│   └── README.md                # PDE usage reference, parameter guides & physics manual
├── reproduction/                # PILLAR 4: Publication Figure Reproduction Suite
│   ├── plot_figure1_schematic.py
│   ├── plot_figure2_rigid_wall.py
│   ├── plot_figure3_floquet.py
│   ├── plot_figure4_wall_metrics.py
│   ├── plot_figure5_phase_diagram.py
│   ├── plot_figure6_nc_ew.py
│   ├── plot_figure7_sp_cascade.py
│   ├── plot_figure8_sc_scaling.py
│   ├── plot_figure9_phase_diagram_L8.py
│   ├── reproduce_all_figures.py # Master CLI executing all 9 figure pipelines
│   └── README.md                # Figure-by-figure reproduction reference
├── output/                      # Pre-computed high-resolution cluster datasets
├── figures/                     # Generated PDF and PNG figures
└── paper/                       # Complete LaTeX publication manuscript (RevTeX 4-1)
    ├── main.tex                 # Manuscript source
    ├── Makefile                 # Automated verification & compilation
    └── figuras/                 # Target directory for publication figures
```

### 1. Rigid Wall Dynamics (`rigid_wall/`)
- Analyzes the 0D Walker limit without spatial dependence ($\partial_x^2 = 0$).
- Transformed angle $\psi_1 = 2\phi + \arctan\alpha$ in rescaled time $\tau = \frac{\Omega}{\alpha^2+1}t$:
  $$\dot{\psi}_1 = \frac{1}{\Omega}[h(\tau) - \sin\psi_1]$$
- Relates translational velocity $\langle\dot q_1\rangle_\tau$ directly to angular velocity $\langle\dot\psi_1\rangle_\tau$:
  $$\langle\dot q_1\rangle_\tau = \frac{\alpha^2+1}{2\Omega}h_0 - \frac{1}{2\alpha}\langle\dot\psi_1\rangle_\tau$$
- Solves Shapiro step boundaries using high-order analytical Bessel expansions $J_n(\alpha h_a / \Omega)$ and numerical Runge-Kutta integration.
- Includes OpenMP CPU (`rigid_wall_sweep_cpu`) and CUDA GPU (`rigid_wall_sweep_gpu`) engines.

### 2. Floquet Stability Analysis (`floquet/`)
- Evaluates the linear stability of flexural perturbations $Q_\kappa(\tau)$ and $\Phi_\kappa(\tau)$ around the periodic flat solution:
  $$\dot{Q}_\kappa = \frac{1}{\Omega}\left[-\alpha\frac{\kappa^2}{2}Q_\kappa + \left(\frac{\kappa^2}{2} + \cos\psi_1(\tau)\right)\Phi_\kappa\right]$$
  $$\dot{\Phi}_\kappa = \frac{1}{\Omega}\left[-\frac{\kappa^2}{2}Q_\kappa - \alpha\left(\frac{\kappa^2}{2} + \cos\psi_1(\tau)\right)\Phi_\kappa\right]$$
- Integrates the fundamental $2\times 2$ monodromy matrix over period $T = 2\pi / \Omega$.
- Mode $\kappa$ is linearly unstable if the largest Floquet multiplier modulus satisfies $|\mu_\kappa| > 1$.
- Computes Arnold tongue boundaries and maps the maximum unstable mode $\kappa^*$.
- Implements OpenMP CPU (`floquet_sweep_cpu`) and CUDA GPU (`floquet_sweep_gpu`) grid sweeps.

### 3. Full PDE Simulations (`pde/`)
- Integrates the complete 1D non-linear Slonczewski system $z(x,t) = u - i\phi$ for arbitrarily large systems ($L/L_0$ up to $262,144$).
- Unconditionally stable Crank-Nicolson pseudospectral integration on CPU (FFTW3 + OpenMP) and GPU (cuFFT + CUDA).
- Includes 7 specialized executables:
  - `main`: Observable time-series, interface roughness, and spatial histograms.
  - `time_evo`: Logarithmic temporal sampling ($t = 2^p \delta t$) for non-equilibrium scaling.
  - `sweep`: Automated 1D/2D parameter sweeps over $(h_0, h_a)$ with stroboscopic order parameters.
  - `moments`: Higher-order statistical central moments ($\mu_2, \mu_3, \mu_4$), skewness, and kurtosis.
  - `snapshots`: High-precision spatial profile snapshots $u(x,t)$ and $\phi(x,t)$.
  - `movies`: Streaming binary frame writer for high-framerate video generation.
  - `sat_check`: Dynamic saturation detector for finite-size crossover times.

### 4. Publication Reproduction Suite (`reproduction/`)
- Standalone, modular Python scripts dedicated to recreating every figure in the publication from scratch or from pre-computed cluster datasets.
- Fully compatible with Physical Review B styling (RevTeX column widths, STIX fonts, vector PDF outputs).
- One-click master CLI: `reproduce_all_figures.py`.

---

## Master Paper Figure Reproduction Guide

### Summary Map of Publication Figures

| Fig # | Paper Label | Subject / Description | Script in `reproduction/` | Output PDF in `figures/` & `paper/figuras/` |
|:---:|:---|:---|:---|:---|
| **1** | `fig:snap` | DW Coordinate System & Real-Space Schematic | `plot_figure1_schematic.py` | `snap_system_345.pdf` |
| **2** | `fig:ej_walker` | Rigid Wall Shapiro Steps & Velocities | `plot_figure2_rigid_wall.py` | `figure2.pdf` |
| **3** | `fig:sync_stability` | Floquet Linear Stability Diagrams ($h_a=2, 3$) | `plot_figure3_floquet.py` | `figure3.pdf` |
| **4** | `fig:wall_metrics` | Flexible DW Dynamics vs $h_0$ ($L/L_0=8, 1024$) | `plot_figure4_wall_metrics.py` | `sweep_ha_3.pdf` |
| **5** | `fig:fases` | Full Dynamical Phase Diagram ($L/L_0=256$) | `plot_figure5_phase_diagram.py` | `phasediagram_L256.pdf` |
| **6** | `fig:estructura` | Edwards-Wilkinson Scaling Collapse (NR Phase) | `plot_figure6_nc_ew.py` | `NC_EW_inset.pdf` |
| **7** | `fig:sc_cascade` | SP Phase Spatiotemporal Profile Cascade | `plot_figure7_sp_cascade.py` | `sc_cascade.pdf` |
| **8** | `fig:estructuraSCvsL` | SP Phase Finite-Size Scaling vs $L/L_0$ | `plot_figure8_sc_scaling.py` | `SC_scaling.pdf` |
| **9** | `fig:phasediagram_smallL` | Small-System Phase Diagram ($L/L_0=8$) | `plot_figure9_phase_diagram_L8.py` | `phasediagram_L8.pdf` |

### One-Click Recreation of All Figures

To reproduce all 9 publication figures in sequence and automatically update the LaTeX paper assets:

```bash
python3 reproduction/reproduce_all_figures.py --all --export-paper
```

Individual figures can also be reproduced or re-simulated as detailed below.

---

### Figure 1: Domain Wall Coordinates & Schematic
- **Physical Description**: Schematic illustration of the 1D precessing domain wall configuration showing coordinates $u(x,t)$ and internal magnetization angle $\phi(x,t)$ in thin PMA films.
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure1_schematic.py --out-dir figures --export-paper
  ```
- **Output**: `figures/snap_system_345.pdf` and `paper/figuras/snap_system_345.pdf`.

---

### Figure 2: Rigid Wall Synchronization & Shapiro Steps
- **Physical Description**: Steady-state angular velocity $\langle\dot\psi_1\rangle_\tau$ and translational velocity $\langle\dot q_1\rangle_\tau$ vs. DC field $h_0$ for $h_a=0$ and $h_a=5$. Compares numerical 4th-order Runge-Kutta integration of the rigid Walker ODE against 1st- and 2nd-order analytical Bessel perturbation expansions, showing exact Shapiro step plateaus ($n=0, 1, 2$) and velocity reversal dips.
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure2_rigid_wall.py --out-dir figures --export-paper
  ```
- **Output**: `figures/figure2.pdf` and `paper/figuras/figure2.pdf`.

---

### Figure 3: Floquet Linear Stability Diagrams
- **Physical Description**: Linear stability spectrum of flexural spatial modes $\kappa \in [0.7, 1.2]$ around the synchronized flat DW for $h_a=2$ and $h_a=3$. Shows the largest Floquet multiplier modulus $|\mu_\kappa| > 1$ calculated by integrating the $2\times 2$ monodromy matrix over period $T = 2\pi/\Omega$, revealing Arnold tongue hybridization and the reentrant stable waist.
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure3_floquet.py --out-dir figures --export-paper
  ```
- **Output**: `figures/figure3.pdf` and `paper/figuras/figure3.pdf`.
- **Recomputing the Grid from Scratch**:
  ```bash
  python3 reproduction/plot_figure3_floquet.py --recompute --h0-pts 300 --kappa-pts 200
  ```

---

### Figure 4: Flexible Extended DW Dynamics vs $h_0$ ($h_a=3$)
- **Physical Description**: Mean DW translational velocity $\langle\dot{u}\rangle_{x,t}$, precessional velocity $\langle\dot{\phi}\rangle_{x,t}$, and steady-state interface roughness $\langle w^2\rangle_t$ across $h_0 \in [0, 10]$ at fixed $h_a=3$. Directly contrasts a small system ($L/L_0=8$), an extended thermodynamic system ($L/L_0=1024$), and the 0D rigid wall.
- **Reference Dataset**: `output/sweep_ha3_L8_L1024_979329/`
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure4_wall_metrics.py --out-dir figures --export-paper
  ```
- **Output**: `figures/sweep_ha_3.pdf` and `paper/figuras/sweep_ha_3.pdf`.

---

### Figure 5: Full Dynamical Phase Diagram ($L/L_0=256$)
- **Physical Description**: The complete 5-phase macroscopic dynamical phase diagram (PF, SF, SP, NF, NR) over $0 \le h_0 \le 10$ and $0 \le h_a \le 20$ in the thermodynamic limit ($L/L_0 = 256$), with solid contiguous regions, analytical Shapiro boundaries, Floquet instability contours, and the bottom DC baseline strip ($h_c \approx 7.075$).
- **Reference Dataset**: `output/sweep_full_973970/` and `output/floquet_transient_omega_1.000_L_256.bin`
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure5_phase_diagram.py --out-dir figures --export-paper
  ```
- **Output**: `figures/phasediagram_L256.pdf`, `figures/prb_rugosity.pdf`, `figures/prb_strobo_variance.pdf`.

---

### Figure 6: Universal Edwards-Wilkinson Scaling Collapse (NR Phase)
- **Physical Description**: Non-equilibrium surface roughening dynamics in the chaotic NR regime ($h_0=5, h_a=3, \alpha=0.27, \Omega=1, L/L_0=1000, t=2^{28}\delta t$). Demonstrates that the spatiotemporally chaotic DW belongs to the 1D Edwards-Wilkinson universality class:
  - Structure factor power-law decay: $S_\kappa \propto \kappa^{-2}$.
  - Interface roughness growth: $w^2 \propto t^{1/2}$.
  - Universal master curve collapse: $S_\kappa t^{-1}$ vs. $\kappa t^{1/2}$ ($\zeta = 1/2$, $z = 2$).
- **Reference Dataset**: `output/reference_nc_ew_data/`
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure6_nc_ew.py --out-dir figures --export-paper
  ```
- **Output**: `figures/NC_EW_inset.pdf` and `paper/figuras/NC_EW_inset.pdf`.

---

### Figure 7: Spatiotemporal Cascade of the SP Phase
- **Physical Description**: Real-space profile configurations $u(x,t) - \langle u \rangle_x$ along a section of the domain wall ($x \in [0, 128]$) sampled evenly across an entire AC driving period, colored continuously by local magnetization angle $\phi(x,t) \pmod{2\pi}$, with drive phase $\cos(\Omega t)$ shown on the right.
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure7_sp_cascade.py --out-dir figures --export-paper
  ```
- **Output**: `figures/sc_cascade.pdf` and `paper/figuras/sc_cascade.pdf`.

---

### Figure 8: Finite-Size Scaling of the SP Phase
- **Physical Description**: Finite-size scaling analysis demonstrating that in the SP phase, steady-state interface roughness $\langle w^2 \rangle_t$ and peak wavenumber $\kappa_{\mathrm{peak}}$ converge to constants in the thermodynamic limit. Confirms that small systems ($L/L_0 \le 64$) remain dynamically flat due to discrete mode quantization, while large systems converge to the maximum Floquet growth mode $\kappa^* \approx 0.833$.
- **Reference Dataset**: `output/scaling_data/`
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure8_sc_scaling.py --out-dir figures --export-paper
  ```
- **Output**: `figures/SC_scaling.pdf` and `paper/figuras/SC_scaling.pdf`.

---

### Figure 9: Small-System Phase Diagram ($L/L_0=8$)
- **Physical Description**: Complete dynamical phase diagram for small systems ($L/L_0 = 8$) illustrating how discrete spatial mode quantization suppresses the chaotic NR phase and shifts the critical field $h_c$ from $\approx 7.08$ down to $\approx 3.825$.
- **Reference Dataset**: `output/sweep_full_973971/` and `output/floquet_transient_omega_1.000_L_8.bin`
- **Reproduction Command**:
  ```bash
  python3 reproduction/plot_figure9_phase_diagram_L8.py --out-dir figures --export-paper
  ```
- **Output**: `figures/phasediagram_L8.pdf` and `paper/figuras/phasediagram_L8.pdf`.

---

## Prerequisites & Build System

### System Dependencies

- **C++ Compiler**: GCC $\ge 9.0$ or Clang $\ge 11.0$ (C++17 standard support).
- **Build Tools**: CMake $\ge 3.19$, Make or Ninja.
- **CPU Libraries**: FFTW3 with OpenMP support (`libfftw3`, `libfftw3_omp`), OpenMP (`libgomp`).
- **Python**: Python $\ge 3.8$, NumPy, Matplotlib.
- **Optional GPU Acceleration**: NVIDIA CUDA Toolkit $\ge 11.0$ (11.x, 12.x fully supported) and cuFFT.

#### Package Installation Commands
- **Ubuntu / Debian**:
  ```bash
  sudo apt update && sudo apt install -y cmake build-essential libfftw3-dev libfftw3-omp-dev libomp-dev
  ```
- **Fedora / RHEL / Rocky Linux**:
  ```bash
  sudo dnf install -y cmake gcc-c++ fftw-devel fftw-libs-openmp libgomp
  ```
- **Python Environment**:
  ```bash
  pip install numpy matplotlib
  ```

### Automated Quick Builds

Helper scripts configure compiler flags, ccache directories, and build targets automatically:

```bash
# Compile CPU targets locally (OpenMP + FFTW3):
./compile_local.sh

# Compile GPU targets on CUDA-enabled systems (NVCC + cuFFT):
./compile_cluster.sh
```

### Manual CMake Build (CPU Only)

```bash
mkdir -p build_local && cd build_local
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Executables are placed in `build_local/bin/`.

### Manual CMake Build (CUDA GPU Acceleration)

```bash
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Executables are placed in `build/bin/`.

---

## Executables & Module Reference

### Rigid Wall Binaries

Located in `build_local/bin/` (or `build/bin/`):

- **CPU OpenMP Solver**:
  ```bash
  ./build_local/bin/rigid_wall_sweep_cpu [options]
  ```
  Options:
  - `--h0-min <float>`, `--h0-max <float>`, `--num-h0 <int>`: DC field range and grid points.
  - `--ha-min <float>`, `--ha-max <float>`, `--num-ha <int>`: AC amplitude range and grid points.
  - `-o <filename>`: Output file path.
- **CUDA GPU Solver**:
  ```bash
  ./build/bin/rigid_wall_sweep_gpu [h0_min] [h0_max] [num_h0] [ha_min] [ha_max] [num_ha] [out_file]
  ```
  Performs thousands of concurrent ODE sweeps in parallel across GPU threads.

### Floquet Stability Binaries

- **CPU OpenMP Solver**:
  ```bash
  # 2D Parameter Sweep over (h0, ha):
  ./build_local/bin/floquet_sweep_cpu --num-h0 100 --num-ha 100 -o floquet_sweep.bin

  # 1D Slice over (h0, kappa) at fixed ha (Figure 3 Mode):
  ./build_local/bin/floquet_sweep_cpu --ha 2.0 --num-h0 300 --num-k 200 -o floquet_slice.txt
  ```
- **CUDA GPU Solver**:
  ```bash
  ./build/bin/floquet_sweep_gpu [num_h0] [num_ha] [num_k] [out_file]
  ```

### PDE Simulation Binaries

All PDE executables support an optional final argument `backend`: `cpu` or `gpu`.

```bash
# 1. Standard Time Evolution & Observables
./build_local/bin/main <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                       <transient_time> <averaging_time> <window_time> <subsample_rate> \
                       <output_prefix> [eps=0.05] [seed=42] [backend=cpu]

# 2. Logarithmic Surface Growth Scaling (for Figure 6)
./build_local/bin/time_evo <size> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> \
                           <output_prefix> [eps=0.05] [seed=42] [backend=cpu]

# 3. 1D / 2D Parameter Sweeps & Order Parameters (for Figures 4, 5, 9)
./build_local/bin/sweep <size> <L> <alpha> <omega> <dt> <transient_time> <eval_time> \
                        <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> [backend=cpu]

# 4. Central Moments & Cumulant Evolution
./build_local/bin/moments <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <total_time> \
                          <start_time> <num_checkpoints> <num_sq_checkpoints> <output_prefix> \
                          [eps=0.05] [seed=42] [backend=cpu]

# 5. Real-Space Profile Snapshots (for Figure 7)
./build_local/bin/snapshots <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                            <transient_time> <snapshot_time> <window_time> <snapshot_rate> \
                            <output_prefix> [eps=0.05] [seed=42] [backend=cpu]

# 6. High-Framerate Binary Video Streaming
./build_local/bin/movies <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <t_max> \
                         <window_time> <movie_start> <movie_end> <movie_rate> \
                         <output_prefix> [eps=0.05] [seed=42] [backend=cpu]

# 7. Roughness Saturation & Convergence Detector
./build_local/bin/sat_check <size> <L> <alpha> <dt> [backend=cpu]
```

---

## Verifying and Compiling the LaTeX Manuscript

The `paper/` directory contains the complete RevTeX 4-1 manuscript source. An automated verification target checks that all 9 publication figures exist before building:

```bash
cd paper

# 1. Verify that all 9 figures are present in paper/figuras/:
make check-figs

# 2. Compile the paper and bibliography:
make

# 3. Clean temporary LaTeX auxiliary files:
make clean
```

The resulting publication-ready document will be saved at `paper/main.pdf`.

---

## Citation

If you use this simulation framework, numerical methods, or reference datasets in your research, please cite our paper:

```bibtex
@article{perezherrera2026synchronization,
  title   = {Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls},
  author  = {P{\'e}rez Herrera, Luz and Pusiol, E. F. and Kolton, A. B.},
  journal = {Physical Review B},
  year    = {2026},
  note    = {Under review}
}

@misc{github_dw_uphi_simulator,
  author       = {Luz P{\'e}rez Herrera},
  title        = {{DW-uphi-simulator: High-performance pseudospectral framework for 1D domain wall dynamics}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/lightph/dw-uphi-simulator}}
}
```

---

## Acknowledgments

Simulations were performed using the high-performance computing cluster at the *Gerencia de Física, Centro Atómico Bariloche (CAB-CNEA / SNCAD, Argentina)*. We thank T. Giamarchi for fruitful discussions.
