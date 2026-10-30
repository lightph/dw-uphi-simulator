# Domain Wall Simulator: Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5%2B-red.svg)](https://www.openmp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.19%2B-orange.svg)](https://cmake.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

High-performance C++17 / CUDA simulation framework, Floquet monodromy stability analyzer, analytical perturbation solvers, and complete publication reproduction suite for:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, and A. B. Kolton  
> *Instituto Balseiro (UNCUYO) & Centro Atómico Bariloche (CNEA-CONICET), Argentina.*  
> Physical Review B (2026) • [GitHub Repository](https://github.com/lightph/dw-uphi-simulator)

---

## ⚡ Quickstart (Under 2 Minutes)

### 1. Prerequisites & Installation
- **C++ Compiler**: GCC $\ge 9.0$ or Clang $\ge 11.0$ (C++17), CMake $\ge 3.19$.
- **Libraries**: FFTW3 with OpenMP (`libfftw3`, `libfftw3_omp`), OpenMP (`libgomp`).
- **Python**: Python $\ge 3.8$ (`numpy`, `matplotlib`, `scipy`).
- **Optional GPU**: NVIDIA CUDA Toolkit $\ge 11.0$ and cuFFT.

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y cmake build-essential libfftw3-dev libfftw3-omp-dev libomp-dev
pip install numpy matplotlib scipy

# Fedora / RHEL / Rocky Linux
sudo dnf install -y cmake gcc-c++ fftw-devel fftw-libs-openmp libgomp
pip install numpy matplotlib scipy
```

### 2. Compile
```bash
# Compile CPU targets locally (OpenMP + FFTW3):
./compile_local.sh

# Or compile GPU targets on CUDA-enabled systems:
./compile_cluster.sh
```
Executables are placed in `build_local/bin/` (CPU) or `build/bin/` (GPU).

### 3. Reproduce All Paper Figures
Pre-computed simulation datasets are included in `output/`. Recreate all 9 publication figures in one command:
```bash
python3 reproduction/reproduce_all_figures.py --all
```
Generated publication-quality vector PDFs will be saved to `figures/` and copied to `paper/figuras/`.

---

## 📊 Reproducing Publication Figures

Every figure in the paper can be **plotted immediately** from pre-computed data in `output/` or **regenerated from scratch** via dedicated simulation scripts:

| Fig # | Paper Label | Subject / Physical Description | Plot Pre-computed Data (Instant) | Regenerate Simulation Data |
|:---:|:---|:---|:---|:---|
| **1** | `fig:snap` | DW coordinates & real-space schematic | `python3 reproduction/plot_figure1_schematic.py` | `./scripts/generate_fig1_data.sh --backend cpu` |
| **2** | `fig:ej_walker` | Rigid wall Shapiro steps & velocities | `python3 reproduction/plot_figure2_rigid_wall.py` | `./scripts/generate_fig2_data.sh --backend cpu` |
| **3** | `fig:sync_stability` | Floquet linear stability diagrams ($h_a=2, 3$) | `python3 reproduction/plot_figure3_floquet.py` | `./scripts/generate_fig3_data.sh --backend cpu` |
| **4** | `fig:wall_metrics` | Flexible DW dynamics vs $h_0$ ($L/L_0=8, 1024$) | `python3 reproduction/plot_figure4_wall_metrics.py` | `./scripts/generate_fig4_data.sh --backend cpu` |
| **5** | `fig:fases` | Full 5-phase diagram ($L/L_0=256$) | `python3 reproduction/plot_figure5_phase_diagram.py` | `./scripts/generate_fig5_data.sh --backend cpu` |
| **6** | `fig:estructura` | Edwards-Wilkinson scaling collapse (NR phase) | `python3 reproduction/plot_figure6_nc_ew.py` | `./scripts/generate_fig6_data.sh --backend cpu` |
| **7** | `fig:sc_cascade` | SP phase spatiotemporal cascade profiles | `python3 reproduction/plot_figure7_sp_cascade.py` | `./scripts/generate_fig7_data.sh --backend cpu` |
| **8** | `fig:estructuraSCvsL` | SP phase finite-size scaling vs $L/L_0$ | `python3 reproduction/plot_figure8_sc_scaling.py` | `./scripts/generate_fig8_data.sh --backend cpu` |
| **9** | `fig:phasediagram_smallL` | Small-system phase diagram ($L/L_0=8$) | `python3 reproduction/plot_figure9_phase_diagram_L8.py` | `./scripts/generate_fig9_data.sh --backend cpu` |

> **Options for Plotting**:
> - Pass `--export-paper` to export directly into `paper/figuras/` for compilation with `cd paper && make`.
> - Pass `--out-dir <dir>` to redirect PDF/PNG outputs to a custom folder.
> - For Figure 3, pass `--recompute` to solve the Floquet monodromy matrix numerically on the fly:
>   `python3 reproduction/plot_figure3_floquet.py --recompute`

---

## 🎛️ Running Simulations & Changing Parameters

You can easily run new simulations with modified parameters (fields, damping, domain size, frequencies) through three convenient workflows:

### Workflow 1: High-Level Shell Scripts (Recommended)

The scripts in `scripts/` provide friendly command-line options and run either on **CPU** or **GPU** (or submit to a **SLURM cluster** with `--slurm`):

```bash
# 1. Sweep DC field h0 at a different AC amplitude ha (e.g. ha = 4.0) on CPU:
./scripts/generate_fig4_data.sh --backend cpu --ha 4.0 --num-h0 100 --h0-min 0.0 --h0-max 10.0

# 2. Test a different domain length L (e.g. L = 512, N = 1024):
./scripts/generate_fig4_data.sh --backend cpu --L-big 512.0 --size-big 1024

# 3. Compute Floquet stability slice for a custom AC field ha = 3.5:
./scripts/generate_fig3_data.sh --backend cpu --ha 3.5

# 4. Use GPU acceleration (just change --backend cpu to --backend gpu):
./scripts/generate_fig4_data.sh --backend gpu --ha 3.0

# 5. Check all available options for any script:
./scripts/generate_fig4_data.sh --help
```

### Workflow 2: Direct C++ Simulation Binaries

For custom pipelines, invoke the compiled binaries in `build_local/bin/` (or `build/bin/`):

#### 1. Full PDE Time Evolution (`main`)
Integrates the 1D Slonczewski equations and writes time-series observables, roughness $w^2(t)$, and mean velocities $\langle\dot u\rangle$, $\langle\dot\phi\rangle$:
```bash
# Syntax:
./build_local/bin/main <N> <batch> <L> <alpha> <h0> <ha> <omega> <dt> \
                       <t_transient> <t_avg> <t_window> <subsample_rate> \
                       <out_prefix> [eps] [seed] [backend]

# Example run:
./build_local/bin/main 512 1 256.0 0.27 4.0 3.0 1.0 0.019635 \
                       2000.0 1000.0 100.0 10 \
                       output/my_sim 0.0005 42 cpu
```

#### 2. PDE Parameter Sweep (`sweep`)
Sweeps across AC fields $h_a$ at fixed $h_0$ to extract steady-state order parameters:
```bash
# Syntax:
./build_local/bin/sweep <N> <L> <alpha> <omega> <dt> <t_transient> <t_eval> \
                        <h0> <ha_min> <ha_max> <ha_steps> <out_file> [backend] [eps] [seed]

# Example run:
./build_local/bin/sweep 512 256.0 0.27 1.0 0.019635 2000.0 1000.0 \
                        4.0 0.0 6.0 61 output/sweep_h0_4.txt cpu 0.0005 42
```

#### 3. Logarithmic Scaling Integrator (`time_evo`)
Samples system state at powers-of-two steps ($t = 2^p \delta t$) to track non-equilibrium surface roughening (Edwards-Wilkinson growth):
```bash
./build_local/bin/time_evo <N> <L> <alpha> <h0> <ha> <omega> <dt> <max_power> <out_prefix> [eps] [seed] [backend]
```

#### 4. Rigid Wall 0D ODE & Shapiro Steps (`rigid_wall_sweep_cpu`)
Integrates the rigid Walker ODE across $(h_0, h_a)$ to map phase-locking steps and velocity reversal:
```bash
./build_local/bin/rigid_wall_sweep_cpu --h0-min 0.0 --h0-max 10.0 --num-h0 500 --ha 5.0 -o output/rigid_ha5.txt
```

#### 5. Floquet Linear Stability (`floquet_sweep_cpu`)
Integrates the monodromy matrix over period $T = 2\pi/\Omega$ to compute Floquet multipliers $|\mu_\kappa|$:
```bash
./build_local/bin/floquet_sweep_cpu --ha 3.0 --h0-min 0.0 --h0-max 8.0 --num-h0 300 --k-min 0.7 --k-max 1.2 --num-k 150 -o output/floquet_ha3.txt
```

### Workflow 3: Interactive Python Prototyping (No C++ Compilation)
If you want to explore the 0D ODE or Floquet stability without building C++ code:
- **0D Rigid Wall ODE**: `rigid_wall/python/rigid_wall.py` (pure NumPy Runge-Kutta solver and analytical Bessel formulas).
- **Floquet Stability**: `floquet/python/floquet_solver.py` (vectorized NumPy monodromy integrator and Arnold tongue mapper).

---

## ⚙️ Standard Physical & Numerical Parameters

| Parameter | Symbol | Paper Default | Physical Units (Co/Pt Multilayers) |
|:---|:---:|:---:|:---|
| **DC Field** | $h_0$ | `0.0 – 10.0` | $H_0 = h_0 H_W$ ($H_W \approx 2.48\text{ mT}$) |
| **AC Field Amplitude** | $h_a$ | `0.0 – 20.0` | $H_a = h_a H_W$ |
| **Drive Frequency** | $\Omega$ | `1.0` | $f = \Omega / (2\pi T_0) \approx 480\text{ MHz}$ ($T_0 \approx 0.33\text{ ns}$) |
| **Gilbert Damping** | $\alpha$ | `0.27` | Dimensionless damping constant |
| **System Length** | $L$ | `8.0` / `256.0` / `1024.0` | $L_{\text{phys}} = L \times L_0$ ($L_0 \approx 58\text{ nm}$) |
| **Spatial Grid Points** | $N$ | $2 \times L$ | Grid spacing $\delta x = L/N = 0.5$ ($\approx 29\text{ nm}$) |
| **Integration Step** | $\delta t$ | `0.019635` | $2\pi / (320\,\Omega)$ (320 steps per driving cycle) |
| **Initial Perturbation** | $\epsilon$ | `0.0005` | Amplitude of random transverse noise on $u$ and $\phi$ |

---

## 📁 Repository Directory Map

```text
dw-uphi-simulator/
├── reproduction/        # Standalone reproduction scripts for Figures 1 to 9
├── scripts/             # Data regeneration scripts (CPU, GPU, SLURM cluster)
├── output/              # Pre-computed publication datasets for all figures
├── figures/             # Target directory for generated PDF/PNG figures
├── paper/               # Complete LaTeX publication manuscript (RevTeX 4-1)
│   ├── main.tex         # Paper source
│   ├── Makefile         # Automated verification and compilation
│   └── figuras/         # Figure assets embedded in the PDF manuscript
├── pde/                 # 1D Pseudospectral PDE engine (Crank-Nicolson)
├── floquet/             # Floquet monodromy stability analyzer
├── rigid_wall/          # 0D rigid-wall ODE solver & analytical Bessel theory
├── include/             # Core C++17 headers (FFTW3, cuFFT, backend abstractions)
├── src/                 # Main C++ and CUDA simulation kernels
├── compile_local.sh     # Quick CPU build script (FFTW3 + OpenMP)
└── compile_cluster.sh   # Quick GPU build script (CUDA + cuFFT)
```

---

## 📄 Compiling the LaTeX Manuscript

The `paper/` directory contains the complete RevTeX 4-1 manuscript. To compile `paper/main.pdf`:

```bash
cd paper
make check-figs   # Verifies all 9 publication figures exist in paper/figuras/
make              # Builds main.pdf with pdflatex and bibtex
make clean        # Cleans auxiliary files (.aux, .log, .bbl)
```

---

## 📖 Physical Model Summary

The 1D collective dynamics of a precessing domain wall in thin ferromagnetic films with perpendicular anisotropy are governed by its interface displacement $u(x,t)$ and internal magnetization angle $\phi(x,t)$ under magnetic field $h(t) = h_0 + h_a \cos(\Omega t)$:

$$\dot{u} = \frac{1}{2} \left[ \alpha^2 h(t) + \sin(2\phi) + \alpha \partial_x^2 u - \partial_x^2 \phi \right]$$

$$\dot{\phi} = \frac{1}{2} \left[ \alpha h(t) - \alpha \sin(2\phi) + \partial_x^2 u + \alpha \partial_x^2 \phi \right]$$

In terms of the complex coordinate $z(x,t) = u(x,t) - i\phi(x,t)$:

$$\dot{z} = \frac{\alpha - i}{2} \left[ \alpha h(t) + i\sin\left(2\,\mathrm{Im}(z)\right) + \partial_x^2 z \right]$$

Linear diffusion and dispersion $\mathcal{L}(z) = \frac{\alpha-i}{2}\partial_x^2 z$ is integrated unconditionally in Fourier space via the Crank-Nicolson method, while non-linear gyrotropic terms are stepped in real space. For full derivations and phase characterization, see Sections II–IV of the paper.

---

## 📜 Citation

If you use this simulation framework, numerical methods, or datasets, please cite:

```bibtex
@article{perezherrera2026synchronization,
  title   = {Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls},
  author  = {P{\'e}rez Herrera, Luz and Pusiol, E. F. and Kolton, A. B.},
  journal = {Physical Review B},
  year    = {2026},
  note    = {Under review}
}

@misc{dw_uphi_simulator,
  author       = {Luz P{\'e}rez Herrera},
  title        = {{DW-uphi-simulator: High-performance pseudospectral framework for 1D domain wall dynamics}},
  year         = {2026},
  howpublished = {\url{https://github.com/lightph/dw-uphi-simulator}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
