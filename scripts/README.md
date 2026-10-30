# Simulation Data Regeneration Suite

This directory contains automated production scripts to regenerate the numerical simulation datasets for all figures in the publication:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Physical Review B (2026)

All scripts are **dual-mode**: they run locally on your workstation (**CPU** or **GPU**) or can be submitted directly to a **SLURM cluster** using the `--slurm` flag.

---

## 1. Quick Figure Regeneration Map

| Script | Paper Figure | Physical System | Primary Executable | Default Domain Size |
|:---|:---:|:---|:---:|:---:|
| [`generate_fig1_data.sh`](generate_fig1_data.sh) | **Fig. 1** | DW profile snapshots $u(x,t)$, $\phi(x,t)$ | `snapshots` | $L=512$, $N=1024$ |
| [`generate_fig2_data.sh`](generate_fig2_data.sh) | **Fig. 2** | Rigid wall Shapiro steps vs $h_0$ ($h_a=0, 5$) | `rigid_wall_sweep_cpu/gpu` | 0D ODE (500 pts) |
| [`generate_fig3_data.sh`](generate_fig3_data.sh) | **Fig. 3** | Floquet stability diagrams ($h_a=2, 3$) | `floquet_sweep_cpu/gpu` | $300 \times 200$ grid |
| [`generate_fig4_data.sh`](generate_fig4_data.sh) | **Fig. 4** | Velocities & roughness vs $h_0$ ($h_a=3$) | `sweep` | $L=8$ and $L=1024$ |
| [`generate_fig5_data.sh`](generate_fig5_data.sh) | **Fig. 5** | Full 2D Phase Diagram ($L=256$) | `sweep` + `floquet` | $L=256$, $N=512$ ($100\times 201$) |
| [`generate_fig6_data.sh`](generate_fig6_data.sh) | **Fig. 6** | Edwards-Wilkinson scaling collapse | `time_evo` | $L=1024$ / $L=8192$ |
| [`generate_fig7_data.sh`](generate_fig7_data.sh) | **Fig. 7** | SP phase spatiotemporal cascade profiles | `snapshots` | $L=512$, $N=1024$ |
| [`generate_fig8_data.sh`](generate_fig8_data.sh) | **Fig. 8** | Finite-size scaling vs $L \in [1, 262144]$ | `main` | $L = 2^{\text{exp}}$, $N = 2L$ |
| [`generate_fig9_data.sh`](generate_fig9_data.sh) | **Fig. 9** | Small-system phase diagram ($L=8$) | `sweep` + `floquet` | $L=8$, $N=16$ ($100\times 201$) |
| [`generate_all_data.sh`](generate_all_data.sh) | **All** | Master runner across all figures | All | Standardized |

---

## 2. Command-Line Options & Changing Parameters

All scripts share a unified argument parser defined in [`common_env.sh`](common_env.sh). Pass `--help` to any script to see its supported options:

```bash
./scripts/generate_fig4_data.sh --help
```

### Parameter Reference

| Category | Flag | Default | Description |
|:---|:---|:---:|:---|
| **Backend** | `--backend <cpu\|gpu>` | auto | Compute engine: `cpu` (OpenMP) or `gpu` (CUDA) |
| **Cluster** | `--slurm` | false | Dispatch via `sbatch` instead of running locally |
| | `--partition <name>` | `gpu` / `knl_tacc` | SLURM cluster queue |
| **Magnetic Fields** | `--h0 <val>` | - | DC magnetic field |
| | `--ha <val>` | - | AC magnetic field amplitude |
| | `--h0-min <val>`, `--h0-max <val>` | - | DC field range for sweeps |
| | `--num-h0 <int>` | - | Number of DC field sample points |
| | `--ha-min <val>`, `--ha-max <val>` | - | AC amplitude range for sweeps |
| | `--ha-steps <int>` | - | Number of AC amplitude sample points |
| **Domain Size** | `--L <val>` | - | Domain length $L/L_0$ |
| | `--size <int>` | - | Number of grid points $N$ (standard: $2 \times L$) |
| **Physics** | `--alpha <val>` | `0.27` | Gilbert damping parameter |
| | `--omega <val>` | `1.0` | AC driving frequency $\Omega$ |
| **Numerics** | `--dt <val>` | `0.019635` | Integration step ($2\pi / (320\,\Omega)$) |
| | `--eps <val>` | `0.0005` | Initial random noise amplitude |
| | `--seed <int>` | `42` | RNG seed for initial conditions |
| **Output** | `--out-dir <path>` | varies | Destination directory for output files |

---

## 3. Usage Examples

### Local Execution (CPU)
```bash
# Run Figure 4 sweep on CPU with default parameters:
./scripts/generate_fig4_data.sh --backend cpu

# Customize AC amplitude (ha = 3.5) and increase resolution to 400 points:
./scripts/generate_fig4_data.sh --backend cpu --ha 3.5 --num-h0 400

# Test a different system length (e.g. L = 512, N = 1024):
./scripts/generate_fig4_data.sh --backend cpu --L-big 512.0 --size-big 1024
```

### Local Execution (GPU)
```bash
# Run Figure 4 sweep with CUDA acceleration:
./scripts/generate_fig4_data.sh --backend gpu

# Fast test run with shorter transient and evaluation times:
./scripts/generate_fig4_data.sh --backend gpu --transient 2000 --eval 1000
```

### SLURM Cluster Submission
```bash
# Submit Figure 4 simulation to SLURM cluster GPU queue:
./scripts/generate_fig4_data.sh --slurm --partition gpu

# Submit to CPU partition (e.g., KNL / Xeon):
./scripts/generate_fig4_data.sh --slurm --partition knl_tacc
```

### Master Generation (Multiple Figures)
```bash
# Regenerate data for Figures 2, 3, and 4 in one command on CPU:
./scripts/generate_all_data.sh --figures 2 3 4 --backend cpu
```

---

## 4. Plotting Generated Data

Once simulation outputs are produced, plot the figures and update paper assets:

```bash
# Plot all figures:
python3 reproduction/reproduce_all_figures.py --all

# Or plot an individual figure pointing to your output directory:
python3 reproduction/plot_figure4_wall_metrics.py --data-dir output/sweep_ha3_L8_L1024_recomputed --export-paper
```
