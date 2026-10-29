# Simulation Data Regeneration Suite

This directory contains production scripts to regenerate the simulation datasets required for **every figure** in the Physical Review B paper:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, and A. B. Kolton  
> *Instituto Balseiro (UNCUYO) & Centro Atómico Bariloche (CNEA-CONICET)*

Each script is dual-mode: it can be executed **locally** (on CPU or GPU) or submitted to a **SLURM cluster** using either the **`gpu`** partition or the **`knl_tacc`** (Intel KNL / Xeon CPU) partition.

---

## Standardized Defaults & Parameters

| Parameter | Symbol / Variable | Standard Default | Customization |
|:---|:---:|:---:|:---|
| **Initial Perturbation Amplitude** | `eps` | `0.0005` | `--eps <val>` or `EPS=<val>` |
| **Random RNG Seed** | `seed` | `42` | `--seed <int>` or `SEED=<int>` |
| **Small System Size** | $L/L_0$ | `8.0` ($N = 16$, $dx = 0.5$) | `--L 8.0 --size 16` |
| **Phase Diagram Size** | $L/L_0$ | `256.0` ($N = 512$, $dx = 0.5$) | `--L 256.0 --size 512` |
| **Extended Thermodynamic Size** | $L/L_0$ | `1024.0` ($N = 2048$, $dx = 0.5$) | `--L 1024.0 --size 2048` |
| **Gilbert Damping** | $\alpha$ | `0.27` | `--alpha <val>` or `ALPHA=<val>` |
| **Drive Angular Frequency** | $\Omega$ | `1.0` | `--omega <val>` or `OMEGA=<val>` |
| **Standard Time Step** | $\delta t$ | `0.019634954` ($2\pi / (320\Omega)$) | `--dt <val>` or `DT=<val>` |

---

## Quick Reference: Scripts Mapping to Paper Figures

| Script | Paper Figure | Subject / Description | Executable Used | Standard Default Size |
|:---|:---:|:---|:---:|:---:|
| [`generate_fig1_data.sh`](generate_fig1_data.sh) | **Fig. 1** | Real-space profile snapshots $u(x,t)$ and $\phi(x,t)$ | `snapshots` | $L=512$, $N=1024$ |
| [`generate_fig2_data.sh`](generate_fig2_data.sh) | **Fig. 2** | Rigid wall Shapiro steps vs $h_0$ ($h_a=0, 5$) | `rigid_wall_sweep_cpu/gpu` | 0D ODE (500 points) |
| [`generate_fig3_data.sh`](generate_fig3_data.sh) | **Fig. 3** | Floquet stability diagrams for $h_a=2$ and $h_a=3$ | `floquet_sweep_cpu/gpu` | $300 \times 200$ grid |
| [`generate_fig4_data.sh`](generate_fig4_data.sh) | **Fig. 4** | Flexible DW velocities & roughness vs $h_0$ at $h_a=3$ | `sweep` | $L=8$ ($N=16$) and $L=1024$ ($N=2048$) |
| [`generate_fig5_data.sh`](generate_fig5_data.sh) | **Fig. 5** | Full 2D Phase Diagram in $(h_0, h_a)$ ($L=256$) | `sweep` + `floquet` | $L=256$, $N=512$ ($100 \times 201$ grid) |
| [`generate_fig6_data.sh`](generate_fig6_data.sh) | **Fig. 6** | Edwards-Wilkinson scaling collapse (NR phase) | `time_evo` | $L=1024$ ($N=2048$) to $L=8192$ ($N=65536$) |
| [`generate_fig7_data.sh`](generate_fig7_data.sh) | **Fig. 7** | SP phase spatiotemporal profile cascade snapshots | `snapshots` | $L=512$, $N=1024$ |
| [`generate_fig8_data.sh`](generate_fig8_data.sh) | **Fig. 8** | SP phase finite-size scaling vs $L/L_0 \in [1, 262144]$ | `main` | $L = 2^{\text{exp}}$ ($N = 2 \times L$), $\text{exp} \in [0, 18]$ |
| [`generate_fig9_data.sh`](generate_fig9_data.sh) | **Fig. 9** | Small-system phase diagram ($L=8$) | `sweep` + `floquet` | $L=8$, $N=16$ ($100 \times 201$ grid) |
| [`generate_all_data.sh`](generate_all_data.sh) | **All** | Master orchestrator regenerating data across figures | All | Standardized |

---

## Execution Modes & Examples

### 1. Local Execution (CPU Backend)
Uses OpenMP multi-threading with compiled CPU binaries:
```bash
# Run locally on CPU with default eps=0.0005:
./scripts/generate_fig4_data.sh --backend cpu

# Override perturbation scale and grid size:
./scripts/generate_fig4_data.sh --backend cpu --eps 0.001 --num-h0 100
```

### 2. Local Execution (CUDA GPU Backend)
Uses CUDA double-precision acceleration:
```bash
# Run locally on GPU:
./scripts/generate_fig4_data.sh --backend gpu

# Fast test run with custom transient:
./scripts/generate_fig4_data.sh --backend gpu --transient 1000 --eval 1000
```

### 3. SLURM Cluster Execution (GPU Partition)
Submits a SLURM job to the `gpu` partition with `--gres=gpu:1`:
```bash
# Via CLI flag:
./scripts/generate_fig4_data.sh --slurm --partition gpu

# Or directly with sbatch:
sbatch --partition=gpu scripts/generate_fig4_data.sh
```

### 4. SLURM Cluster Execution (KNL / CPU Partition)
Submits a SLURM job to the `knl_tacc` compute partition:
```bash
# Via CLI flag:
./scripts/generate_fig4_data.sh --slurm --partition knl_tacc

# Or directly with sbatch:
sbatch --partition=knl_tacc scripts/generate_fig4_data.sh
```

---

## Master Regeneration Runner

To regenerate simulation data for all figures (or a subset) in one unified command:

```bash
# Run locally on CPU:
./scripts/generate_all_data.sh --backend cpu

# Run locally on GPU:
./scripts/generate_all_data.sh --backend gpu

# Submit array/batch jobs to SLURM gpu partition:
./scripts/generate_all_data.sh --slurm --partition gpu

# Regenerate specific figures only (e.g. Figures 2, 3, and 4):
./scripts/generate_all_data.sh --figures 2 3 4 --backend cpu
```

---

## Plotting & Publication Reproduction

Once simulation data is generated, run the master plotting suite to recreate the publication PDF figures:

```bash
python3 reproduction/reproduce_all_figures.py --all --export-paper
```
This automatically reads the generated outputs, formats figures according to PRB guidelines, and exports them directly to `figures/` and `paper/figuras/`.

