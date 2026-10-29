# Publication Figure Reproduction Guide

This directory provides automated scripts and pipelines to reproduce **every single figure** from the Physical Review B publication:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, and A. B. Kolton  
> *Physical Review B*

---

## 1. Quick Reproduction

To regenerate all 9 publication figures in a single automated pass and export them directly to `paper/figuras/`:

```bash
python3 reproduction/reproduce_all_figures.py --all --export-paper
```

Once complete, compile the LaTeX manuscript:
```bash
cd paper && make
```

---

## 2. Section-by-Section Figure Catalog

| Fig # | File in `paper/figuras/` | Label in `main.tex` | Module / Method | Physical Subject |
|:---:|:---|:---|:---:|:---|
| **Fig. 1** | `snap_system_345.pdf` | `fig:snap` | Schematic | Schematic diagram of 1D domain wall collective coordinates $u(x,t)$ and $\phi(x,t)$. |
| **Fig. 2** | `figure2.pdf` | `fig:ej_walker` | `rigid_wall/` | Steady-state velocities $\langle\dot\psi_1\rangle_\tau$ and $\langle\dot q_1\rangle_\tau$ vs $h_0$ for $h_a=0$ and $h_a=5$, Shapiro steps, and 1st/2nd-order perturbation boundaries. |
| **Fig. 3** | `figure3.pdf` | `fig:sync_stability` | `floquet/` | Linear Floquet stability diagrams for $h_a=2$ and $h_a=3$ across modes $\kappa \in [0.7, 1.2]$ and fields $h_0 \in [0, 8]$. |
| **Fig. 4** | `sweep_ha_3.pdf` | `fig:wall_metrics` | `pde/` | Flexible extended DW velocities $\langle\dot u\rangle$, $\langle\dot\phi\rangle$, and roughness $\langle w^2\rangle_t$ vs $h_0$ at $h_a=3$ ($L/L_0=8$ and $1024$). |
| **Fig. 5** | `phasediagram_L256.pdf` | `fig:fases` | `pde/` + `floquet/` | Full 5-phase macroscopic dynamical phase diagram in $(h_0, h_a)$ plane for $L/L_0=256$ (PF, SF, SP, NF, NR). |
| **Fig. 6** | `NC_EW_inset.pdf` | `fig:estructura` | `pde/` | Non-equilibrium surface roughening in the NR phase: Edwards-Wilkinson scaling collapse ($S_\kappa \propto \kappa^{-2}$, $w^2 \propto t^{1/2}$, $z=2, \zeta=1/2$). |
| **Fig. 7** | `sc_cascade.pdf` | `fig:sc_cascade` | `pde/` | Real-space profile snapshots $u(x,t)-\langle u\rangle_x$ in the SP phase over an AC drive period, colored by local precession angle $\phi(x,t)$. |
| **Fig. 8** | `SC_scaling.pdf` | `fig:estructuraSCvsL` | `pde/` + `floquet/` | Finite-size scaling of the SP phase: roughness $\langle w^2\rangle_t$ and peak mode $\kappa_{\mathrm{peak}}$ vs $L/L_0$, showing convergence to Floquet mode $\kappa^* \approx 0.833$. |
| **Fig. 9** | `phasediagram_L8.pdf` | `fig:phasediagram_smallL` | `pde/` + `floquet/` | Small-system phase diagram ($L/L_0=8$) illustrating mode quantization and suppression of the chaotic NR phase. |

---

## 3. Individual Figure Reproduction Commands

### Figure 1: Model Schematic
```bash
python3 reproduction/plot_figure1_schematic.py --export-paper
```

### Figure 2: Rigid Domain Wall Shapiro Steps
```bash
python3 reproduction/plot_figure2_rigid_wall.py --export-paper
```

### Figure 3: Floquet Linear Stability Diagrams
```bash
python3 reproduction/plot_figure3_floquet.py --export-paper
```

### Figure 4: Flexible Extended DW Dynamics ($h_a = 3$)
```bash
python3 reproduction/plot_figure4_wall_metrics.py --export-paper
```

### Figure 5: Full Phase Diagram ($L/L_0 = 256$)
```bash
python3 reproduction/plot_figure5_phase_diagram.py --export-paper
```

### Figure 6: Universal Edwards-Wilkinson Scaling Collapse
```bash
python3 reproduction/plot_figure6_nc_ew.py --export-paper
```

### Figure 7: SP Phase Spatiotemporal Cascade Profiles
```bash
python3 reproduction/plot_figure7_sp_cascade.py --export-paper
```

### Figure 8: Finite-Size Scaling of the SP Phase
```bash
python3 reproduction/plot_figure8_sc_scaling.py --export-paper
```

### Figure 9: Small-System Phase Diagram ($L/L_0 = 8$)
```bash
python3 reproduction/plot_figure9_phase_diagram_L8.py --export-paper
```

