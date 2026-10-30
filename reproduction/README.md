# Publication Figure Reproduction Guide

This directory contains standalone Python plotting scripts to reproduce **every figure** in the paper:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, and A. B. Kolton  
> *Physical Review B (2026)*

All figures are formatted to Physical Review B standards (RevTeX column widths, STIX math fonts, vector PDF outputs).

---

## 1. Quick Reproduction (All Figures)

To regenerate all 9 publication figures from pre-computed reference data in `output/` and export them directly to `paper/figuras/`:

```bash
python3 reproduction/reproduce_all_figures.py --all --export-paper
```

Then build the publication PDF:
```bash
cd paper && make
```

---

## 2. Figure Catalog & Mapping

| Fig # | Paper Label | Standardized Paper Name (`paper/figuras/`) | Output File (`figures/`) | Physical Subject | Standalone Command |
|:---:|:---|:---|:---|:---|:---|
| **Fig. 1** | `fig:snap` | `fig1_dw_schematic.pdf` | `snap_system_345.pdf` | DW collective coordinates $u(x,t)$, $\phi(x,t)$ | `python3 reproduction/plot_figure1_schematic.py` |
| **Fig. 2** | `fig:ej_walker` | `fig2_rigid_wall_shapiro.pdf` | `figure2.pdf` | Rigid DW Shapiro steps & average velocities | `python3 reproduction/plot_figure2_rigid_wall.py` |
| **Fig. 3** | `fig:sync_stability` | `fig3_floquet_stability.pdf` | `figure3.pdf` | Floquet linear stability diagrams ($h_a=2, 3$) | `python3 reproduction/plot_figure3_floquet.py` |
| **Fig. 4** | `fig:wall_metrics` | `fig4_wall_metrics_ha3.pdf` | `sweep_ha_3.pdf` | Extended DW velocities & roughness vs $h_0$ ($h_a=3$) | `python3 reproduction/plot_figure4_wall_metrics.py` |
| **Fig. 5** | `fig:fases` | `fig5_phase_diagram_L256.pdf` | `phasediagram_L256.pdf` | Full 5-phase diagram in $(h_0, h_a)$ ($L=256$) | `python3 reproduction/plot_figure5_phase_diagram.py` |
| **Fig. 6** | `fig:estructura` | `fig6_edwards_wilkinson_scaling.pdf` | `NC_EW_inset.pdf` | Edwards-Wilkinson scaling collapse (NR phase) | `python3 reproduction/plot_figure6_nc_ew.py` |
| **Fig. 7** | `fig:sc_cascade` | `fig7_sp_phase_cascade.pdf` | `sc_cascade.pdf` | SP phase real-space spatiotemporal cascade | `python3 reproduction/plot_figure7_sp_cascade.py` |
| **Fig. 8** | `fig:estructuraSCvsL` | `fig8_sp_finite_size_scaling.pdf` | `SC_scaling.pdf` | Finite-size scaling vs $L$ in SP phase | `python3 reproduction/plot_figure8_sc_scaling.py` |
| **Fig. 9** | `fig:phasediagram_smallL` | `fig9_phase_diagram_L8.pdf` | `phasediagram_L8.pdf` | Small-system phase diagram ($L=8$, mode quantization) | `python3 reproduction/plot_figure9_phase_diagram_L8.py` |

---

## 3. Common Options for Plotting Scripts

Every script in this directory accepts standard options:

- `--export-paper`: Automatically copies the generated PDF directly into `paper/figuras/` for LaTeX compilation.
- `--out-dir <path>`: Specifies a custom output directory (default: `figures/`).
- `--data-dir <path>`: Points the script to newly regenerated simulation data instead of the pre-computed reference data in `output/`.

### Special Flags
- **Figure 3 (Floquet Stability)**:
  By default, uses pre-computed stability grids. To solve the Floquet monodromy matrix numerically on the fly:
  ```bash
  python3 reproduction/plot_figure3_floquet.py --recompute --h0-pts 300 --kappa-pts 200
  ```

---

## 4. Re-running with Modified Parameters

If you have regenerated simulation data with customized parameters using `./scripts/generate_figX_data.sh`, plot your new results by pointing `--data-dir` to the output folder:

```bash
# Example for Figure 4:
python3 reproduction/plot_figure4_wall_metrics.py \
    --data-dir output/sweep_ha3_L8_L1024_recomputed \
    --out-dir figures/custom_run \
    --export-paper
```
