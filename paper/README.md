# Publication Manuscript & Supplementary Material

This directory contains the LaTeX manuscript, bibliography, supplementary materials, and publication-ready figures for the study:

> **Synchronization and Spatiotemporal Phases in Driven Precessing Domain Walls**  
> Luz Pérez Herrera, E. F. Pusiol, et al.

---

## Directory Organization

```
paper/
├── Makefile                     # Build automation (pdflatex + bibtex)
├── main.tex                     # Primary manuscript source
├── main.pdf                     # Compiled manuscript
├── refs.bib                     # BibTeX references
├── supplement.tex               # Supplementary Material (e.g. video descriptions)
├── supplement.pdf               # Compiled Supplementary Material
├── supporting_information.tex   # Analytical derivations (diagonalization, secular Hill expansion)
├── .gitignore                   # Ignore LaTeX temporary build artifacts
└── figuras/                     # Publication figures (9 active figures)
    ├── snap_system_345.pdf      # Fig. 1: DW coordinate system & schematic
    ├── figure2.pdf              # Fig. 2: Rigid domain wall synchronization / Shapiro steps
    ├── figure3.pdf              # Fig. 3: Floquet linear stability spectrum and stability maps
    ├── sweep_ha_3.pdf           # Fig. 4: Flexible extended DW dynamics (velocity & roughness sweeps)
    ├── phasediagram_L256.pdf    # Fig. 5: Large-system dynamical phase diagram (L/L0 = 256)
    ├── NC_EW_inset.pdf          # Fig. 6: Non-synchronized rough (NR) phase Edwards-Wilkinson scaling
    ├── sc_cascade.pdf           # Fig. 7: Synchronized periodic (SP) phase structure factor & cascade inhibition
    ├── SC_scaling.pdf           # Fig. 8: Synchronized periodic (SP) phase roughness & mode scaling
    └── phasediagram_L8.pdf      # Fig. 9: Small-system dynamical phase diagram (L/L0 = 8, finite size effects)
```

---

## Figures Mapping

| Figure # | File in `figuras/` | Label in `main.tex` | Source / Generation Script |
|---|---|---|---|
| **Fig. 1** | `snap_system_345.pdf` | `fig:snap` | Schematic of DW coordinate system $u(x,t)$ and $\phi(x,t)$ |
| **Fig. 2** | `figure2.pdf` | `fig:ej_walker` | Rigid wall velocities and Shapiro synchronization steps |
| **Fig. 3** | `figure3.pdf` | `fig:sync_stability` | Linear Floquet stability diagrams for $h_\mathrm{a}=2$ and $h_\mathrm{a}=3$ |
| **Fig. 4** | `sweep_ha_3.pdf` | `fig:wall_metrics` | `python/plot_figure4_wall_metrics.py` |
| **Fig. 5** | `phasediagram_L256.pdf` | `fig:fases` | `python/plot_prb_publication.py --export-paper` |
| **Fig. 6** | `NC_EW_inset.pdf` | `fig:estructura` | `python/plot_figure6_nc_ew.py` |
| **Fig. 7** | `sc_cascade.pdf` | `fig:sc_cascade` | `python/plot_scaling.py` |
| **Fig. 8** | `SC_scaling.pdf` | `fig:estructuraSCvsL` | `python/plot_figure8_sc_scaling.py` |
| **Fig. 9** | `phasediagram_L8.pdf` | `fig:phasediagram_smallL` | `python/plot_prb_publication.py --export-paper` |

*Note: Exploratory figure variations, side-by-side versions, and comparison plots are stored in the project-level `figures/` directory.*

---

## Building the Documents

Using `make`:

```bash
# Build both main.pdf and supplement.pdf (runs pdflatex + bibtex passes)
make

# Verify that all 9 required publication figures are present
make check-figs

# Clean auxiliary LaTeX files (*.aux, *.log, *.out, *.toc, *.bbl, etc.)
make clean

# Remove all auxiliary files and compiled PDFs for a clean rebuild
make distclean
```

