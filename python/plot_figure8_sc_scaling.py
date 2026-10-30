#!/usr/bin/env python3
r"""
plot_figure8_sc_scaling.py

Generates PRB publication-ready Figure 8 (fig:estructuraSCvsL):
Finite-size analysis of the synchronized corrugated (SC / SP) phase.

Ale's corrections applied:
  1. Panel (a) y-axis: Changed label from w^2 to \langle w^2 \rangle_t (matching Eq. \eqref{eq:roughnessvst}).
  2. Panel (b) x-axis: Changed label from L to L/L_0.
  3. Panel (b) legend: Changed fundamental discrete mode label from 2\pi/L to 2\pi/(L/L_0).
  4. Converged multi-batch dataset from the cluster (/mnt/cluster/inv1/proyecto-v2/output/scaling)
     used, resolving the transient noise artifact at L/L_0 = 16 and 64 and showing exact
     monotonic adherence to the 2\pi/(L/L_0) dispersion relation in the flat regime.

Panels:
  (a) DW steady-state roughness \langle w^2 \rangle_t vs system length L/L_0.
      Values in the flat regime (\langle w^2 \rangle_t = 0 for L/L_0 \le 64) are shown
      at the 10^-7 floor for visualization, matching the convention in Figure 4.
  (b) Dominant wavenumber \kappa_\mathrm{peak} vs system length L/L_0.
      - Theoretical discrete fundamental mode: 2\pi / (L/L_0) (black dashed line).
      - Floquet maximum growth mode: \kappa^* \approx 0.833375 (black dotted line).
"""

import os
import re
import argparse
import tempfile
from pathlib import Path
import numpy as np

# Set writable matplotlib cache directory
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# PRB Style Setup
# ==============================================================================
def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 10.0,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 3.8,
        'ytick.major.size': 3.8,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.75,
        'ytick.major.width': 0.75,
        'xtick.minor.width': 0.55,
        'ytick.minor.width': 0.55,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.45,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def load_dataset(base_dir, suffix='(1)'):
    """
    Loads scaling data from directory.
    Default suffix '(1)' selects the fully relaxed 4-batch average run (Sep 9).
    suffix='' selects the initial run (Aug 24).
    """
    results = []
    base_path = Path(base_dir)
    pattern = "main_SC_scaling_*"
    
    for d in sorted(base_path.glob(pattern)):
        if not d.is_dir():
            continue
        if suffix == '(1)' and not d.name.endswith('(1)'):
            continue
        if suffix == '' and d.name.endswith('(1)'):
            continue
            
        m = re.search(r'main_SC_scaling_([0-9.]+)_', d.name)
        if not m:
            continue
        L = float(m.group(1))
        
        sum_file = d / 'time_data_final_summary.txt'
        ps_file = d / 'time_data_final_ps.txt'
        
        var_u = np.nan
        k_max = np.nan
        
        if sum_file.exists():
            s = np.genfromtxt(sum_file, names=True)
            if s.size > 0:
                if 'var_u' in s.dtype.names:
                    var_u = float(np.mean(s['var_u']))
                elif s.ndim > 0:
                    var_u = float(s[0][1])
                    
        if ps_file.exists():
            ps = np.genfromtxt(ps_file, names=True)
            if ps.size > 0:
                if 'batch' in ps.dtype.names:
                    batches = np.unique(ps['batch'])
                    k_vals = ps[ps['batch'] == batches[0]]['k']
                    powers = [ps[ps['batch'] == b]['power'] for b in batches]
                    mean_p = np.mean(powers, axis=0)
                    mask = k_vals > 0
                    k_vals, mean_p = k_vals[mask], mean_p[mask]
                    if len(k_vals) > 0:
                        k_max = float(k_vals[np.argmax(mean_p)])
                else:
                    k = ps['k'] if 'k' in ps.dtype.names else ps[ps.dtype.names[0]]
                    p = ps['power'] if 'power' in ps.dtype.names else ps[ps.dtype.names[1]]
                    mask = k > 0
                    k, p = k[mask], p[mask]
                    if len(k) > 0:
                        k_max = float(k[np.argmax(p)])
                        
        results.append({'L': L, 'var_u': var_u, 'k_max': k_max})
        
    results.sort(key=lambda x: x['L'])
    return results


def plot_figure8(data, out_pdf, out_png=None, clip_floor=1e-7, kappa_star=0.833375, open_floor_markers=False):
    """
    Renders PRB Figure 8 with the exact publication geometry and Ale's corrections.
    """
    setup_prb_style()
    
    L = np.array([d['L'] for d in data])
    var_u = np.array([d['var_u'] for d in data])
    k_max = np.array([d['k_max'] for d in data])
    
    # Target figure dimensions: 3.32 x 2.90 in (PRB single-column)
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.32, 2.90), sharex=True)
    fig.subplots_adjust(left=0.183, right=0.97, bottom=0.160, top=0.965, hspace=0.05)
    
    # -------------------------------------------------------------------------
    # Panel (a): DW steady-state roughness \langle w^2 \rangle_t vs L/L_0
    # -------------------------------------------------------------------------
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    if clip_floor is not None:
        plot_var = np.maximum(var_u, clip_floor)
        if open_floor_markers:
            is_floor = var_u <= clip_floor
            # Open markers for flat floor points
            ax_a.plot(L[is_floor], plot_var[is_floor], 'o', color='#1f77b4',
                      markerfacecolor='white', markeredgewidth=1.0, markersize=4.5,
                      linestyle='none', zorder=3)
            # Filled markers for finite roughness points
            ax_a.plot(L[~is_floor], plot_var[~is_floor], 'o', color='#1f77b4',
                      markerfacecolor='#1f77b4', markersize=4.5,
                      linestyle='none', zorder=3)
        else:
            ax_a.plot(L, plot_var, color='#1f77b4', marker='o', markersize=4.5,
                      linestyle='none', clip_on=True, zorder=3)
    else:
        valid = var_u > 0
        ax_a.plot(L[valid], var_u[valid], color='#1f77b4', marker='o', markersize=4.5,
                  linestyle='none', clip_on=True, zorder=3)
                    
    ax_a.set_ylabel(r'$\langle w^2 \rangle_t$', labelpad=2)
    ax_a.set_ylim(bottom=4e-9, top=0.2)
    ax_a.tick_params(axis='x', which='both', labelbottom=False)
    
    # -------------------------------------------------------------------------
    # Panel (b): Dominant wavenumber \kappa_\mathrm{peak} vs L/L_0
    # -------------------------------------------------------------------------
    valid_k = k_max > 0

    # Theoretical discrete fundamental mode: 2\pi / (L/L_0) (drawn with high visibility)
    L_ref = np.logspace(0, np.log10(128), 100)
    k_ref = 2.0 * np.pi / L_ref
    ax_b.loglog(L_ref, k_ref, 'k--', linewidth=1.2, dashes=(4, 2), label=r'$2\pi/(L/L_0)$', zorder=4)

    # Floquet maximum growth rate mode \kappa^*
    ax_b.axhline(kappa_star, color='k', linestyle=':', linewidth=1.3, label=r'$\kappa^*$', zorder=4)

    # Plot data points without connecting lines so the theoretical dashed line remains unobstructed
    ax_b.plot(L[valid_k], k_max[valid_k], color='#d62728', marker='^', markersize=4.8,
              linestyle='none', clip_on=True, zorder=3)
    
    ax_b.set_xlabel(r'$L/L_0$', labelpad=2)
    ax_b.set_ylabel(r'$\kappa_\mathrm{peak}$', labelpad=2)
    ax_b.set_xlim(left=0.55, right=4.8e5)
    ax_b.set_ylim(bottom=0.08, top=7.7)
    
    ax_b.legend(loc='upper right', frameon=False, borderaxespad=0.6,
                handlelength=1.6, labelspacing=0.25)
                
    # Save files
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf)
    if out_png:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate PRB publication-ready Figure 8.")
    parser.add_argument("--data-dir", default="output/scaling_data",
                        help="Path to scaling data directory (default: output/scaling_data)")
    parser.add_argument("--cluster-dir", default="/mnt/cluster/inv1/proyecto-v2/output/scaling",
                        help="Optional fallback to cluster directory")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory (default: figures)")
    parser.add_argument("--export-paper", action="store_true",
                        help="Also export to paper/figuras/")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.isdir(data_dir) and os.path.isdir(args.cluster_dir):
        data_dir = args.cluster_dir
        
    print(f"Loading scaling data from: {data_dir}")
    
    # Load converged dataset (Sep 9, 4-batch)
    sep_data = load_dataset(data_dir, suffix='(1)')
    # Load initial dataset (Aug 24, single run)
    aug_data = load_dataset(data_dir, suffix='')
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary PRB Figure (Sep 9 data with 10^-7 floor for flat state)
    plot_figure8(sep_data, str(out_dir / "SC_scaling.pdf"), str(out_dir / "SC_scaling.png"), clip_floor=1e-7)

    if args.export_paper:
        paper_dest = Path("paper/figuras/SC_scaling.pdf")
        paper_dest.parent.mkdir(parents=True, exist_ok=True)
        plot_figure8(sep_data, str(paper_dest), clip_floor=1e-7)
        print(f"Exported to paper figures: {paper_dest}")
    
    # 2. Variants for comparison saved to out_dir:
    # 2a. Open markers at floor
    plot_figure8(sep_data, str(out_dir / "SC_scaling_open_markers.pdf"), str(out_dir / "SC_scaling_open_markers.png"),
                 clip_floor=1e-7, open_floor_markers=True)
    # 2b. Strictly positive / unclipped
    plot_figure8(sep_data, str(out_dir / "SC_scaling_no_floor.pdf"), str(out_dir / "SC_scaling_no_floor.png"),
                 clip_floor=None)
    # 2c. Original Aug 24 dataset with Ale's corrections
    plot_figure8(aug_data, str(out_dir / "SC_scaling_aug.pdf"), str(out_dir / "SC_scaling_aug.png"),
                 clip_floor=None)
                 
    print("Done generating all Figure 8 variants!")


if __name__ == '__main__':
    main()
