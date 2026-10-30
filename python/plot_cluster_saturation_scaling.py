#!/usr/bin/env python3
r"""
plot_cluster_saturation_scaling.py

Processes output from `scripts/generate_saturation_scaling_gpu.sh`:
Plots saturated interface roughness \langle w^2 \rangle_t vs L/L_0 across all sizes
L \in [8, 8192] to analyze the true Edwards-Wilkinson / KPZ saturation exponent.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 9.5,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 1.8,
        'ytick.minor.size': 1.8,
        'legend.fontsize': 8.0,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def load_saturation_data(data_dir):
    base_path = Path(data_dir)
    dirs = sorted(base_path.glob("main_NC_sat_*"))
    
    results = []
    for d in dirs:
        pfile = d / "parameters.txt"
        sfile = d / "time_data_final_summary.txt"
        
        L = None
        if pfile.exists():
            for line in open(pfile):
                if line.startswith("L:"):
                    L = float(line.split(":")[1])
                    
        if L is None:
            continue
            
        var_u = np.nan
        if sfile.exists():
            lines = open(sfile).readlines()
            if len(lines) > 1:
                var_u = float(lines[1].split()[1])
                
        results.append({'L': L, 'var_u': var_u, 'dir': str(d)})
        
    results.sort(key=lambda x: x['L'])
    return results

def plot_saturation_results(results, out_pdf, out_png=None):
    setup_prb_style()
    
    L_vals = np.array([r['L'] for r in results if not np.isnan(r['var_u'])])
    w2_vals = np.array([r['var_u'] for r in results if not np.isnan(r['var_u'])])
    
    if len(L_vals) == 0:
        print("No valid saturation data found.")
        return
        
    fig, ax = plt.subplots(figsize=(3.375, 2.7))
    
    # Separate flat state from rough state
    flat_mask = (w2_vals < 1e-4) | (L_vals <= 16.0)
    rough_mask = ~flat_mask
    
    if np.any(rough_mask):
        L_fit = L_vals[rough_mask]
        w2_fit = w2_vals[rough_mask]
        
        # Fit power law w^2_sat \propto L^{2\zeta}
        poly = np.polyfit(np.log(L_fit), np.log(w2_fit), 1)
        two_zeta = poly[0]
        amp = np.exp(poly[1])
        
        L_line = np.logspace(np.log10(L_fit.min() * 0.8), np.log10(L_fit.max() * 1.2), 100)
        fit_curve = amp * (L_line ** two_zeta)
        ew_curve = 0.038 * (L_line ** 1.0)
        
        ax.loglog(L_line, ew_curve, 'k--', linewidth=1.1, label=r'EW: $\propto L^{1.0}$')
        ax.loglog(L_line, fit_curve, 'r:', linewidth=1.2,
                  label=rf'Fit: $\propto L^{{{two_zeta:.2f}}}$ ($2\zeta={two_zeta:.2f}$)')
        
        ax.plot(L_fit, w2_fit, 'o', color='#1f77b4', markersize=5.0,
                label=r'Simulated $w^2_\mathrm{sat}$')
                
    if np.any(flat_mask):
        ax.plot(L_vals[flat_mask], np.maximum(w2_vals[flat_mask], 1e-6), 'o',
                color='#1f77b4', markerfacecolor='white', markeredgewidth=1.0,
                markersize=4.5, label='Flat (below onset)')
                
    ax.set_xlabel(r'System length $L/L_0$')
    ax.set_ylabel(r'Saturated roughness $\langle w^2 \rangle_t$')
    ax.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Analyze cluster saturation scaling data.")
    parser.add_argument("--data-dir", default="output/saturation_scaling_gpu",
                        help="Path to saturation directory")
    parser.add_argument("--out-dir", default="figures",
                        help="Output figure directory")
    args = parser.parse_args()
    
    results = load_saturation_data(args.data_dir)
    print(f"Found {len(results)} system size outputs:")
    for r in results:
        print(f"  L = {r['L']:7.1f} | w^2_sat = {r['var_u']}")
        
    out_dir = Path(args.out_dir)
    out_pdf = out_dir / "cluster_saturation_scaling.pdf"
    out_png = out_dir / "cluster_saturation_scaling.png"
    plot_saturation_results(results, str(out_pdf), str(out_png))

if __name__ == '__main__':
    main()
