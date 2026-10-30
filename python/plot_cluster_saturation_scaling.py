#!/usr/bin/env python3
r"""
plot_cluster_saturation_scaling.py

Processes output from `scripts/generate_saturation_scaling_gpu.sh`:
Plots saturated interface roughness \langle w^2 \rangle_t vs L/L_0 across all sizes
L \in [8, 8192] to analyze the true Edwards-Wilkinson / KPZ saturation exponent.
"""

import os
import argparse
import tempfile
from pathlib import Path
import numpy as np

# Ensure writable matplotlib cache dir
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

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
        tfile = d / "time_data_transient_metrics.txt"
        
        L = None
        if pfile.exists():
            for line in open(pfile):
                if line.startswith("L:"):
                    L = float(line.split(":")[1])
                    
        if L is None:
            continue
            
        var_u = np.nan
        is_complete = False
        t_current = np.nan
        
        if sfile.exists():
            lines = [l.strip() for l in open(sfile) if l.strip()]
            if len(lines) > 1:
                parts = lines[1].split()
                # batch total_steps var_u u_dot phi_dot spectral_entropy
                if len(parts) >= 3:
                    try:
                        var_u = float(parts[2])
                        is_complete = True
                    except ValueError:
                        pass
                        
        if not is_complete and tfile.exists():
            tlines = [l.strip().split() for l in open(tfile) if l.strip()]
            if len(tlines) > 1:
                try:
                    t_current = float(tlines[-1][2])
                    var_u = float(tlines[-1][3])
                except (ValueError, IndexError):
                    pass
                
        results.append({
            'L': L,
            'var_u': var_u,
            'is_complete': is_complete,
            't_current': t_current,
            'dir': str(d)
        })
        
    results.sort(key=lambda x: x['L'])
    return results

def plot_saturation_results(results, out_pdf, out_png=None):
    setup_prb_style()
    
    comp_L = np.array([r['L'] for r in results if r['is_complete'] and not np.isnan(r['var_u'])])
    comp_w2 = np.array([r['var_u'] for r in results if r['is_complete'] and not np.isnan(r['var_u'])])
    
    run_L = np.array([r['L'] for r in results if not r['is_complete'] and not np.isnan(r['var_u'])])
    run_w2 = np.array([r['var_u'] for r in results if not r['is_complete'] and not np.isnan(r['var_u'])])
    
    if len(comp_L) == 0 and len(run_L) == 0:
        print("No valid saturation data found.")
        return
        
    fig, ax = plt.subplots(figsize=(3.375, 2.85))
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.15, top=0.95)
    
    # Reference Edwards-Wilkinson line w^2 = 0.038 * L
    L_line = np.logspace(np.log10(6.0), np.log10(10000.0), 100)
    ew_curve = 0.038 * (L_line ** 1.0)
    ax.loglog(L_line, ew_curve, 'k--', linewidth=1.1, label=r'EW: $w^2_\mathrm{sat} \approx 0.038\,(L/L_0)$')
    
    # Separate completed into flat (L <= 16) and rough (L >= 32)
    rough_mask = (comp_w2 > 1e-4) & (comp_L >= 30.0)
    flat_mask = ~rough_mask
    
    if np.any(rough_mask):
        L_fit = comp_L[rough_mask]
        w2_fit = comp_w2[rough_mask]
        
        # Fit power law w^2_sat \propto L^{2\zeta}
        poly = np.polyfit(np.log(L_fit), np.log(w2_fit), 1)
        two_zeta = poly[0]
        amp = np.exp(poly[1])
        
        fit_curve = amp * (L_line ** two_zeta)
        ax.loglog(L_line, fit_curve, 'r:', linewidth=1.2,
                  label=rf'Fit: $(L/L_0)^{{{two_zeta:.2f}}}$ ($2\zeta={two_zeta:.2f}$)')
        
        ax.plot(L_fit, w2_fit, 'o', color='#1f77b4', markersize=5.2,
                label=r'Converged $w^2_\mathrm{sat}$ ($t \gg L^2$)')
                
    if np.any(flat_mask):
        ax.plot(comp_L[flat_mask], np.maximum(comp_w2[flat_mask], 1e-6), 'o',
                color='#1f77b4', markerfacecolor='white', markeredgewidth=1.0,
                markersize=4.8, label=r'Flat regime ($L \leq 16$)')
                
    if len(run_L) > 0:
        ax.plot(run_L, run_w2, 's', color='#ff7f0e', markerfacecolor='#ffbb78',
                markeredgewidth=0.8, markersize=4.8,
                label=r'Running transient (WIP)')
                
    ax.set_xlabel(r'System length $L/L_0$')
    ax.set_ylabel(r'Interface roughness $\langle w^2 \rangle_t$')
    ax.set_xlim(5.0, 1.2e4)
    ax.set_ylim(2e-7, 100.0)
    ax.legend(loc='lower right', frameon=False, borderaxespad=0.3, handlelength=1.4)
    
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

