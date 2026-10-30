#!/usr/bin/env python3
r"""
plot_stationary_roughness_scaling.py

Generates a PRB publication-ready figure demonstrating the stationary regime
Edwards-Wilkinson scaling of domain wall roughness:
    w^2_sat \propto (L/L_0)^{2\zeta} = (L/L_0)^1  (\zeta = 1/2)
for systems in the Non-synchronized Rough (NR) phase (h0=5.0, ha=3.0, alpha=0.27, omega=1.0).

Panels:
  (a) Transient roughness w^2(t) vs time t across multiple system sizes L/L_0 \in [32, 1024].
      Shows the early t^{1/2} dynamic growth followed by saturation into stationary plateaus
      for t \gg (L/L_0)^2.
  (b) Saturated roughness w^2_sat vs L/L_0 on log-log scale.
      - Saturated systems (L \le 256, t_max \gg L^2) obey w^2_sat \propto L (2\zeta \approx 1.0).
      - Large systems (L \ge 512) where t_max < L^2 have not yet reached saturation.
"""

import os
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
        'legend.fontsize': 7.8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def load_data(base_dir='output/scaling_NC'):
    base_path = Path(base_dir)
    dirs = sorted(base_path.glob('main_NC_scaling_*'))
    
    summary_data = []
    transient_data = {}
    
    for d in dirs:
        pfile = d / 'parameters.txt'
        sfile = d / 'time_data_final_summary.txt'
        tfile = d / 'time_data_transient_metrics.txt'
        
        L = None
        if pfile.exists():
            for line in open(pfile):
                if 'L:' in line:
                    L = float(line.split(':')[1])
        
        if L is None:
            continue
            
        var_u = np.nan
        if sfile.exists():
            lines = open(sfile).readlines()
            if len(lines) > 1:
                var_u = float(lines[1].split()[1])
                
        if tfile.exists():
            try:
                t_arr = np.loadtxt(tfile, skiprows=1)
                if t_arr.ndim > 1 and t_arr.shape[0] > 1:
                    t = t_arr[:, 1]
                    w2 = t_arr[:, 2]
                    mask = (t > 0) & (w2 > 0)
                    transient_data[L] = (t[mask], w2[mask])
            except Exception:
                pass
                
        summary_data.append({'L': L, 'var_u': var_u})
        
    summary_data.sort(key=lambda x: x['L'])
    return summary_data, transient_data

def plot_stationary_scaling(summary_data, transient_data, out_pdf, out_png=None):
    setup_prb_style()
    
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.85, 2.75))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.92, wspace=0.28)
    
    # -------------------------------------------------------------------------
    # Panel (a): Transient roughness w^2(t) vs time t
    # -------------------------------------------------------------------------
    selected_sizes = [32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    colors = cm.viridis(np.linspace(0.1, 0.9, len(selected_sizes)))
    
    for idx, L in enumerate(selected_sizes):
        if L in transient_data:
            t, w2 = transient_data[L]
            # Plot transient curve
            ax_a.loglog(t, w2, marker='o', markersize=3.2, linewidth=1.0,
                        color=colors[idx], label=f'$L/L_0={int(L)}$')
            
    # Reference EW growth: w^2 \propto t^{1/2}
    t_ref = np.logspace(2.0, 4.0, 50)
    w2_ref = 0.05 * np.sqrt(t_ref)
    ax_a.loglog(t_ref, w2_ref, 'k--', linewidth=1.2, label=r'$\propto t^{1/2}$')
    
    ax_a.set_xlabel(r'Time $t$')
    ax_a.set_ylabel(r'Roughness $w^2(t)$')
    ax_a.set_xlim(1.0, 1.5e5)
    ax_a.set_ylim(5e-5, 35.0)
    ax_a.legend(loc='upper left', ncol=2, borderaxespad=0.3, handlelength=1.4, columnspacing=0.8)
    ax_a.text(0.04, 0.06, r'\textbf{(a)}', transform=ax_a.transAxes, fontweight='bold', fontsize=9.5)
    
    # -------------------------------------------------------------------------
    # Panel (b): Saturated roughness w^2_sat vs L/L_0
    # -------------------------------------------------------------------------
    L_all = np.array([d['L'] for d in summary_data])
    w2_all = np.array([d['var_u'] for d in summary_data])
    
    # Saturated regime: L in [32, 256] where t_max = 10^5 \gg L^2
    sat_mask = (L_all >= 32.0) & (L_all <= 256.0)
    unsat_mask = (L_all >= 512.0) & (L_all <= 4096.0)
    flat_mask = (L_all <= 16.0)
    
    # Power-law fit over saturated regime
    log_L_sat = np.log(L_all[sat_mask])
    log_w2_sat = np.log(w2_all[sat_mask])
    poly = np.polyfit(log_L_sat, log_w2_sat, 1)
    two_zeta_fit = poly[0]
    amp_fit = np.exp(poly[1])
    
    # Theoretical EW line w^2_sat \approx 0.038 L
    L_line = np.logspace(1.4, 3.6, 50)
    w2_ew_line = 0.038 * L_line
    ax_b.loglog(L_line, w2_ew_line, 'k--', linewidth=1.2,
                label=r'EW: $w^2_\mathrm{sat} \approx 0.038\,(L/L_0)$')
    
    # Best-fit line
    w2_fit_line = amp_fit * (L_line ** two_zeta_fit)
    ax_b.loglog(L_line, w2_fit_line, 'r:', linewidth=1.3,
                label=rf'Fit: $(L/L_0)^{{{two_zeta_fit:.2f}}}$ ($2\zeta \approx 1$)')
    
    # Plot saturated data points (filled circles)
    ax_b.plot(L_all[sat_mask], w2_all[sat_mask], 'o', color='#1f77b4',
              markersize=5.5, label=r'Saturated ($t \gg L^2$)')
    
    # Plot un-saturated data points (open circles)
    ax_b.plot(L_all[unsat_mask], w2_all[unsat_mask], 's', color='#7f7f7f',
              markerfacecolor='none', markeredgewidth=1.2, markersize=5.0,
              label=r'Pre-asymptotic ($t < L^2$)')
    
    ax_b.set_xlabel(r'$L/L_0$')
    ax_b.set_ylabel(r'Steady-state roughness $\langle w^2 \rangle_t$')
    ax_b.set_xlim(20.0, 5000.0)
    ax_b.set_ylim(0.4, 80.0)
    ax_b.legend(loc='lower right', borderaxespad=0.3, handlelength=1.4)
    ax_b.text(0.04, 0.90, r'\textbf{(b)}', transform=ax_b.transAxes, fontweight='bold', fontsize=9.5)
    
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_pdf}")

def main():
    data_dir = 'output/scaling_NC'
    out_dir = Path('figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data, transient_data = load_data(data_dir)
    out_pdf = out_dir / 'stationary_roughness_scaling.pdf'
    out_png = out_dir / 'stationary_roughness_scaling.png'
    
    plot_stationary_scaling(summary_data, transient_data, str(out_pdf), str(out_png))
    
    # Also save to paper/figuras for convenience
    paper_pdf = Path('paper/figuras/stationary_roughness_scaling.pdf')
    plot_stationary_scaling(summary_data, transient_data, str(paper_pdf))

if __name__ == '__main__':
    main()

