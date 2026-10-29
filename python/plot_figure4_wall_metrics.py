#!/usr/bin/env python3
r"""
plot_figure4_wall_metrics.py

Generates PRB publication-ready Figure 4 (fig:wall_metrics):
Flexible extended domain wall dynamics under combined DC + AC driving (h_a = 3, Omega = 1, alpha = 0.27).

Requirements:
  1) No panel labels (a), (b), (c) on the plot itself (added via LaTeX overpic or caption).
  2) Variable symbols strictly for axis labels:
     - x-axis: $h_0$
     - y-axes: $\langle \dot{u} \rangle_{x,t}$, $\langle \dot{\phi} \rangle_{x,t}$, $\langle w^2 \rangle_t$
  3) Supports two alternatives:
     - Alternative A: Caption-only (no in-plot legend; completely clean panels).
     - Alternative B: In-plot legend (compact legend in panel b).

Panels:
  Top:    Mean DW translational velocity $\langle \dot{u} \rangle_{x,t}$ vs $h_0$
  Middle: Mean precession / angular velocity $\langle \dot{\phi} \rangle_{x,t}$ vs $h_0$
  Bottom: Interface roughness $\langle w^2 \rangle_t$ vs $h_0$ (clipped at 10^-6)

Extended wall sizes: L/L_0 = 8 and L/L_0 = 1024, together with the rigid-wall solution.
Background shading: White = Synchronized rigid DW, Gray = Unsynchronized rigid DW.
"""

import os
import sys
import glob
import argparse
import tempfile
import numpy as np

# Set writable matplotlib cache
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        'legend.fontsize': 8.0,
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '0.85',
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.45,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

# ==============================================================================
# Rigid Wall Sweep with Caching
# ==============================================================================
CACHE_FILE = os.path.join(tempfile.gettempdir(), "rigid_wall_ha3_cache.npz")

def compute_rigid_wall_sweep(h0_grid, alpha=0.27, omega=1.0, ha=3.0, dt=0.019634954,
                             trans_cycles=150, eval_cycles=150):
    if os.path.isfile(CACHE_FILE):
        try:
            cached = np.load(CACHE_FILE)
            if np.allclose(cached['h0_grid'], h0_grid) and cached['ha'] == ha and cached['alpha'] == alpha:
                return cached['vu_rigid'], cached['vphi_rigid']
        except Exception:
            pass

    T_lock = 4.0 * np.pi / omega
    steps_per_cycle = int(round(T_lock / dt))
    dt_act = T_lock / steps_per_cycle
    
    n_h0 = len(h0_grid)
    phi = np.zeros(n_h0)
    
    trans_steps = trans_cycles * steps_per_cycle
    eval_steps = eval_cycles * steps_per_cycle
    
    t = 0.0
    for _ in range(trans_steps):
        h1 = h0_grid + ha * np.cos(omega * t)
        h2 = h0_grid + ha * np.cos(omega * (t + 0.5 * dt_act))
        h3 = h0_grid + ha * np.cos(omega * (t + dt_act))
        
        k1 = 0.5 * (alpha * h1 - alpha * np.sin(2.0 * phi))
        k2 = 0.5 * (alpha * h2 - alpha * np.sin(2.0 * (phi + 0.5 * dt_act * k1)))
        k3 = 0.5 * (alpha * h2 - alpha * np.sin(2.0 * (phi + 0.5 * dt_act * k2)))
        k4 = 0.5 * (alpha * h3 - alpha * np.sin(2.0 * (phi + dt_act * k3)))
        
        phi += (dt_act / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt_act
        
    phi_start = phi.copy()
    sum_vu = np.zeros(n_h0)
    
    for _ in range(eval_steps):
        h1 = h0_grid + ha * np.cos(omega * t)
        h2 = h0_grid + ha * np.cos(omega * (t + 0.5 * dt_act))
        h3 = h0_grid + ha * np.cos(omega * (t + dt_act))
        
        k1 = 0.5 * (alpha * h1 - alpha * np.sin(2.0 * phi))
        k2 = 0.5 * (alpha * h2 - alpha * np.sin(2.0 * (phi + 0.5 * dt_act * k1)))
        k3 = 0.5 * (alpha * h2 - alpha * np.sin(2.0 * (phi + 0.5 * dt_act * k2)))
        k4 = 0.5 * (alpha * h3 - alpha * np.sin(2.0 * (phi + dt_act * k3)))
        
        phi_mid = phi + 0.5 * dt_act * (k1 + k2) / 2.0
        sum_vu += 0.5 * (alpha**2 * h2 + np.sin(2.0 * phi_mid))
        phi += (dt_act / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt_act
        
    total_eval_time = eval_steps * dt_act
    vu_rigid = sum_vu / eval_steps
    vphi_rigid = (phi - phi_start) / total_eval_time
    
    try:
        np.savez(CACHE_FILE, h0_grid=h0_grid, vu_rigid=vu_rigid, vphi_rigid=vphi_rigid,
                 ha=ha, alpha=alpha, omega=omega)
    except Exception:
        pass
        
    return vu_rigid, vphi_rigid

# ==============================================================================
# Helper to Load Sweep Output Directory
# ==============================================================================
def load_sweep_data(dir_path):
    files = sorted(glob.glob(os.path.join(dir_path, "results_h0_*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(dir_path, "results_*.txt")))
    if not files:
        return None
        
    records = []
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        try:
            d = np.genfromtxt(f, names=True)
            if d.size == 0:
                continue
            row = d[0] if d.ndim > 0 else d
            records.append((
                float(row['h0']),
                float(row['ha']),
                float(row['rugosity']),
                float(row['v_u']),
                float(row['v_phi']),
                float(row['mean_var_strobo']) if 'mean_var_strobo' in row.dtype.names else 0.0
            ))
        except Exception:
            continue
            
    if not records:
        return None
        
    records.sort(key=lambda r: r[0])
    data = np.array(records)
    return {
        'h0': data[:, 0],
        'ha': data[:, 1],
        'rugosity': data[:, 2],
        'v_u': data[:, 3],
        'v_phi': data[:, 4],
        'mean_var_strobo': data[:, 5]
    }

# ==============================================================================
# Plot Figure 4 Function
# ==============================================================================
def plot_figure_4(data_dir, output_file, include_legend=False):
    setup_prb_style()
    
    dir_l8 = os.path.join(data_dir, "L_8")
    dir_l1024 = os.path.join(data_dir, "L_1024")
    
    data_l8 = load_sweep_data(dir_l8)
    data_l1024 = load_sweep_data(dir_l1024)
    
    if data_l8 is None or data_l1024 is None:
        raise ValueError(f"Could not load L_8 or L_1024 data from {data_dir}")
        
    # Rigid wall solution on fine grid
    h0_fine = np.linspace(0.0, 8.0, 801)
    vu_rigid, vphi_rigid = compute_rigid_wall_sweep(h0_fine)
    
    # Synchronization condition: sigma = | <dot_phi> / (Omega/2) - round(<dot_phi> / (Omega/2)) | < tol
    sigma = np.abs(vphi_rigid / 0.5 - np.round(vphi_rigid / 0.5))
    is_sync = sigma < 2e-3
    
    # PRB Single-column dimensions (width: 3.375 in, height: ~4.7 in)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(3.375, 4.65), sharex=True)
    fig.subplots_adjust(hspace=0.065, top=0.985, bottom=0.095, left=0.175, right=0.965)
    
    # Background shading: White for sync, Light Gray for unsync
    def add_background_shading(ax):
        in_unsync = False
        start_h = 0.0
        for i in range(len(h0_fine)):
            if not is_sync[i] and not in_unsync:
                in_unsync = True
                start_h = h0_fine[i]
            elif is_sync[i] and in_unsync:
                in_unsync = False
                ax.axvspan(start_h, h0_fine[i], color='0.91', zorder=0, lw=0)
        if in_unsync:
            ax.axvspan(start_h, h0_fine[-1], color='0.91', zorder=0, lw=0)
            
    for ax in [ax_a, ax_b, ax_c]:
        add_background_shading(ax)
        ax.set_xlim(0.0, 8.0)
        ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)
        
    ax_a.tick_params(labelbottom=False)
    ax_b.tick_params(labelbottom=False)

    # Color palette
    c_rigid = 'black'
    c_l8 = '#1f77b4'    # PRB blue
    c_l1024 = '#d62728' # PRB red
    
    # --------------------------------------------------------------------------
    # Panel (a): Mean DW velocity <dot{u}>_{x,t}
    # --------------------------------------------------------------------------
    ax_a.plot(h0_fine, vu_rigid, color=c_rigid, linestyle='--', linewidth=1.1, zorder=3, label='Rigid wall')
    ax_a.plot(data_l8['h0'], data_l8['v_u'], color=c_l8, linestyle='-', linewidth=1.1, zorder=4, label=r'$L/L_0 = 8$')
    ax_a.plot(data_l1024['h0'], data_l1024['v_u'], color=c_l1024, linestyle='-', linewidth=1.1, zorder=5, label=r'$L/L_0 = 1024$')
    
    ax_a.set_ylabel(r'$\langle \dot{u} \rangle_{x,t}$')
    ax_a.set_ylim(-0.01, 0.49)
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax_a.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    # --------------------------------------------------------------------------
    # Panel (b): Mean precession velocity <dot{phi}>_{x,t}
    # --------------------------------------------------------------------------
    # Reference lines for Shapiro steps (Omega/2 = 0.5, Omega = 1.0)
    ax_b.axhline(0.5, color='0.45', linestyle=':', linewidth=0.75, zorder=1)
    ax_b.axhline(1.0, color='0.45', linestyle=':', linewidth=0.75, zorder=1)
    
    ax_b.plot(h0_fine, vphi_rigid, color=c_rigid, linestyle='--', linewidth=1.1, zorder=3, label='Rigid wall')
    ax_b.plot(data_l8['h0'], data_l8['v_phi'], color=c_l8, linestyle='-', linewidth=1.1, zorder=4, label=r'$L/L_0 = 8$')
    ax_b.plot(data_l1024['h0'], data_l1024['v_phi'], color=c_l1024, linestyle='-', linewidth=1.1, zorder=5, label=r'$L/L_0 = 1024$')
    
    ax_b.set_ylabel(r'$\langle \dot{\phi} \rangle_{x,t}$')
    ax_b.set_ylim(-0.03, 1.15)
    ax_b.set_yticks([0.0, 0.5, 1.0])
    ax_b.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    if include_legend:
        ax_b.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92,
                    edgecolor='0.85', borderpad=0.32, labelspacing=0.22, handlelength=1.5, handletextpad=0.4)

    # --------------------------------------------------------------------------
    # Panel (c): Roughness <w^2>_t (clipped at 10^-6)
    # --------------------------------------------------------------------------
    rug_l8 = np.maximum(data_l8['rugosity'], 1e-6)
    rug_l1024 = np.maximum(data_l1024['rugosity'], 1e-6)
    
    ax_c.plot(data_l8['h0'], rug_l8, color=c_l8, linestyle='-', linewidth=1.1, zorder=4)
    ax_c.plot(data_l1024['h0'], rug_l1024, color=c_l1024, linestyle='-', linewidth=1.1, zorder=5)
    
    ax_c.set_yscale('log')
    ax_c.set_ylim(3e-7, 2e3)
    ax_c.set_yticks([1e-6, 1e-4, 1e-2, 1e0, 1e2])
    ax_c.set_ylabel(r'$\langle w^2 \rangle_t$')
    ax_c.set_xlabel(r'$h_0$')
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax_c.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    
    # Save figure
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    fig.savefig(output_file)
    png_file = os.path.splitext(output_file)[0] + ".png"
    fig.savefig(png_file)
    print(f"Saved: {output_file} and {png_file}")
    plt.close(fig)

# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PRB Figure 4 variants.")
    parser.add_argument("--data-dir", default="output/sweep_ha3_L8_L1024_979329", help="Directory with sweep output")
    parser.add_argument("--out-dir", default="figures", help="Directory for output figures")
    args = parser.parse_args()
    
    # 1. Alternative A: Caption-only (no in-plot legend)
    out_cap_only = os.path.join(args.out_dir, "sweep_ha_3.003_caption_only.pdf")
    plot_figure_4(args.data_dir, out_cap_only, include_legend=False)
    
    # 2. Alternative B: In-plot legend
    out_with_legend = os.path.join(args.out_dir, "sweep_ha_3.003_with_legend.pdf")
    plot_figure_4(args.data_dir, out_with_legend, include_legend=True)
    
    # 3. Also update paper/figuras/sweep_ha_3.pdf (caption-only / clean, matching paper/main.tex)
    plot_figure_4(args.data_dir, "paper/figuras/sweep_ha_3.pdf", include_legend=False)
    
    # 4. Also copy to output/sweep_ha3_L8_L1024_979329/figures
    plot_figure_4(args.data_dir, "output/sweep_ha3_L8_L1024_979329/figures/figure4_caption_only.pdf", include_legend=False)
    plot_figure_4(args.data_dir, "output/sweep_ha3_L8_L1024_979329/figures/figure4_with_legend.pdf", include_legend=True)
