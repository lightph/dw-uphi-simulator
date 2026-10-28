#!/usr/bin/env python3
"""
Recreates the 3-panel figure (panels b, c, d) for Flexible Extended Domain Wall Dynamics:
  (b) Mean DW velocity <u_dot> vs h_0
  (c) Mean angular velocity <phi_dot> vs h_0
  (d) Roughness (rugosity) vs h_0 (clipped at 10^-6)
for extended wall sizes L = 8 and L = 1024, together with the rigid-wall result,
with white and gray shading for synchronized and unsynchronized rigid-wall solutions.
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PRB Style Settings
# ==============================================================================
def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.edgecolor': 'none',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

# ==============================================================================
# Numerical Integration for Rigid Wall ODE
# ==============================================================================
def compute_rigid_wall_sweep(h0_grid, alpha=0.27, omega=1.0, ha=3.0, dt=0.019634954):
    """
    Simulates the zero-dimensional rigid wall ODEs using 4th-order Runge-Kutta:
      du/dt   = 0.5 * (alpha^2 * h(t) + sin(2*phi))
      dphi/dt = 0.5 * (alpha * h(t) - alpha * sin(2*phi))
    over 200 transient cycles and 200 evaluation cycles of T_lock = 4*pi/omega.
    """
    T_lock = 4.0 * np.pi / omega
    steps_per_cycle = int(round(T_lock / dt))
    dt_act = T_lock / steps_per_cycle
    
    n_h0 = len(h0_grid)
    phi = np.zeros(n_h0)
    
    trans_steps = 200 * steps_per_cycle
    eval_steps = 200 * steps_per_cycle
    
    t = 0.0
    # Transient phase
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
        
    # Evaluation phase
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
    
    return vu_rigid, vphi_rigid

# ==============================================================================
# Helper to Load Sweep Output Directory
# ==============================================================================
def load_sweep_data(dir_path):
    """Loads all results_h0_*.txt from a directory and returns sorted arrays."""
    files = sorted(glob.glob(os.path.join(dir_path, "results_h0_*.txt")))
    if not files:
        # Check if files are directly in dir_path
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
                float(row['v_phi'])
            ))
        except Exception as e:
            print(f"Warning: Failed to parse {f}: {e}")
            continue
            
    if not records:
        return None
        
    records.sort(key=lambda r: r[0])  # sort by h0
    data = np.array(records)
    return {
        'h0': data[:, 0],
        'ha': data[:, 1],
        'rugosity': data[:, 2],
        'v_u': data[:, 3],
        'v_phi': data[:, 4]
    }

# ==============================================================================
# Main Plotting Function
# ==============================================================================
def plot_flexible_dynamics(master_dir, output_file=None):
    setup_prb_style()
    
    dir_l8 = os.path.join(master_dir, "L_8")
    dir_l1024 = os.path.join(master_dir, "L_1024")
    
    data_l8 = load_sweep_data(dir_l8) if os.path.isdir(dir_l8) else None
    data_l1024 = load_sweep_data(dir_l1024) if os.path.isdir(dir_l1024) else None
    
    if data_l8 is None and data_l1024 is None:
        # Fallback: check if master_dir itself contains results files
        data_direct = load_sweep_data(master_dir)
        if data_direct is not None:
            data_l1024 = data_direct
            print("Loaded sweep data directly from directory.")
        else:
            print(f"Error: Could not find results files in {master_dir}")
            return

    # Compute rigid-wall solutions and sync/unsync intervals on fine grid
    print("Computing rigid-wall reference solution...")
    h0_fine = np.linspace(0.0, 8.0, 801)
    vu_rigid, vphi_rigid = compute_rigid_wall_sweep(h0_fine)
    
    # Synchronization mask (|v_phi| < 1e-4)
    is_sync = np.abs(vphi_rigid) < 1e-4

    # Create 3-panel figure: (b), (c), (d)
    fig, (ax_b, ax_c, ax_d) = plt.subplots(3, 1, figsize=(4.8, 6.2), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    
    # Shading function for background (White = Synchronized, Gray = Unsynchronized)
    def add_background_shading(ax):
        # Scan fine grid intervals where is_sync is False
        in_unsync = False
        start_h = 0.0
        for i in range(len(h0_fine)):
            if not is_sync[i] and not in_unsync:
                in_unsync = True
                start_h = h0_fine[i]
            elif is_sync[i] and in_unsync:
                in_unsync = False
                ax.axvspan(start_h, h0_fine[i], color='0.90', zorder=0, lw=0)
        if in_unsync:
            ax.axvspan(start_h, h0_fine[-1], color='0.90', zorder=0, lw=0)

    # Apply background shading to all 3 axes
    for ax in [ax_b, ax_c, ax_d]:
        add_background_shading(ax)
        ax.set_xlim(0.0, 8.0)

    # --------------------------------------------------------------------------
    # Panel (b): Mean DW Velocity <u_dot>
    # --------------------------------------------------------------------------
    ax_b.plot(h0_fine, vu_rigid, color='black', linestyle='--', linewidth=1.1, label='Rigid wall', zorder=3)
    if data_l8 is not None:
        ax_b.plot(data_l8['h0'], data_l8['v_u'], color='tab:blue', marker='o', markersize=2.2,
                  linewidth=0.8, label=r'$L = 8$', zorder=4)
    if data_l1024 is not None:
        ax_b.plot(data_l1024['h0'], data_l1024['v_u'], color='tab:red', marker='s', markersize=2.2,
                  linewidth=0.8, label=r'$L = 1024$', zorder=5)
                  
    ax_b.set_ylabel(r'$\langle \dot{u} \rangle$')
    ax_b.legend(loc='upper left', ncol=3)
    ax_b.text(0.02, 0.85, '(b)', transform=ax_b.transAxes, fontweight='bold', fontsize=11)

    # --------------------------------------------------------------------------
    # Panel (c): Mean Angular Velocity <phi_dot>
    # --------------------------------------------------------------------------
    ax_c.plot(h0_fine, vphi_rigid, color='black', linestyle='--', linewidth=1.1, label='Rigid wall', zorder=3)
    if data_l8 is not None:
        ax_c.plot(data_l8['h0'], data_l8['v_phi'], color='tab:blue', marker='o', markersize=2.2,
                  linewidth=0.8, label=r'$L = 8$', zorder=4)
    if data_l1024 is not None:
        ax_c.plot(data_l1024['h0'], data_l1024['v_phi'], color='tab:red', marker='s', markersize=2.2,
                  linewidth=0.8, label=r'$L = 1024$', zorder=5)
                  
    ax_c.set_ylabel(r'$\langle \dot{\phi} \rangle$')
    ax_c.text(0.02, 0.85, '(c)', transform=ax_c.transAxes, fontweight='bold', fontsize=11)

    # --------------------------------------------------------------------------
    # Panel (d): Roughness (Values smaller than 10^-6 shown at 10^-6)
    # --------------------------------------------------------------------------
    if data_l8 is not None:
        rug_l8 = np.maximum(data_l8['rugosity'], 1e-6)
        ax_d.plot(data_l8['h0'], rug_l8, color='tab:blue', marker='o', markersize=2.2,
                  linewidth=0.8, label=r'$L = 8$', zorder=4)
    if data_l1024 is not None:
        rug_l1024 = np.maximum(data_l1024['rugosity'], 1e-6)
        ax_d.plot(data_l1024['h0'], rug_l1024, color='tab:red', marker='s', markersize=2.2,
                  linewidth=0.8, label=r'$L = 1024$', zorder=5)

    ax_d.set_yscale('log')
    ax_d.set_ylim(bottom=1e-6)
    ax_d.set_ylabel(r'Roughness $W^2$')
    ax_d.set_xlabel(r'DC driving field $h_0$')
    ax_d.text(0.02, 0.85, '(d)', transform=ax_d.transAxes, fontweight='bold', fontsize=11)

    # Save output
    if output_file is None:
        output_file = os.path.join(master_dir, "flexible_domain_wall_dynamics.pdf")
        
    png_file = os.path.splitext(output_file)[0] + ".png"
    fig.savefig(output_file)
    fig.savefig(png_file)
    plt.close(fig)
    print(f"Successfully generated figures:\n  - {output_file}\n  - {png_file}")

# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3-panel Flexible Extended Domain Wall Dynamics.")
    parser.add_argument("master_dir", help="Master sweep directory containing L_8 and L_1024 folders")
    parser.add_argument("--out", "-o", default=None, help="Output figure filename (.pdf)")
    args = parser.parse_args()
    
    plot_flexible_dynamics(args.master_dir, args.out)

