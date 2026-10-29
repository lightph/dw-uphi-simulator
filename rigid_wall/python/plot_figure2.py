#!/usr/bin/env python3
r"""
plot_figure2.py - Reproduces PRB Figure 2:
Steady-state average velocities in rigid domain walls and Shapiro synchronization steps.

Features:
  - Top panel: Mean internal precessional velocity <\dot{\psi}_1>_\tau vs h_0
  - Bottom panel: Mean translational velocity <\dot{q}_1>_\tau vs h_0
  - Compares h_a = 0 (dashed black) and h_a = 5 (solid blue)
  - White regions: Shapiro synchronization steps (<d\psi_1/d\tau> = 0, 1, 2)
  - Gray regions: Unsynchronized regimes
  - Vertical red dotted lines: 1st-order perturbation theory boundaries
  - Vertical green dashed lines: 2nd-order perturbation theory boundaries
  - Saves to figures/figure2.pdf and optionally exports to paper/figuras/figure2.pdf
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
import numpy as np

# Configure writable matplotlib cache directory
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

# Setup import path for rigid_wall module
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from rigid_wall import adler_velocities, perturbation_boundaries, solve_rigid_wall_rk4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11.5,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 10.0,
        'ytick.labelsize': 10.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'legend.fontsize': 9.5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'none',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_figure2(out_dirs=None, alpha=0.27, omega=1.0, ha=5.0, num_h0=600):
    setup_prb_style()

    if out_dirs is None:
        out_dirs = [REPO_ROOT / "figures", REPO_ROOT / "paper" / "figuras"]

    h0_grid = np.linspace(0.0, 10.0, num_h0)

    # 1. Exact analytical velocities for ha = 0
    v_psi_ha0, v_q_ha0 = adler_velocities(h0_grid, alpha=alpha, omega=omega)

    # 2. Numerical RK4 integration for ha = 5
    print("Integrating 0D rigid wall ODEs for h_a = 5 across h_0 in [0, 10]...")
    _, v_psi_ha5, v_q_ha5, is_sync = solve_rigid_wall_rk4(
        h0_grid, ha=ha, alpha=alpha, omega=omega, trans_cycles=250, eval_cycles=250, steps_per_cycle=1000
    )

    # 3. Analytical perturbation theory boundaries
    pert = perturbation_boundaries(n_steps=3, alpha=alpha, omega=omega, ha=ha)

    # 4. Construct two-panel figure matching paper/main.tex fig:ej_walker
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.375, 3.7), sharex=True,
                                         gridspec_kw={'hspace': 0.06})

    # Shading: find contiguous synchronized intervals where is_sync is True
    # Background is gray; synchronized intervals are shaded white
    ax_top.set_facecolor('#dedede')
    ax_bot.set_facecolor('#dedede')

    # Find contiguous sync segments
    sync_diff = np.diff(is_sync.astype(int))
    starts = np.where(sync_diff == 1)[0] + 1
    ends = np.where(sync_diff == -1)[0] + 1
    if is_sync[0]:
        starts = np.insert(starts, 0, 0)
    if is_sync[-1]:
        ends = np.append(ends, len(is_sync) - 1)

    for s, e in zip(starts, ends):
        h_left = h0_grid[s]
        h_right = h0_grid[e if e < len(h0_grid) else len(h0_grid) - 1]
        ax_top.axvspan(h_left, h_right, color='#ffffff', zorder=1)
        ax_bot.axvspan(h_left, h_right, color='#ffffff', zorder=1)

    # Top panel: <d\psi_1/d\tau> vs h0
    ax_top.axhline(0.0, color='gray', linestyle=':', lw=0.8, alpha=0.7, zorder=2)
    ax_top.axhline(1.0, color='gray', linestyle=':', lw=0.8, alpha=0.7, zorder=2)
    ax_top.axhline(2.0, color='gray', linestyle=':', lw=0.8, alpha=0.7, zorder=2)

    ax_top.plot(h0_grid, v_psi_ha0, color='black', linestyle='--', lw=1.3,
                label=r'$h_a = 0$', zorder=4)
    ax_top.plot(h0_grid, v_psi_ha5, color='#1f77b4', linestyle='-', lw=1.4,
                label=r'$h_a = 5$', zorder=5)

    # Bottom panel: <dq_1/d\tau> vs h0
    ax_bot.plot(h0_grid, v_q_ha0, color='black', linestyle='--', lw=1.3, zorder=4)
    ax_bot.plot(h0_grid, v_q_ha5, color='#1f77b4', linestyle='-', lw=1.4, zorder=5)

    # Draw vertical perturbation boundaries (1st order red dotted, 2nd order green dashed)
    for n in range(3):
        # 1st order (red dotted)
        _, b1_low, b1_high = pert['first_order'][n]
        if n > 0:  # n=0 left bound is negative
            for ax in (ax_top, ax_bot):
                ax.axvline(b1_low, color='#d62728', linestyle=':', lw=1.1, zorder=6)
        for ax in (ax_top, ax_bot):
            ax.axvline(b1_high, color='#d62728', linestyle=':', lw=1.1, zorder=6)

        # 2nd order (green dashed)
        _, b2_low, b2_high, _ = pert['second_order'][n]
        if n > 0:
            for ax in (ax_top, ax_bot):
                ax.axvline(b2_low, color='#2ca02c', linestyle='--', lw=1.1, zorder=6)
        for ax in (ax_top, ax_bot):
            ax.axvline(b2_high, color='#2ca02c', linestyle='--', lw=1.1, zorder=6)

    # Labels and limits
    ax_top.set_xlim(0.0, 10.0)
    ax_bot.set_xlim(0.0, 10.0)

    ax_top.set_ylim(-0.2, 2.7)
    ax_top.set_yticks([0, 1, 2])
    ax_top.set_ylabel(r'$\langle \dot{\psi}_1 \rangle_\tau$')

    ax_bot.set_ylim(-0.15, 0.55)
    ax_bot.set_yticks([0.0, 0.2, 0.4])
    ax_bot.set_ylabel(r'$\langle \dot{q}_1 \rangle_\tau$')
    ax_bot.set_xlabel(r'$h_0$')

    ax_top.legend(loc='upper left', framealpha=0.9, borderaxespad=0.6)

    for ax in (ax_top, ax_bot):
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Save to all requested directories
    saved_paths = []
    for out_d in out_dirs:
        out_p = Path(out_d)
        out_p.mkdir(parents=True, exist_ok=True)
        target = out_p / "figure2.pdf"
        fig.savefig(target)
        saved_paths.append(target)
        # Also save PNG for easy inspection
        fig.savefig(out_p / "figure2.png")
        print(f"Saved: {target}")

    plt.close(fig)
    return saved_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PRB Figure 2 (Rigid DW Dynamics).")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    parser.add_argument("--h0-pts", type=int, default=600, help="Number of points in h0 grid")
    args = parser.parse_args()

    dirs = [Path(args.out_dir)]
    if args.export_paper:
        dirs.append(REPO_ROOT / "paper" / "figuras")

    plot_figure2(out_dirs=dirs, num_h0=args.h0_pts)
