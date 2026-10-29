#!/usr/bin/env python3
r"""
plot_floquet_map.py - 2D Floquet Multiplier & Instability Map Visualizer

Plots:
  1. Maximum Floquet multiplier \max_\kappa |\mu_\kappa| across (h_0, h_a)
  2. Critical wavenumber \kappa_{\max} corresponding to maximum growth
  3. Overlay of analytical Bessel boundaries for Shapiro steps
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
import numpy as np

if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

RESULT_DTYPE = np.dtype([
    ('h', '<f8'), ('h_ac', '<f8'), ('omega', '<f8'), ('x', '<f8'),
    ('max_k_mu', '<f8'), ('max_mu', '<f8'), ('max_k_sigma', '<f8'),
    ('max_sigma', '<f8'), ('converged', '<?')
], align=True)


def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11.0,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9.0,
        'ytick.labelsize': 9.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.fontsize': 9.0,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def main():
    parser = argparse.ArgumentParser(description="Plot 2D Floquet Stability Map.")
    parser.add_argument("bin_file", type=str, help="Path to binary Floquet output file.")
    parser.add_argument("-o", "--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--num-h", type=int, default=1000, help="Grid points in h0")
    parser.add_argument("--num-hac", type=int, default=1000, help="Grid points in ha")
    parser.add_argument("--h-max", type=float, default=10.0, help="Maximum h0")
    parser.add_argument("--hac-max", type=float, default=20.0, help="Maximum ha")
    args = parser.parse_args()

    setup_prb_style()
    bin_path = Path(args.bin_file)
    if not bin_path.exists():
        print(f"Error: file {bin_path} not found.")
        sys.exit(1)

    raw = np.fromfile(bin_path, dtype=RESULT_DTYPE)
    expected = args.num_h * args.num_hac
    if len(raw) != expected:
        # Try inferring square grid
        side = int(np.round(np.sqrt(len(raw))))
        if side * side == len(raw):
            num_h = side
            num_hac = side
        else:
            num_h = args.num_h
            num_hac = len(raw) // num_h
    else:
        num_h = args.num_h
        num_hac = args.num_hac

    conv_grid = raw['converged'].reshape((num_h, num_hac))
    mu_grid = raw['max_mu'].reshape((num_h, num_hac))
    k_grid = raw['max_k_mu'].reshape((num_h, num_hac))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extent = [0.0, args.h_max, 0.0, args.hac_max]

    # Plot 1: Maximum Floquet multiplier
    fig, ax = plt.subplots(figsize=(3.375, 3.2))
    ax.set_facecolor('#dedede')

    masked_mu = np.where(conv_grid, mu_grid, np.nan)
    norm = Normalize(vmin=1.0, vmax=2.0)
    im = ax.imshow(masked_mu.T, origin='lower', extent=extent, aspect='auto',
                   cmap='magma', norm=norm, interpolation='nearest')

    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.set_label(r'$\max_\kappa |\mu_\kappa|$')

    ax.set_xlabel(r'$h_0$')
    ax.set_ylabel(r'$h_\mathrm{a}$')
    fig.savefig(out_dir / "floquet_multiplier_map.pdf")
    fig.savefig(out_dir / "floquet_multiplier_map.png")
    plt.close(fig)

    print(f"Saved Floquet multiplier map to {out_dir / 'floquet_multiplier_map.pdf'}")


if __name__ == '__main__':
    main()

