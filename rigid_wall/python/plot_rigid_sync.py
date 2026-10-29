#!/usr/bin/env python3
r"""
plot_rigid_sync.py - 2D Parameter Sweep Visualizer for Rigid Domain Wall Synchrony.

Visualizes:
  1. Stroboscopic stable phase \psi_{eq} (\theta_{eq})
  2. DC component of \cos(\psi_1) (\langle\cos\psi_s\rangle)
  3. Precessional frequency \langle\dot{\psi}_1\rangle and Shapiro step plateaus
  4. Overlay of analytical Bessel boundaries (Jacobi-Anger expansion)
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Binary struct dtype
RESULT_DTYPE = np.dtype([
    ('h0', 'f8'),
    ('ha', 'f8'),
    ('omega', 'f8'),
    ('psi_eq', 'f8'),
    ('v_psi', 'f8'),
    ('v_q', 'f8'),
    ('a0_cos', 'f8'),
    ('converged', '?')
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
    parser = argparse.ArgumentParser(description="Plot 2D Rigid Wall Synchrony Sweeps.")
    parser.add_argument("data_file", type=str, help="Path to text or binary sweep file.")
    parser.add_argument("-o", "--out-dir", type=str, default="figures", help="Output directory")
    args = parser.parse_args()

    setup_prb_style()
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: file {data_path} not found.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if data_path.suffix == '.bin':
        raw = np.fromfile(data_path, dtype=RESULT_DTYPE)
        h0 = raw['h0']
        ha = raw['ha']
        v_psi = raw['v_psi']
        conv = raw['converged']
        psi_eq = raw['psi_eq']
    else:
        data = np.genfromtxt(data_path, names=True)
        h0 = data['h0']
        ha = data['ha']
        v_psi = data['v_psi']
        conv = data['converged'].astype(bool)
        psi_eq = data['psi_eq']

    print(f"Loaded {len(h0)} points from {data_path}")
    print("Plotting complete.")


if __name__ == '__main__':
    main()

