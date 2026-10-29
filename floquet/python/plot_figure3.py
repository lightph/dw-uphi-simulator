#!/usr/bin/env python3
r"""
plot_figure3.py - Reproduces PRB Figure 3:
Linear stability diagram of domain wall modes for \alpha = 0.27 and \Omega = 1.0.

Panels:
  (a) h_a = 2.0: Linear stability tongues across \kappa \in [0.7, 1.2] and h_0 \in [0, 8].
  (b) h_a = 3.0: Hybridized tongues and reentrant stable waist.

Conventions:
  - White regions: Stable flat synchronized DW (|\mu_\kappa| <= 1.0).
  - Gray regions: Unsynchronized rigid DW (no stable periodic cycle exists).
  - Colormap: Inferno heatmap of largest Floquet multiplier |\mu_\kappa| > 1.0 (range [1.0, 1.8]).
  - Right colorbar: Shared between panels, labeled |\mu_\kappa|.
  - Supports caching for fast subsequent plotting.
  - Outputs to figures/figure3.pdf and optionally exports to paper/figuras/figure3.pdf.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
import numpy as np

# Configure writable matplotlib cache
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from floquet_solver import compute_floquet_stability_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11.0,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
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
        'legend.fontsize': 9.0,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def get_cached_grid(ha, h0_vals, kappa_vals, cache_dir, alpha=0.27, omega=1.0, recompute=False):
    cache_file = Path(cache_dir) / f"floquet_grid_ha_{ha:.2f}.npz"
    if not recompute and cache_file.exists():
        try:
            cached = np.load(cache_file)
            if (np.allclose(cached['h0_vals'], h0_vals) and
                np.allclose(cached['kappa_vals'], kappa_vals) and
                cached['ha'] == ha):
                print(f"Loaded cached Floquet grid for h_a = {ha:.2f} from {cache_file}")
                return cached['mu_grid']
        except Exception:
            pass

    print(f"Computing Floquet stability grid for h_a = {ha:.2f} ({len(h0_vals)} x {len(kappa_vals)})...")
    mu_grid = compute_floquet_stability_grid(
        h0_vals, kappa_vals, ha, alpha=alpha, omega=omega, steps_per_cycle=400, verbose=True
    )
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, h0_vals=h0_vals, kappa_vals=kappa_vals,
                            ha=ha, alpha=alpha, omega=omega, mu_grid=mu_grid)
    except Exception:
        pass

    return mu_grid


def plot_figure3(out_dirs=None, recompute=False, n_h0=300, n_kappa=200):
    setup_prb_style()
    cache_dir = REPO_ROOT / "output" / ".cache"

    if out_dirs is None:
        out_dirs = [REPO_ROOT / "figures", REPO_ROOT / "paper" / "figuras"]

    # Parameter ranges matching publication figure 3
    h0_vals = np.linspace(0.0, 8.0, n_h0)
    kappa_vals = np.linspace(0.7, 1.2, n_kappa)

    # 1. Obtain Floquet stability matrices for ha = 2.0 and ha = 3.0
    mu_grid_ha2 = get_cached_grid(2.0, h0_vals, kappa_vals, cache_dir, recompute=recompute)
    mu_grid_ha3 = get_cached_grid(3.0, h0_vals, kappa_vals, cache_dir, recompute=recompute)

    # 2. Build PRB 2-panel figure
    fig = plt.figure(figsize=(3.375, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.05], hspace=0.08, wspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[:, 1])

    # Colormap and Normalization
    cmap = cm.get_cmap('inferno').copy()
    norm = Normalize(vmin=1.0, vmax=1.8)

    extent = [h0_vals[0], h0_vals[-1], kappa_vals[0], kappa_vals[-1]]

    for ax, mu_grid, label in [(ax_top, mu_grid_ha2, "(a) $h_\\mathrm{a}=2$"),
                               (ax_bot, mu_grid_ha3, "(b) $h_\\mathrm{a}=3$")]:
        ax.set_facecolor('#bcbcbc')  # Gray background for unsynchronized

        # Create RGBA image:
        # - nan -> transparent (shows gray background)
        # - mu <= 1.0 -> white
        # - mu > 1.0 -> inferno colormap
        rgba_img = np.zeros((len(kappa_vals), len(h0_vals), 4), dtype=np.float32)

        # Synchronized mask
        is_sync = ~np.isnan(mu_grid)
        # Stable mask
        is_stable = is_sync & (mu_grid <= 1.0001)
        # Unstable mask
        is_unstable = is_sync & (mu_grid > 1.0001)

        # White for stable
        rgba_img[is_stable] = [1.0, 1.0, 1.0, 1.0]

        # Inferno for unstable
        if np.any(is_unstable):
            colors = cmap(norm(mu_grid[is_unstable]))
            rgba_img[is_unstable] = colors

        # nan remains (0, 0, 0, 0) -> shows ax facecolor
        rgba_img[~is_sync] = [0.74, 0.74, 0.74, 1.0]

        ax.imshow(rgba_img, origin='lower', extent=extent, aspect='auto', interpolation='bilinear')
        ax.set_xlim(0.0, 8.0)
        ax.set_ylim(0.7, 1.2)
        ax.set_ylabel(r'$\kappa$')

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    ax_top.tick_params(labelbottom=False)
    ax_bot.set_xlabel(r'$h_0$')

    # Colorbar
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(r'$|\mu_\kappa|$', fontsize=11.0)
    cb.set_ticks([1.0, 1.2, 1.4, 1.6, 1.8])
    cax.tick_params(labelsize=9.0, direction='in')

    # Save to requested directories
    saved_paths = []
    for out_d in out_dirs:
        out_p = Path(out_d)
        out_p.mkdir(parents=True, exist_ok=True)
        target = out_p / "figure3.pdf"
        fig.savefig(target)
        saved_paths.append(target)
        fig.savefig(out_p / "figure3.png")
        print(f"Saved: {target}")

    plt.close(fig)
    return saved_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate PRB Figure 3 (Floquet Stability Diagrams).")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation without cache")
    parser.add_argument("--h0-pts", type=int, default=240, help="Number of points in h0 grid")
    parser.add_argument("--kappa-pts", type=int, default=150, help="Number of points in kappa grid")
    args = parser.parse_args()

    dirs = [Path(args.out_dir)]
    if args.export_paper:
        dirs.append(REPO_ROOT / "paper" / "figuras")

    plot_figure3(out_dirs=dirs, recompute=args.recompute, n_h0=args.h0_pts, n_kappa=args.kappa_pts)

