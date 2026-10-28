#!/usr/bin/env python3
r"""
plot_prb_publication.py - Physical Review B publication-ready plotter.

Generates individual publication-quality figures for:
  1. Rugosity: <w^2>_t (logarithmic interface roughness)
  2. Stroboscopic variance: <\sigma^2>_x (temporal synchronization variance)
  3. 5-State Phase diagram (P, SF, SP, NF, NR) as solid contiguous blocks with top legend

Includes high-contrast Floquet stability and synchrony boundary overlays from binary data.
All PDF outputs use rasterized heatmaps for compact file sizes while keeping all text,
ticks, legends, and boundary contours as pure vectors.
"""

import os
import sys
import glob
import time
import argparse
import tempfile
from pathlib import Path

# Configure writable matplotlib cache to prevent warnings
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe

# ==============================================================================
# PRB PUBLICATION-GRADE TYPOGRAPHY & STYLE
# ==============================================================================
def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 13,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5.0,
        'ytick.major.size': 5.0,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.7,
        'ytick.minor.width': 0.7,
        'legend.fontsize': 10.5,
        'legend.frameon': True,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

# ==============================================================================
# DATA LOADERS
# ==============================================================================
def load_sweep_dataset(data_dir, quiet=False):
    """Loads simulation sweep data with automatic cache checking."""
    cache_file = os.path.join(data_dir, ".sweep_cache.npz")
    if os.path.exists(cache_file):
        if not quiet:
            print(f"Loading cached sweep data: {cache_file}")
        with np.load(cache_file) as npz:
            data = {k: npz[k] for k in npz.files}
        return data

    files = sorted(glob.glob(os.path.join(data_dir, "results_h0_*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        files = [f for f in files if "parameter" not in os.path.basename(f).lower()]

    if not files:
        raise FileNotFoundError(f"No results files found in {data_dir}")

    if not quiet:
        print(f"Reading {len(files)} sweep files from {data_dir} ...")

    with open(files[0], 'r') as fp:
        cols = fp.readline().strip().split()

    arrays = []
    for f in files:
        if os.path.getsize(f) > 0:
            arr = np.loadtxt(f, skiprows=1)
            if arr.ndim == 1 and arr.size > 0:
                arr = arr.reshape(1, -1)
            if arr.size > 0:
                arrays.append(arr)

    all_data = np.vstack(arrays)
    data = {col: all_data[:, i] for i, col in enumerate(cols)}
    data['h0'] = np.round(data['h0'], 5)
    data['ha'] = np.round(data['ha'], 5)

    try:
        np.savez_compressed(cache_file, **data)
        if not quiet:
            print(f"Created cache: {cache_file}")
    except Exception:
        pass

    return data

def load_floquet_boundaries(bin_path, num_h=1000, num_hac=1000, h_max=10.0, hac_max=20.0, quiet=False):
    """Loads Floquet binary stability and synchrony data."""
    if not bin_path or not os.path.exists(bin_path):
        if not quiet:
            print(f"Note: Floquet bin file '{bin_path}' not found. Skipping overlay.")
        return False, None, None, None, None

    dtype = np.dtype([
        ('h', '<f8'), ('h_ac', '<f8'), ('omega', '<f8'), ('x', '<f8'),
        ('max_k_mu', '<f8'), ('max_mu', '<f8'), ('max_k_sigma', '<f8'),
        ('max_sigma', '<f8'), ('converged', '<?')
    ], align=True)

    try:
        raw_data = np.fromfile(bin_path, dtype=dtype)
        if len(raw_data) != num_h * num_hac:
            if not quiet:
                print(f"Warning: Unexpected bin size ({len(raw_data)}). Skipping.")
            return False, None, None, None, None

        converged_grid = raw_data['converged'].reshape((num_h, num_hac))
        mu_grid = raw_data['max_mu'].reshape((num_h, num_hac))
        mu_grid = np.where(converged_grid, mu_grid, np.nan)

        h_vals = np.linspace(0, h_max, num_h)
        hac_vals = np.linspace(0, hac_max, num_hac)
        H_GRID, HAC_GRID = np.meshgrid(h_vals, hac_vals, indexing='ij')

        if not quiet:
            print(f"Loaded Floquet boundary data from {bin_path}")
        return True, H_GRID, HAC_GRID, converged_grid, mu_grid
    except Exception as e:
        if not quiet:
            print(f"Warning: Failed to parse Floquet binary: {e}")
        return False, None, None, None, None

# ==============================================================================
# CONTOUR OVERLAY HELPER
# ==============================================================================
def add_floquet_contours(ax, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=0.65):
    """
    Draws thin, solid boundary contours without outlines and without transparency.
    - Synchrony boundary: clean thin solid white line
    - Stability boundary (mu = 1): clean thin dashed black line
    """
    # Synchrony boundary: clean thin solid white line (no outlines, full opacity)
    ax.contour(H_GRID, HAC_GRID, conv_grid, levels=[0.5], colors=['#ffffff'],
               linestyles=['solid'], linewidths=[lw])

    # Stability boundary (mu = 1): clean thin dashed black line (no outlines, full opacity)
    ax.contour(H_GRID, HAC_GRID, mu_grid, levels=[1.0001], colors=['#000000'],
               linestyles=['dashed'], linewidths=[lw])

# ==============================================================================
# ANALYZED POINTS OVERLAY
# ==============================================================================
ANALYZED_POINTS = [
    {'h0': 3.70, 'ha': 3.00, 'label': '1'},
    {'h0': 3.50, 'ha': 3.00, 'label': '2'},
    {'h0': 3.80, 'ha': 2.10, 'label': '3'},
    {'h0': 3.15, 'ha': 7.30, 'label': '4'},
    {'h0': 3.45, 'ha': 3.675, 'label': '5'}
]

def add_analyzed_markers(ax, points=ANALYZED_POINTS, markersize=8, mew=1.8):
    """Marks analyzed parameter cases with high-contrast, halo-outlined 'x' markers."""
    for p in points:
        ax.plot(p['h0'], p['ha'], marker='x', markersize=markersize, mew=mew, color='#ffffff',
                path_effects=[pe.Stroke(linewidth=mew + 1.8, foreground='#000000'), pe.Normal()],
                zorder=25)

# ==============================================================================
# 2D DESPANGLING / MAJORITY FILTER (Solid blocks of phases)
# ==============================================================================
def clean_grainy_grid(grid, iterations=2):
    """
    Applies a discrete 3x3 majority filter to remove isolated single-pixel
    numerical noise grains, producing solid, smooth, contiguous phase blocks.
    """
    cleaned = grid.copy()
    h, w = grid.shape
    for _ in range(iterations):
        padded = np.pad(cleaned, 1, mode='edge')
        neighborhood = np.stack([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
            padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
        ], axis=0)
        counts = np.zeros((5, h, w), dtype=int)
        for cat in range(5):
            counts[cat] = np.sum(neighborhood == cat, axis=0)
        mode_cat = np.argmax(counts, axis=0)
        max_count = np.max(counts, axis=0)
        # Replace if central pixel is isolated from majority (>= 5 out of 9 neighbors)
        replace_mask = (max_count >= 5) & (cleaned != mode_cat)
        cleaned[replace_mask] = mode_cat[replace_mask]
    return cleaned

# ==============================================================================
# DC CRITICAL FIELD AUTO-DETECTION (Fabri's Suggestion)
# ==============================================================================
def detect_hc(clean_phase_grid, u_h0, default_hc=3.825):
    """
    Detects critical field hc under purely DC drive (ha = 0).
    Under pure DC:
      - 0 <= h0 <= 1.0: P (non-precessional)
      - 1.0 < h0 < hc: NR (precessional rough)
      - h0 >= hc: NF (non-synchronized flat)
    hc is the onset where the wall permanently becomes flat (NF, phase 3) for all h0 >= hc.
    """
    row0 = clean_phase_grid[0, :]
    nr_indices = np.where((row0 == 4) | (u_h0 <= 1.0))[0]
    if len(nr_indices) > 0 and nr_indices[-1] < len(u_h0) - 1:
        return float(u_h0[nr_indices[-1] + 1])
    return default_hc

# ==============================================================================
# PHASE DIAGRAM FIGURE BUILDER WITH DC STRIP (Fabri's Suggestion)
# ==============================================================================
def create_phase_diagram_figure(X, Y, clean_phase_grid, phase_cmap, phase_patches,
                                has_floquet=False, H_GRID=None, HAC_GRID=None,
                                conv_grid=None, mu_grid=None, line_width=0.65,
                                hc=3.825, with_dc_strip=True, analyzed_points=None):
    """
    Constructs the PRB publication phase diagram figure.
    If with_dc_strip is True (Fabri's suggestion):
      - Top axes: 2D (h0, ha) phase diagram with Floquet contours and optional markers.
      - Bottom axes: 1D horizontal strip showing purely DC phases (ha = 0):
          * P (non-precessional) for 0 <= h0 <= 1.0
          * NR (precessional rough) for 1.0 < h0 < hc
          * NF (precessional flat) for hc <= h0 <= 10.0
    """
    if with_dc_strip:
        fig = plt.figure(figsize=(4.8, 4.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.08], hspace=0.08,
                               top=0.91, bottom=0.11, left=0.14, right=0.96)
        ax_main = fig.add_subplot(gs[0])
        ax_dc = fig.add_subplot(gs[1], sharex=ax_main)
    else:
        fig, ax_main = plt.subplots(figsize=(4.8, 3.8))
        ax_dc = None

    # 1. Main 2D phase diagram
    ax_main.pcolormesh(X, Y, clean_phase_grid, cmap=phase_cmap, vmin=-0.5, vmax=4.5,
                       shading='nearest', rasterized=True)
    if has_floquet:
        add_floquet_contours(ax_main, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=line_width)
    if analyzed_points is not None:
        add_analyzed_markers(ax_main, analyzed_points)

    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, 20)
    ax_main.set_ylabel(r'$h_\mathrm{a}$', fontsize=12)
    ax_main.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_main.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Legend above the main plot
    ax_main.legend(handles=phase_patches, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                   ncol=5, frameon=True, framealpha=1.0, edgecolor='none',
                   handlelength=1.2, handletextpad=0.4, columnspacing=1.0, fontsize=10.0)

    # 2. Bottom DC strip (Fabri's suggestion)
    if with_dc_strip and ax_dc is not None:
        ax_main.tick_params(labelbottom=False, bottom=True, top=True, left=True, right=True)

        ax_dc.set_xlim(0, 10)
        ax_dc.set_ylim(0, 1)
        # Colored intervals matching the 5-phase palette
        ax_dc.axvspan(0.0, 1.0, facecolor=phase_cmap.colors[0], edgecolor='none')
        ax_dc.axvspan(1.0, hc, facecolor=phase_cmap.colors[4], edgecolor='none')
        ax_dc.axvspan(hc, 10.0, facecolor=phase_cmap.colors[3], edgecolor='none')

        # Phase boundary dividers
        ax_dc.axvline(1.0, color='white', lw=0.9)
        ax_dc.axvline(hc, color='white', lw=0.9)

        # Region text labels inside strip
        ax_dc.text(0.5, 0.5, r'$\mathrm{P}$', color='white', fontsize=9, fontweight='bold', ha='center', va='center')
        ax_dc.text((1.0 + hc) / 2.0, 0.5, r'$\mathrm{NR}$', color='white', fontsize=9, fontweight='bold', ha='center', va='center')
        ax_dc.text((hc + 10.0) / 2.0, 0.5, r'$\mathrm{NF}$', color='black', fontsize=9, fontweight='bold', ha='center', va='center')

        ax_dc.set_ylabel(r'$\mathrm{DC}$', fontsize=10.5, rotation=0, ha='right', va='center')
        ax_dc.set_yticks([])
        ax_dc.set_xlabel(r'$h_0$', fontsize=12)
        ax_dc.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_dc.tick_params(bottom=True, top=False, left=True, right=True)
    else:
        ax_main.set_xlabel(r'$h_0$', fontsize=12)
        plt.tight_layout()

    return fig

# ==============================================================================
# MAIN PUBLICATION GENERATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Physical Review B publication-ready plots for sweep data."
    )
    parser.add_argument("--data-dir", default="output/sweep_full_973971",
                        help="Directory containing sweep results files")
    parser.add_argument("--floquet-bin", default="output/floquet_transient_omega_1.000_L_1024.bin",
                        help="Path to Floquet stability binary file")
    parser.add_argument("-o", "--out-dir", default=None,
                        help="Directory to save figures (defaults to <data-dir>/figures)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster PNG output (default: 300)")
    parser.add_argument("--line-width", "--lw", type=float, default=0.65,
                        help="Line width for boundary contours (default: 0.65)")
    parser.add_argument("--hc", type=float, default=None,
                        help="Critical field hc separating NR from NF in DC strip (default: auto-detect from ha=0 data)")
    parser.add_argument("--no-floquet", action="store_true",
                        help="Disable Floquet boundary contours overlay")
    parser.add_argument("--no-dc-strip", action="store_true",
                        help="Disable 1D DC phase strip below the phase diagram")
    parser.add_argument("--export-paper", action="store_true", default=True,
                        help="Also export phase diagram figure to paper/figuras/")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Quiet execution mode")
    args = parser.parse_args()

    setup_prb_style()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "figures")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Simulation Data
    data = load_sweep_dataset(args.data_dir, quiet=args.quiet)

    # 2. Load Floquet Boundary Data
    has_floquet = False
    H_GRID, HAC_GRID, conv_grid, mu_grid = None, None, None, None
    if not args.no_floquet:
        has_floquet, H_GRID, HAC_GRID, conv_grid, mu_grid = load_floquet_boundaries(
            args.floquet_bin, quiet=args.quiet
        )

    # 3. Grid coordinates and pivot preparation
    u_h0, h0_idx = np.unique(data['h0'], return_inverse=True)
    u_ha, ha_idx = np.unique(data['ha'], return_inverse=True)
    X, Y = np.meshgrid(u_h0, u_ha)
    n_ha, n_h0 = len(u_ha), len(u_h0)
    flat_idx = ha_idx * n_h0 + h0_idx

    def get_grid(arr):
        arr_valid = np.nan_to_num(arr, nan=0.0)
        valid_mask = np.isfinite(arr)
        sums = np.bincount(flat_idx[valid_mask], weights=arr_valid[valid_mask], minlength=n_ha * n_h0)
        counts = np.bincount(flat_idx[valid_mask], minlength=n_ha * n_h0)
        with np.errstate(invalid='ignore', divide='ignore'):
            grid_vals = np.where(counts > 0, sums / counts, np.nan)
        return grid_vals.reshape((n_ha, n_h0))

    # 4. Classify Phase Diagram with optimal physical thresholds for solid blocks
    # - t_lock = 3.5e-12 captures the full synchronized Arnold tongue across the 8-order gap
    # - t_rug = 1e-4 separates flat walls (< 1e-6) from macroscopic corrugation/roughness (> 1e-3)
    # - t_vphi = 1e-5 separates pinned states (v_phi ~ 0) from steadily rotating states (v_phi >= 0.25)
    t_lock = 3.5e-12
    t_rug = 1e-4
    t_vphi = 1e-5

    is_locked = data['mean_var_strobo'] < t_lock
    is_flat = data['rugosity'] < t_rug
    is_pinned = is_locked & (np.abs(data['v_phi']) < t_vphi)
    conditions = [
        is_pinned,
        is_locked & is_flat & ~is_pinned,
        is_locked & ~is_flat,
        ~is_locked & is_flat,
        ~is_locked & ~is_flat
    ]
    raw_phase_diagram = np.select(conditions, [0, 1, 2, 3, 4], default=-1)

    raw_phase_grid = get_grid(raw_phase_diagram).astype(int)
    # Apply majority filter to eliminate isolated numerical noise grains into solid blocks
    clean_phase_grid = clean_grainy_grid(raw_phase_grid, iterations=2)

    # Determine critical field hc for purely DC driving (Fabri's suggestion)
    hc_val = args.hc if args.hc is not None else detect_hc(clean_phase_grid, u_h0)
    with_dc = not args.no_dc_strip

    if not args.quiet:
        print("\nThresholds applied:")
        print(f"  - Stroboscopic Locking THRESH_LOCK : {t_lock:.1e}")
        print(f"  - Rugosity Flatness   THRESH_RUG  : {t_rug:.1e}")
        print(f"  - Angular Pinning     THRESH_VPHI : {t_vphi:.1e}")
        if with_dc:
            print(f"  - DC Strip Critical Field hc   : {hc_val:.3f}\n")
        else:
            print()

    # Clean up obsolete 3-panel files if present
    for old_file in ["prb_sweep_3panel.png", "prb_sweep_3panel.pdf"]:
        old_path = os.path.join(args.out_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # =========================================================================
    # FIGURE 1: RUGOSITY <w^2>_t
    # =========================================================================
    rug_grid = get_grid(data['rugosity'])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    mesh = ax.pcolormesh(X, Y, np.clip(rug_grid, 1e-15, None), cmap='inferno',
                         norm=LogNorm(vmin=1e-15, vmax=1.0), shading='nearest', rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(r'$\langle w^2 \rangle_t$', fontsize=13)
    cbar.set_ticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0])
    cbar.ax.tick_params(direction='in', which='both')

    if has_floquet:
        add_floquet_contours(ax, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)

    ax.set_xlabel(r'$h_0$', fontsize=13)
    ax.set_ylabel(r'$h_\mathrm{a}$', fontsize=13)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    f1_png = os.path.join(args.out_dir, "prb_rugosity.png")
    f1_pdf = os.path.join(args.out_dir, "prb_rugosity.pdf")
    fig.savefig(f1_png, dpi=args.dpi)
    fig.savefig(f1_pdf)
    plt.close(fig)
    if not args.quiet:
        print(f"Generated: {f1_png} and {f1_pdf}")

    # =========================================================================
    # FIGURE 2: STROBOSCOPIC VARIANCE <\sigma^2>_x
    # =========================================================================
    strobo_grid = get_grid(data['mean_var_strobo'])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    mesh = ax.pcolormesh(X, Y, np.clip(strobo_grid, 1e-15, None), cmap='magma',
                         norm=LogNorm(vmin=1e-15, vmax=10.0), shading='nearest', rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(r'$\langle \sigma^2 \rangle_x$', fontsize=13)
    cbar.set_ticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0])
    cbar.ax.tick_params(direction='in', which='both')

    if has_floquet:
        add_floquet_contours(ax, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)

    ax.set_xlabel(r'$h_0$', fontsize=13)
    ax.set_ylabel(r'$h_\mathrm{a}$', fontsize=13)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    f2_png = os.path.join(args.out_dir, "prb_strobo_variance.png")
    f2_pdf = os.path.join(args.out_dir, "prb_strobo_variance.pdf")
    fig.savefig(f2_png, dpi=args.dpi)
    fig.savefig(f2_pdf)
    plt.close(fig)
    if not args.quiet:
        print(f"Generated: {f2_png} and {f2_pdf}")

    # =========================================================================
    # FIGURE 3: SOLID 5-STATE PHASE DIAGRAM (Clean PRB with Fabri's DC strip)
    # =========================================================================
    phase_cmap = ListedColormap(['#1b7837', '#2166ac', '#4393c3', '#f4a582', '#b2182b'])
    phase_labels = ['P', 'SF', 'SP', 'NF', 'NR']
    phase_patches = [mpatches.Patch(color=phase_cmap.colors[i], label=phase_labels[i]) for i in range(5)]

    fig3 = create_phase_diagram_figure(
        X, Y, clean_phase_grid, phase_cmap, phase_patches,
        has_floquet=has_floquet, H_GRID=H_GRID, HAC_GRID=HAC_GRID,
        conv_grid=conv_grid, mu_grid=mu_grid, line_width=args.line_width,
        hc=hc_val, with_dc_strip=with_dc, analyzed_points=None
    )
    f3_png = os.path.join(args.out_dir, "prb_phase_diagram.png")
    f3_pdf = os.path.join(args.out_dir, "prb_phase_diagram.pdf")
    fig3.savefig(f3_png, dpi=args.dpi)
    fig3.savefig(f3_pdf)
    plt.close(fig3)
    if not args.quiet:
        print(f"Generated: {f3_png} and {f3_pdf}")

    # =========================================================================
    # FIGURE 4: PHASE DIAGRAM WITH ANALYZED POINTS (MARKED WITH 'X')
    # =========================================================================
    fig4 = create_phase_diagram_figure(
        X, Y, clean_phase_grid, phase_cmap, phase_patches,
        has_floquet=has_floquet, H_GRID=H_GRID, HAC_GRID=HAC_GRID,
        conv_grid=conv_grid, mu_grid=mu_grid, line_width=args.line_width,
        hc=hc_val, with_dc_strip=with_dc, analyzed_points=ANALYZED_POINTS
    )
    f4_png = os.path.join(args.out_dir, "prb_phase_diagram_points.png")
    f4_pdf = os.path.join(args.out_dir, "prb_phase_diagram_points.pdf")
    fig4.savefig(f4_png, dpi=args.dpi)
    fig4.savefig(f4_pdf)
    plt.close(fig4)
    if not args.quiet:
        print(f"Generated: {f4_png} and {f4_pdf}")

    # =========================================================================
    # FIGURE 5: TRANSLATIONAL VELOCITY <\dot{u}>_t (v_u)
    # =========================================================================
    vu_grid = get_grid(data['v_u'])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    mesh = ax.pcolormesh(X, Y, vu_grid, cmap='coolwarm', shading='nearest', rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(r'$\langle \dot{u} \rangle_t$', fontsize=13)
    cbar.ax.tick_params(direction='in', which='both')

    if has_floquet:
        add_floquet_contours(ax, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)
    add_analyzed_markers(ax, ANALYZED_POINTS)

    ax.set_xlabel(r'$h_0$', fontsize=13)
    ax.set_ylabel(r'$h_\mathrm{a}$', fontsize=13)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    f5_png = os.path.join(args.out_dir, "prb_velocity_u.png")
    f5_pdf = os.path.join(args.out_dir, "prb_velocity_u.pdf")
    fig.savefig(f5_png, dpi=args.dpi)
    fig.savefig(f5_pdf)
    plt.close(fig)
    if not args.quiet:
        print(f"Generated: {f5_png} and {f5_pdf}")

    # =========================================================================
    # FIGURE 6: ANGULAR / PHASE VELOCITY <\dot{\phi}>_t (v_phi)
    # =========================================================================
    vphi_grid = get_grid(data['v_phi'])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    mesh = ax.pcolormesh(X, Y, vphi_grid, cmap='viridis', vmin=0.0, vmax=1.2,
                         shading='nearest', rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(r'$\langle \dot{\phi} \rangle_t$', fontsize=13)
    cbar.ax.tick_params(direction='in', which='both')

    if has_floquet:
        add_floquet_contours(ax, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)
    add_analyzed_markers(ax, ANALYZED_POINTS)

    ax.set_xlabel(r'$h_0$', fontsize=13)
    ax.set_ylabel(r'$h_\mathrm{a}$', fontsize=13)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    f6_png = os.path.join(args.out_dir, "prb_velocity_phi.png")
    f6_pdf = os.path.join(args.out_dir, "prb_velocity_phi.pdf")
    fig.savefig(f6_png, dpi=args.dpi)
    fig.savefig(f6_pdf)
    plt.close(fig)
    if not args.quiet:
        print(f"Generated: {f6_png} and {f6_pdf}")

    # =========================================================================
    # FIGURE 7: COMPOSITE PHASE DIAGRAM & VELOCITIES WITH ANALYZED POINTS
    # =========================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 4.0))

    # (a) Phase Diagram
    m1 = ax1.pcolormesh(X, Y, clean_phase_grid, cmap=phase_cmap, vmin=-0.5, vmax=4.5,
                        shading='nearest', rasterized=True)
    if has_floquet:
        add_floquet_contours(ax1, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)
    add_analyzed_markers(ax1, ANALYZED_POINTS)
    ax1.set_xlabel(r'$h_0$', fontsize=12)
    ax1.set_ylabel(r'$h_\mathrm{a}$', fontsize=12)
    ax1.set_title(r'(a) Phase Diagram', fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 20)
    ax1.legend(handles=phase_patches, loc='lower center', bbox_to_anchor=(0.5, 1.05),
               ncol=5, frameon=True, framealpha=1.0, edgecolor='none',
               handlelength=1.1, handletextpad=0.3, columnspacing=0.8, fontsize=9.5)

    # (b) Velocity u
    m2 = ax2.pcolormesh(X, Y, vu_grid, cmap='coolwarm', shading='nearest', rasterized=True)
    cb2 = fig.colorbar(m2, ax=ax2, pad=0.03)
    cb2.set_label(r'$\langle \dot{u} \rangle_t$', fontsize=12)
    if has_floquet:
        add_floquet_contours(ax2, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)
    add_analyzed_markers(ax2, ANALYZED_POINTS)
    ax2.set_xlabel(r'$h_0$', fontsize=12)
    ax2.set_ylabel(r'$h_\mathrm{a}$', fontsize=12)
    ax2.set_title(r'(b) Velocity $\langle \dot{u} \rangle_t$', fontsize=12)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 20)

    # (c) Velocity phi
    m3 = ax3.pcolormesh(X, Y, vphi_grid, cmap='viridis', vmin=0.0, vmax=1.2,
                        shading='nearest', rasterized=True)
    cb3 = fig.colorbar(m3, ax=ax3, pad=0.03)
    cb3.set_label(r'$\langle \dot{\phi} \rangle_t$', fontsize=12)
    if has_floquet:
        add_floquet_contours(ax3, H_GRID, HAC_GRID, conv_grid, mu_grid, lw=args.line_width)
    add_analyzed_markers(ax3, ANALYZED_POINTS)
    ax3.set_xlabel(r'$h_0$', fontsize=12)
    ax3.set_ylabel(r'$h_\mathrm{a}$', fontsize=12)
    ax3.set_title(r'(c) Angular Velocity $\langle \dot{\phi} \rangle_t$', fontsize=12)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 20)

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    f7_png = os.path.join(args.out_dir, "prb_phase_and_velocities_analyzed.png")
    f7_pdf = os.path.join(args.out_dir, "prb_phase_and_velocities_analyzed.pdf")
    fig.savefig(f7_png, dpi=args.dpi)
    fig.savefig(f7_pdf)
    plt.close(fig)
    if not args.quiet:
        print(f"Generated: {f7_png} and {f7_pdf}")
        print(f"\nAll publication figures successfully created in: {args.out_dir}")

    # =========================================================================
    # OPTIONAL EXPORT TO PAPER FIGURES DIRECTORY
    # =========================================================================
    if args.export_paper:
        paper_fig_dir = os.path.join("paper", "figuras")
        if os.path.isdir(paper_fig_dir):
            import shutil
            target_pdf = os.path.join(paper_fig_dir, "phasediagram_L8.pdf")
            target_png = os.path.join(paper_fig_dir, "phasediagram_L8.png")
            shutil.copyfile(f3_pdf, target_pdf)
            shutil.copyfile(f3_png, target_png)
            # Update smallL.png for compatibility
            shutil.copyfile(f3_png, os.path.join(paper_fig_dir, "smallL.png"))
            if not args.quiet:
                print(f"Exported publication phase diagram to {target_pdf} and {target_png}")

if __name__ == "__main__":
    main()
