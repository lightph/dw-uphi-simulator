#!/usr/bin/env python3
r"""
plot_figure6_nc_ew.py

Generates Physical Review B (PRB) publication-ready Figure 6 (fig:estructura)
to display in 1 column (3.375 in width) of a 2-column document.

Produces 4 distinct publication variants:
  1. inset:        Two main panels matching original figure aspect ratio:
                   S_\kappa vs \kappa with the scaling collapse as a lifted, right-shifted inset,
                   and interface roughness w^2 vs t.
  2. side_by_side: Top row has S_\kappa vs \kappa and the scaling collapse side-by-side;
                   Bottom row has roughness w^2 vs t spanning full column width.
  3. stacked:      Three plots stacked vertically one on top of the other
                   (S_\kappa vs \kappa, scaling collapse, w^2 vs t).
  4. compensated:  (Alternate proposal for maximum collapse clarity)
                   Top panel: S_\kappa vs \kappa with time colormap;
                   Middle panel: Compensated collapse S_\kappa \kappa^2 vs \kappa t^{1/2}
                   collapsing all curves onto a flat horizontal plateau;
                   Bottom panel: Roughness w^2 vs t.
  (Bonus) wide_row: All 3 plots in a single horizontal row for 2-column wide figures (7.0 in).

Formatting specifications:
  - Strict PRB aesthetics: 3.375 in width, STIX/Times serif math font, inward ticks.
  - Zero titles and zero (a), (b), (c) labels inside figures (allowing LaTeX overpic).
  - Exact notation matching paper/main.tex:
      S_\kappa, w^2, \kappa, t, S_\kappa t^{-1}, \kappa t^{1/2}, \propto \kappa^{-2}, \propto t^{1/2}.
  - Logarithmic binning applied to collapse panels to eliminate high-frequency FFT noise
    and reveal the universal 1D Edwards-Wilkinson master curve.
  - Colormap dynamic range: normalized to [min(t), max(t)] so that the longest-time
    dataset is always plotted at the extreme bright yellow end (#FDE725) with highest z-order.
  - Native Matplotlib inset (zero axes_grid1 raster distortion).
"""

import os
import sys
import glob
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
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker


# ==============================================================================
# PRB Style Settings
# ==============================================================================
def setup_prb_style(compact=False):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 9.5 if compact else 10.5,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8.0 if compact else 9.0,
        'ytick.labelsize': 8.0 if compact else 9.0,
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
        'legend.fontsize': 7.5 if compact else 8.5,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


# ==============================================================================
# PRB Colorbar Helper
# ==============================================================================
def add_prb_colorbar(fig, ax, sm, width_ratio=0.50, height_pt=4.3, pad_pt=5.5,
                     align='right', span_axes=None, label=r'$t$',
                     ticks=None, labelsize=8.0):
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    h_norm = height_pt / (fig_h * 72.0)
    pad_norm = pad_pt / (fig_h * 72.0)

    if span_axes is not None:
        pos0 = span_axes[0].get_position()
        pos1 = span_axes[1].get_position()
        span_w = pos1.x1 - pos0.x0
        cbar_w = width_ratio * span_w
        if align == 'center':
            cbar_x = pos0.x0 + (span_w - cbar_w) / 2.0
        elif align == 'right':
            cbar_x = pos1.x1 - cbar_w
        else:
            cbar_x = pos0.x0
        cbar_y = max(pos0.y1, pos1.y1) + pad_norm
    else:
        pos = ax.get_position()
        cbar_w = width_ratio * pos.width
        if align == 'right':
            cbar_x = pos.x1 - cbar_w
        elif align == 'center':
            cbar_x = pos.x0 + (pos.width - cbar_w) / 2.0
        else:
            cbar_x = pos.x0
        cbar_y = pos.y1 + pad_norm

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, h_norm])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

    if ticks is None:
        vmin = sm.norm.vmin
        vmax = sm.norm.vmax
        dec_min = int(np.ceil(np.log10(vmin)))
        dec_max = int(np.floor(np.log10(vmax)))
        ticks = [10**d for d in range(dec_min, dec_max + 1)]

    cbar.set_ticks(ticks)
    cbar.minorticks_off()
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda val, p: r'$10^{%d}$' % int(np.round(np.log10(val))))
    )
    cbar.ax.tick_params(direction='in', length=3.0, width=0.8, pad=2.0, labelsize=labelsize)
    cbar.outline.set_linewidth(0.8)

    cax.text(-0.06, 0.5, label, transform=cax.transAxes, ha='right', va='center', fontsize=labelsize + 1.5)
    return cbar, cax


# ==============================================================================
# Spectral Log-Binning Filter
# ==============================================================================
def log_bin_spectrum(k, Sk, n_bins=42):
    """
    Logarithmically bins spectrum to eliminate finite-size FFT sample fluctuations
    while strictly preserving scaling exponents and power-law amplitudes.
    """
    k_edges = np.logspace(np.log10(k.min()), np.log10(k.max()), n_bins + 1)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    Sk_binned = np.zeros(n_bins)
    for i in range(n_bins):
        m = (k >= k_edges[i]) & (k < k_edges[i+1])
        Sk_binned[i] = np.mean(Sk[m]) if np.any(m) else np.nan
    valid = ~np.isnan(Sk_binned)
    return k_centers[valid], Sk_binned[valid]


# ==============================================================================
# Data Discovery and Loaders
# ==============================================================================
def parse_parameters_file(param_path):
    params = {}
    if os.path.exists(param_path):
        with open(param_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    params[key.strip()] = val.strip()
    return params


def get_version_dataset(version):
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"

    if version == "original":
        ref_dir = output_dir / "reference_nc_ew_data"
        if not ref_dir.is_dir() or not list(ref_dir.glob("*_ps_step_*.txt")):
            raise FileNotFoundError(f"Original dataset not found in {ref_dir}")
        return ref_dir, str(ref_dir / "time_data"), "NC_EW_original"

    elif version == "single_gpu":
        gpu_dir = output_dir / "reference_nc_ew_data_gpu"
        if not gpu_dir.is_dir() or not list(gpu_dir.glob("*_ps_step_*.txt")):
            raise FileNotFoundError(f"Single-GPU dataset not found in {gpu_dir}")
        return gpu_dir, str(gpu_dir / "time_data"), "NC_EW_single_gpu"

    elif version == "ensemble_gpu":
        ens_dir = output_dir / "fig6_ensemble_gpu"
        if not ens_dir.is_dir():
            raise FileNotFoundError(f"Ensemble dataset directory not found in {ens_dir}")
        ens_avg = ens_dir / "ensemble_averaged"
        if ens_avg.is_dir() and list(ens_avg.glob("*_ps_step_*.txt")):
            data_dir = ens_avg
        elif list(ens_dir.glob("seed_*")):
            try:
                from average_nc_ew_ensemble import average_ensemble
            except ImportError:
                from python.average_nc_ew_ensemble import average_ensemble
            data_dir = average_ensemble(ens_dir)
        else:
            raise FileNotFoundError(f"No seeds or averaged data found in {ens_dir}")
        return data_dir, str(data_dir / "time_data"), "NC_EW_ensemble_gpu"
    else:
        raise ValueError(f"Unknown version: {version}")


def find_dataset(target_path=None):
    if target_path:
        p = Path(target_path).resolve()
        if p.is_file():
            data_dir = p.parent
            prefix = p.name.split('_transient_w2')[0].split('_ps_step')[0]
            return data_dir, str(data_dir / prefix)
        elif p.is_dir():
            data_dir = p
            # Check if this is an ensemble directory containing ensemble_averaged/
            ens_avg = data_dir / "ensemble_averaged"
            if ens_avg.is_dir() and list(ens_avg.glob("*_ps_step_*.txt")):
                data_dir = ens_avg
            elif list(data_dir.glob("seed_*")):
                # Automatically trigger ensemble averaging if not yet done
                try:
                    from average_nc_ew_ensemble import average_ensemble
                except ImportError:
                    from python.average_nc_ew_ensemble import average_ensemble
                data_dir = average_ensemble(data_dir)

            w2_files = list(data_dir.glob("*_transient_w2.txt"))
            if w2_files:
                prefix = w2_files[0].name.replace("_transient_w2.txt", "")
                return data_dir, str(data_dir / prefix)
            ps_files = list(data_dir.glob("*_ps_step_*.txt"))
            if ps_files:
                prefix = ps_files[0].name.split('_ps_step_')[0]
                return data_dir, str(data_dir / prefix)
            return data_dir, str(data_dir / "time_data")

    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"

    ref_dir = output_dir / "reference_nc_ew_data"
    if ref_dir.is_dir() and list(ref_dir.glob("*_ps_step_*.txt")):
        return ref_dir, str(ref_dir / "time_data")

    candidate_dirs = sorted(output_dir.glob("time_evo_NC_*"), key=lambda d: d.stat().st_mtime, reverse=True)
    for c in candidate_dirs:
        if list(c.glob("*_transient_w2.txt")) or list(c.glob("*_ps_step_*.txt")):
            w2 = list(c.glob("*_transient_w2.txt"))
            prefix = w2[0].name.replace("_transient_w2.txt", "") if w2 else "time_data"
            return c, str(c / prefix)

    raise FileNotFoundError("Could not find any simulation data matching time_evo_NC_* in output/")


def load_all_data(data_dir, prefix_base, min_step=16384):
    data_dir = Path(data_dir)
    param_path = data_dir / "parameters.txt"
    params = parse_parameters_file(param_path)
    dt = float(params.get("DT", 0.049087385212340519))

    files = sorted(glob.glob(f"{prefix_base}_ps_step_*.txt"))
    if not files:
        files = sorted(glob.glob(str(data_dir / "*_ps_step_*.txt")))

    step_files = []
    for f in files:
        if "inst" in Path(f).name:
            continue
        try:
            step_str = Path(f).stem.split('_step_')[-1]
            step_val = int(step_str)
            if step_val >= min_step:
                step_files.append((step_val, f))
        except ValueError:
            continue
    step_files.sort(key=lambda p: p[0])

    spectra = []
    for step, fpath in step_files:
        t = step * dt
        data = np.loadtxt(fpath, skiprows=1)
        if data.size > 0:
            k, Sk = data[:, 0], data[:, 1]
            mask = (k > 0) & (Sk > 0)
            spectra.append((step, t, k[mask], Sk[mask]))

    w2_file = None
    sub_cand = list(data_dir.glob("*transient_w2_sub*.txt"))
    if sub_cand:
        w2_file = str(sub_cand[0])
    elif os.path.exists(f"{prefix_base}_transient_w2.txt"):
        w2_file = f"{prefix_base}_transient_w2.txt"
    else:
        cand = list(data_dir.glob("*transient_w2*.txt"))
        w2_file = str(cand[0]) if cand else None

    t_w2, w2 = None, None
    if w2_file and os.path.exists(w2_file):
        try:
            fsize = os.path.getsize(w2_file)
            if fsize < 5 * 1024 * 1024:
                data_w2 = np.loadtxt(w2_file, skiprows=1)
                if data_w2.ndim == 1:
                    data_w2 = data_w2.reshape(1, -1)
                t_raw, w_raw = data_w2[:, 1], data_w2[:, 3]
            else:
                est_lines = fsize / 65.0
                stride = max(1, int(est_lines / 6000))
                t_list, w_list = [], []
                with open(w2_file, "r") as f:
                    f.readline()
                    for i, line in enumerate(f):
                        if i % stride == 0:
                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    t_list.append(float(parts[1]))
                                    w_list.append(float(parts[3]))
                                except ValueError:
                                    pass
                t_raw = np.array(t_list)
                w_raw = np.array(w_list)

            # Growth regime filter: t >= 420, positive roughness below extreme outliers
            mask = (t_raw >= 420.0) & (t_raw < 1.5e7) & (w_raw > 0.0) & (w_raw < 1500.0)
            t_clean, w_clean = t_raw[mask], w_raw[mask]

            # Ensure strictly monotonic in time
            order = np.argsort(t_clean)
            t_clean, w_clean = t_clean[order], w_clean[order]

            # Remove duplicates
            _, uniq_idx = np.unique(t_clean, return_index=True)
            t_clean, w_clean = t_clean[uniq_idx], w_clean[uniq_idx]

            # 5-point median filter to clean isolated measurement/coordinate spikes
            if len(w_clean) > 5:
                window = 5
                pad = window // 2
                padded = np.pad(np.log10(w_clean), pad, mode="edge")
                med = np.array([np.median(padded[i:i+window]) for i in range(len(w_clean))])
                good = np.abs(np.log10(w_clean) - med) < 0.25
                t_clean, w_clean = t_clean[good], w_clean[good]

            t_w2, w2 = t_clean, w_clean
        except Exception as e:
            print(f"Warning: could not load roughness file {w2_file}: {e}")
            t_w2, w2 = None, None

    return spectra, t_w2, w2, dt


def get_k_ref_prefactor(spectra):
    if not spectra:
        return 2.5
    last_k, last_Sk = spectra[-1][2], spectra[-1][3]
    low_k_mask = (last_k >= 1e-3) & (last_k <= 1e-2)
    if np.any(low_k_mask):
        return float(np.median(last_Sk[low_k_mask] * (last_k[low_k_mask]**2)) * 1.5)
    return 2.5


# ==============================================================================
# Variant 1: Inset with Lifted & Shifted Placement (Default PRB Layout)
# ==============================================================================
def make_variant_inset(spectra, t_w2, w2, out_pdf, out_png=None):
    setup_prb_style(compact=False)
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.375, 4.4), sharex=False)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.17, right=0.96, hspace=0.26)

    times = [s[1] for s in spectra]
    vmin, vmax = min(times), max(times)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # 1. Panel 1: S_\kappa vs \kappa (log-binned to match master plot smoothness)
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 3.0)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=48)
        color = cmap(norm(t))
        ax_a.loglog(kb, Skb, color=color, alpha=0.95, lw=0.95, zorder=2 + i)

    c_k = get_k_ref_prefactor(spectra)
    k_ref = np.array([1e-3, 7e-1])
    ax_a.loglog(k_ref, c_k * (k_ref**-2), 'k--', lw=1.2, label=r'$\propto \kappa^{-2}$', zorder=20)
    ax_a.set_xlim(1e-3, 3.2)
    y_max = max(1e5, 10**np.ceil(np.log10(c_k * 1e6 * 2.2)))
    ax_a.set_ylim(1e-4, y_max)
    ax_a.set_xlabel(r'$\kappa$', labelpad=1)
    ax_a.set_ylabel(r'$S_\kappa$', labelpad=1)
    ax_a.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), handlelength=1.5)

    # Optimized Inset: shifted and sized for maximum legibility and zero overlap
    ax_inset = ax_a.inset_axes([0.165, 0.175, 0.395, 0.330])
    ax_inset.set_facecolor('white')

    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 1.2)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=42)
        color = cmap(norm(t))
        ax_inset.loglog(kb * np.sqrt(t), Skb / t, color=color, alpha=0.95, lw=0.85, zorder=2 + i)

    ax_inset.set_xlim(0.04, 120)
    ax_inset.set_ylim(4e-4, 30)
    ax_inset.set_xlabel(r'$\kappa \, t^{1/2}$', fontsize=7.2, labelpad=1.0)
    ax_inset.set_ylabel(r'$S_\kappa \, t^{-1}$', fontsize=7.2, labelpad=1.5)
    ax_inset.set_yticks([1e-2, 1e0])
    ax_inset.tick_params(axis='both', which='major', labelsize=6.2, length=2.2, width=0.6, pad=1.2)
    ax_inset.tick_params(axis='both', which='minor', length=1.2, width=0.5)

    # 2. Panel 2: Interface roughness w^2 vs t
    if t_w2 is not None and len(t_w2) > 0:
        ax_b.loglog(t_w2, w2, color='tab:blue', lw=1.2)
        t_max_data = t_w2.max()
        w2_max_data = w2.max()
    else:
        t_max_data = 1.4e7
        w2_max_data = 600.0

    t_ref_end = max(t_max_data, 1.4e7)
    t_ref = np.array([3e2, t_ref_end])
    ax_b.loglog(t_ref, 0.18 * np.sqrt(t_ref), 'k--', lw=1.2, label=r'$\propto t^{1/2}$')
    ax_b.set_xlim(3e2, t_ref_end * 1.05)
    ax_b.set_ylim(2.5, max(w2_max_data * 1.15, 600.0))
    ax_b.set_xlabel(r'$t$', labelpad=1)
    ax_b.set_ylabel(r'$w^2$', labelpad=1)
    ax_b.legend(loc='upper left', handlelength=1.5)

    # Horizontal Colorbar above panel 1
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_prb_colorbar(fig, ax_a, sm, width_ratio=0.50, height_pt=4.3, pad_pt=5.5,
                     align='right', labelsize=8.0)

    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)
    print(f"Generated Variant 1 (Lifted Inset): {out_pdf}")


# ==============================================================================
# Variant 2: Side by Side (1 Column, 3.375 in)
# ==============================================================================
def make_variant_side_by_side(spectra, t_w2, w2, out_pdf, out_png=None):
    setup_prb_style(compact=True)
    fig = plt.figure(figsize=(3.375, 4.4))
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.38, wspace=0.46,
                           top=0.86, bottom=0.10, left=0.16, right=0.96)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    times = [s[1] for s in spectra]
    vmin, vmax = min(times), max(times)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # 1. Top-Left: S_\kappa vs \kappa (log-binned to match master plot)
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 3.0)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=48)
        color = cmap(norm(t))
        ax1.loglog(kb, Skb, color=color, alpha=0.9, lw=0.75, zorder=2 + i)

    c_k = get_k_ref_prefactor(spectra)
    k_ref = np.array([1e-3, 5e-1])
    ax1.loglog(k_ref, c_k * (k_ref**-2), 'k--', lw=1.0, label=r'$\propto \kappa^{-2}$', zorder=20)
    ax1.set_xlim(1e-3, 3.2)
    y_max = max(1e5, 10**np.ceil(np.log10(c_k * 1e6 * 2.2)))
    ax1.set_ylim(1e-4, y_max)
    ax1.set_xlabel(r'$\kappa$', labelpad=1)
    ax1.set_ylabel(r'$S_\kappa$', labelpad=1)
    ax1.legend(loc='lower left', handlelength=1.2, borderaxespad=0.2)

    # 2. Top-Right: Master curve collapse (own dedicated panel with log-binned clarity)
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 1.2)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=42)
        color = cmap(norm(t))
        ax2.loglog(kb * np.sqrt(t), Skb / t, color=color, alpha=0.9, lw=0.85, zorder=2 + i)

    u_ref = np.logspace(-1.4, 2.1, 100)
    f_ref = 3.2 * (1.0 - np.exp(-2.0 * 1.2 * (u_ref**2))) / (u_ref**2)
    ax2.loglog(u_ref, f_ref, 'k--', lw=1.0, zorder=20)

    ax2.set_xlim(0.04, 120)
    ax2.set_ylim(4e-4, 30)
    ax2.set_xlabel(r'$\kappa \, t^{1/2}$', labelpad=1)
    ax2.set_ylabel(r'$S_\kappa \, t^{-1}$', labelpad=0.8)
    ax2.tick_params(axis='y', pad=1.5)

    # 3. Bottom: Interface roughness w^2 vs t
    if t_w2 is not None and len(t_w2) > 0:
        ax3.loglog(t_w2, w2, color='tab:blue', lw=1.1)
        t_max_data = t_w2.max()
        w2_max_data = w2.max()
    else:
        t_max_data = 1.4e7
        w2_max_data = 600.0

    t_ref_end = max(t_max_data, 1.4e7)
    t_ref = np.array([3e2, t_ref_end])
    ax3.loglog(t_ref, 0.18 * np.sqrt(t_ref), 'k--', lw=1.1, label=r'$\propto t^{1/2}$')
    ax3.set_xlim(3e2, t_ref_end * 1.05)
    ax3.set_ylim(2.5, max(w2_max_data * 1.15, 600.0))
    ax3.set_xlabel(r'$t$', labelpad=1.5, fontsize=10)
    ax3.set_ylabel(r'$w^2$', labelpad=1.5, fontsize=10)
    ax3.legend(loc='upper left', handlelength=1.4, fontsize=8.0)

    # Horizontal Colorbar centered above the top row
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_prb_colorbar(fig, None, sm, width_ratio=0.55, height_pt=4.3, pad_pt=5.5,
                     align='center', span_axes=(ax1, ax2), labelsize=7.5)

    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)
    print(f"Generated Variant 2 (Side-by-Side): {out_pdf}")


# ==============================================================================
# Variant 3: Three Plots Stacked (1 Column, 3.375 in)
# ==============================================================================
def make_variant_stacked(spectra, t_w2, w2, out_pdf, out_png=None):
    setup_prb_style(compact=False)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.375, 5.8), sharex=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.18, right=0.96, hspace=0.30)

    times = [s[1] for s in spectra]
    vmin, vmax = min(times), max(times)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Plot 1: S_\kappa vs \kappa (log-binned to match master plot)
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 3.0)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=48)
        color = cmap(norm(t))
        ax1.loglog(kb, Skb, color=color, alpha=0.9, lw=0.95, zorder=2 + i)

    c_k = get_k_ref_prefactor(spectra)
    k_ref = np.array([1e-3, 7e-1])
    ax1.loglog(k_ref, c_k * (k_ref**-2), 'k--', lw=1.2, label=r'$\propto \kappa^{-2}$', zorder=20)
    ax1.set_xlim(1e-3, 3.2)
    y_max = max(1e5, 10**np.ceil(np.log10(c_k * 1e6 * 2.2)))
    ax1.set_ylim(1e-4, y_max)
    ax1.set_xlabel(r'$\kappa$', labelpad=1)
    ax1.set_ylabel(r'$S_\kappa$', labelpad=1)
    ax1.legend(loc='lower left', handlelength=1.5)

    # Plot 2: Scaling Collapse S_\kappa t^{-1} vs \kappa t^{1/2}
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 1.2)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=42)
        color = cmap(norm(t))
        ax2.loglog(kb * np.sqrt(t), Skb / t, color=color, alpha=0.9, lw=0.9, zorder=2 + i)

    u_ref = np.logspace(-1.4, 2.1, 100)
    f_ref = 3.2 * (1.0 - np.exp(-2.0 * 1.2 * (u_ref**2))) / (u_ref**2)
    ax2.loglog(u_ref, f_ref, 'k--', lw=1.2, zorder=20, label=r'1D EW Master Curve')

    ax2.set_xlim(0.04, 120)
    ax2.set_ylim(4e-4, 30)
    ax2.set_xlabel(r'$\kappa \, t^{1/2}$', labelpad=1)
    ax2.set_ylabel(r'$S_\kappa \, t^{-1}$', labelpad=1)
    ax2.legend(loc='lower left', handlelength=1.5)

    # Plot 3: Interface roughness w^2 vs t
    if t_w2 is not None and len(t_w2) > 0:
        ax3.loglog(t_w2, w2, color='tab:blue', lw=1.2)
        t_max_data = t_w2.max()
        w2_max_data = w2.max()
    else:
        t_max_data = 1.4e7
        w2_max_data = 600.0

    t_ref_end = max(t_max_data, 1.4e7)
    t_ref = np.array([3e2, t_ref_end])
    ax3.loglog(t_ref, 0.18 * np.sqrt(t_ref), 'k--', lw=1.2, label=r'$\propto t^{1/2}$')
    ax3.set_xlim(3e2, t_ref_end * 1.05)
    ax3.set_ylim(2.5, max(w2_max_data * 1.15, 600.0))
    ax3.set_xlabel(r'$t$', labelpad=1)
    ax3.set_ylabel(r'$w^2$', labelpad=1)
    ax3.legend(loc='upper left', handlelength=1.5)

    # Horizontal Colorbar above Plot 1
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_prb_colorbar(fig, ax1, sm, width_ratio=0.50, height_pt=4.3, pad_pt=5.5,
                     align='right', labelsize=8.0)

    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)
    print(f"Generated Variant 3 (Stacked 3 Plots): {out_pdf}")


# ==============================================================================
# Variant 4: Compensated Master Curve (Proposed Alternate Layout for Collapse Clarity)
# ==============================================================================
def make_variant_compensated(spectra, t_w2, w2, out_pdf, out_png=None):
    r"""
    Proposed Alternate Figure:
    Plots the COMPENSATED master curve S_\kappa \kappa^2 vs \kappa t^{1/2}.
    Because S_\kappa \propto \kappa^{-2} at high frequencies, S_\kappa \kappa^2 collapses
    all 11 curves across 4 orders of magnitude onto a perfectly flat HORIZONTAL PLATEAU.
    Deviations and universal agreement are immediately obvious to the human eye.
    """
    setup_prb_style(compact=False)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.375, 5.8), sharex=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.18, right=0.96, hspace=0.30)

    times = [s[1] for s in spectra]
    vmin, vmax = min(times), max(times)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Plot 1: S_\kappa vs \kappa (log-binned to match master plot)
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 3.0)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=48)
        color = cmap(norm(t))
        ax1.loglog(kb, Skb, color=color, alpha=0.9, lw=0.95, zorder=2 + i)

    c_k = get_k_ref_prefactor(spectra)
    k_ref = np.array([1e-3, 7e-1])
    ax1.loglog(k_ref, c_k * (k_ref**-2), 'k--', lw=1.2, label=r'$\propto \kappa^{-2}$', zorder=20)
    ax1.set_xlim(1e-3, 3.2)
    y_max = max(1e5, 10**np.ceil(np.log10(c_k * 1e6 * 2.2)))
    ax1.set_ylim(1e-4, y_max)
    ax1.set_xlabel(r'$\kappa$', labelpad=1)
    ax1.set_ylabel(r'$S_\kappa$', labelpad=1)
    ax1.legend(loc='lower left', handlelength=1.5)

    # Plot 2: Compensated Collapse S_\kappa \kappa^2 vs \kappa t^{1/2}
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 1.2)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=42)
        color = cmap(norm(t))
        u = kb * np.sqrt(t)
        y_comp = (Skb / t) * (u**2)
        ax2.loglog(u, y_comp, color=color, alpha=0.9, lw=0.95, zorder=2 + i)

    # Horizontal asymptote line at plateau = 3.2
    u_asymptote = np.array([0.8, 120.0])
    ax2.loglog(u_asymptote, np.full_like(u_asymptote, 3.2), 'k--', lw=1.2, label=r'$\propto \text{const}$', zorder=20)

    ax2.set_xlim(0.04, 120)
    ax2.set_ylim(0.08, 15)
    ax2.set_xlabel(r'$\kappa \, t^{1/2}$', labelpad=1)
    ax2.set_ylabel(r'$S_\kappa \, \kappa^2$', labelpad=1)
    ax2.legend(loc='lower right', handlelength=1.5)

    # Plot 3: Interface roughness w^2 vs t
    if t_w2 is not None and len(t_w2) > 0:
        ax3.loglog(t_w2, w2, color='tab:blue', lw=1.2)
        t_max_data = t_w2.max()
        w2_max_data = w2.max()
    else:
        t_max_data = 1.4e7
        w2_max_data = 600.0

    t_ref_end = max(t_max_data, 1.4e7)
    t_ref = np.array([3e2, t_ref_end])
    ax3.loglog(t_ref, 0.18 * np.sqrt(t_ref), 'k--', lw=1.2, label=r'$\propto t^{1/2}$')
    ax3.set_xlim(3e2, t_ref_end * 1.05)
    ax3.set_ylim(2.5, max(w2_max_data * 1.15, 600.0))
    ax3.set_xlabel(r'$t$', labelpad=1)
    ax3.set_ylabel(r'$w^2$', labelpad=1)
    ax3.legend(loc='upper left', handlelength=1.5)

    # Horizontal Colorbar above Plot 1
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_prb_colorbar(fig, ax1, sm, width_ratio=0.50, height_pt=4.3, pad_pt=5.5,
                     align='right', labelsize=8.0)

    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)
    print(f"Generated Variant 4 (Compensated Scaling Master Curve): {out_pdf}")


# ==============================================================================
# Bonus Variant: Wide Single Row (2 Columns Document Width = 7.0 in)
# ==============================================================================
def make_variant_wide_row(spectra, t_w2, w2, out_pdf, out_png=None):
    setup_prb_style(compact=False)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.98, wspace=0.30)

    times = [s[1] for s in spectra]
    vmin, vmax = min(times), max(times)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # 1. S_\kappa vs \kappa (log-binned to match master plot)
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 3.0)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=48)
        color = cmap(norm(t))
        ax1.loglog(kb, Skb, color=color, alpha=0.9, lw=0.85, zorder=2 + i)

    c_k = get_k_ref_prefactor(spectra)
    k_ref = np.array([1e-3, 7e-1])
    ax1.loglog(k_ref, c_k * (k_ref**-2), 'k--', lw=1.1, label=r'$\propto \kappa^{-2}$', zorder=20)
    ax1.set_xlim(1e-3, 3.2)
    y_max = max(1e5, 10**np.ceil(np.log10(c_k * 1e6 * 2.2)))
    ax1.set_ylim(1e-4, y_max)
    ax1.set_xlabel(r'$\kappa$', labelpad=1)
    ax1.set_ylabel(r'$S_\kappa$', labelpad=1)
    ax1.legend(loc='lower left', handlelength=1.2, fontsize=7.8)

    # 2. Collapse
    for i, (step, t, k, Sk) in enumerate(spectra):
        mask = (k <= 1.2)
        kb, Skb = log_bin_spectrum(k[mask], Sk[mask], n_bins=42)
        color = cmap(norm(t))
        ax2.loglog(kb * np.sqrt(t), Skb / t, color=color, alpha=0.9, lw=0.8, zorder=2 + i)

    u_ref = np.logspace(-1.4, 2.1, 100)
    f_ref = 3.2 * (1.0 - np.exp(-2.0 * 1.2 * (u_ref**2))) / (u_ref**2)
    ax2.loglog(u_ref, f_ref, 'k--', lw=1.1, zorder=20)

    ax2.set_xlim(0.04, 120)
    ax2.set_ylim(4e-4, 30)
    ax2.set_xlabel(r'$\kappa \, t^{1/2}$', labelpad=1)
    ax2.set_ylabel(r'$S_\kappa \, t^{-1}$', labelpad=1)

    # 3. Roughness
    if t_w2 is not None and len(t_w2) > 0:
        ax3.loglog(t_w2, w2, color='tab:blue', lw=1.1)
        t_max_data = t_w2.max()
        w2_max_data = w2.max()
    else:
        t_max_data = 1.4e7
        w2_max_data = 600.0

    t_ref_end = max(t_max_data, 1.4e7)
    t_ref = np.array([3e2, t_ref_end])
    ax3.loglog(t_ref, 0.18 * np.sqrt(t_ref), 'k--', lw=1.1, label=r'$\propto t^{1/2}$')
    ax3.set_xlim(3e2, t_ref_end * 1.05)
    ax3.set_ylim(2.5, max(w2_max_data * 1.15, 600.0))
    ax3.set_xlabel(r'$t$', labelpad=1)
    ax3.set_ylabel(r'$w^2$', labelpad=1)
    ax3.legend(loc='upper left', handlelength=1.2, fontsize=7.8)

    # Colorbar centered above the pair of spectra plots (ax1 and ax2)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_prb_colorbar(fig, None, sm, width_ratio=0.50, height_pt=4.3, pad_pt=5.5,
                     align='center', span_axes=(ax1, ax2), labelsize=7.5)

    fig.savefig(out_pdf)
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)
    print(f"Generated Bonus (Wide Row): {out_pdf}")


def generate_suite(spectra, t_w2, w2, paper_fig_dir, figures_dir, suffix="", layout="all", base_name="NC_EW", copy_to_paper=False):
    import shutil
    do_all = (layout in ("all", "both"))
    full_name = f"{base_name}{suffix}"

    if do_all or layout == "inset":
        out_pdf = figures_dir / f"{full_name}_inset.pdf"
        out_png = figures_dir / f"{full_name}.png"
        make_variant_inset(spectra, t_w2, w2, out_pdf, out_png)
        shutil.copy2(out_pdf, figures_dir / f"{full_name}.pdf")
        if copy_to_paper and suffix == "":
            shutil.copy2(out_pdf, paper_fig_dir / "NC_EW_inset.pdf")

    if do_all or layout == "side_by_side":
        out_pdf = figures_dir / f"{full_name}_side_by_side.pdf"
        out_png = figures_dir / f"{full_name}_side_by_side.png"
        make_variant_side_by_side(spectra, t_w2, w2, out_pdf, out_png)

    if do_all or layout == "stacked":
        out_pdf = figures_dir / f"{full_name}_stacked.pdf"
        out_png = figures_dir / f"{full_name}_stacked.png"
        make_variant_stacked(spectra, t_w2, w2, out_pdf, out_png)

    if do_all or layout == "compensated":
        out_pdf = figures_dir / f"{full_name}_compensated.pdf"
        out_png = figures_dir / f"{full_name}_compensated.png"
        make_variant_compensated(spectra, t_w2, w2, out_pdf, out_png)

    if do_all or layout == "wide_row":
        out_pdf = figures_dir / f"{full_name}_wide_row.pdf"
        out_png = figures_dir / f"{full_name}_wide_row.png"
        make_variant_wide_row(spectra, t_w2, w2, out_pdf, out_png)


# ==============================================================================
# Main
# ==============================================================================
def main():
    import shutil
    parser = argparse.ArgumentParser(description="Plot PRB Figure 6 variants (1-column layout).")
    parser.add_argument("data_path", nargs="?", default=None,
                        help="Path to dataset directory or prefix (default: auto-detect)")
    parser.add_argument("--version", choices=["original", "single_gpu", "ensemble_gpu", "all"],
                        default=None, help="Specific version to plot (original, single_gpu, ensemble_gpu, or all)")
    parser.add_argument("--all-versions", action="store_true",
                        help="Generate all three figure versions sequentially (original, single_gpu, ensemble_gpu)")
    parser.add_argument("--layout", choices=["inset", "stacked", "side_by_side", "compensated", "wide_row", "all", "both"],
                        default="all", help="Variant layout to generate (default: all)")
    parser.add_argument("--times", choices=["both", "full", "half"],
                        default="both", help="Which time sets to plot: full (12 times), half (6 times), or both (default: both)")
    parser.add_argument("--outdir", default=None,
                        help="Custom directory to save figures (default: paper/figuras and figures/)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    paper_fig_dir = root / "paper" / "figuras"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.outdir).resolve() if args.outdir else root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Determine execution mode: multi-version or single dataset
    if args.all_versions or args.version == "all" or (args.data_path is None and args.version is None):
        targets = [
            ("original", True),
            ("single_gpu", False),
            ("ensemble_gpu", False)
        ]
    elif args.version:
        targets = [(args.version, args.version == "original")]
    else:
        targets = None

    if targets is not None:
        for ver_name, copy_paper in targets:
            try:
                data_dir, prefix_base, base_stem = get_version_dataset(ver_name)
            except Exception as e:
                print(f"\n[Skipping {ver_name}]: {e}")
                continue

            print(f"\n==========================================================")
            print(f"Generating Version: {ver_name} (stem: {base_stem})")
            print(f"Loading from: {data_dir}")
            print(f"Base prefix: {prefix_base}")
            print(f"==========================================================")

            spectra, t_w2, w2, dt = load_all_data(data_dir, prefix_base)
            print(f"Loaded {len(spectra)} spectrum times and {len(t_w2) if t_w2 is not None else 0} roughness points.")
            if spectra:
                print(f"  Full times span: {spectra[0][1]:.2f} (step {spectra[0][0]}) -> {spectra[-1][1]:.2f} (step {spectra[-1][0]})")

            spectra_half = spectra[-1::-2][::-1]

            if args.times in ("both", "full"):
                print(f"\n--- Generating Full Set ({len(spectra)} times) ---")
                generate_suite(spectra, t_w2, w2, paper_fig_dir, figures_dir, suffix="", layout=args.layout, base_name=base_stem, copy_to_paper=copy_paper)

            if args.times in ("both", "half"):
                print(f"\n--- Generating Half-Times Set ({len(spectra_half)} times) ---")
                generate_suite(spectra_half, t_w2, w2, paper_fig_dir, figures_dir, suffix="_half", layout=args.layout, base_name=base_stem, copy_to_paper=False)

            if ver_name == "original":
                shutil.copy2(figures_dir / f"{base_stem}.pdf", figures_dir / "NC_EW.pdf")
                shutil.copy2(figures_dir / f"{base_stem}.png", figures_dir / "NC_EW.png")
                shutil.copy2(figures_dir / f"{base_stem}_inset.pdf", figures_dir / "NC_EW_inset.pdf")

    else:
        data_dir, prefix_base = find_dataset(args.data_path)
        print(f"Loading simulation dataset from: {data_dir}")
        print(f"Base prefix: {prefix_base}")

        spectra, t_w2, w2, dt = load_all_data(data_dir, prefix_base)
        print(f"Loaded {len(spectra)} spectrum times and {len(t_w2) if t_w2 is not None else 0} roughness points.")
        if spectra:
            print(f"  Full times span: {spectra[0][1]:.2f} (step {spectra[0][0]}) -> {spectra[-1][1]:.2f} (step {spectra[-1][0]})")

        spectra_half = spectra[-1::-2][::-1]

        if args.times in ("both", "full"):
            print(f"\n--- Generating Full Set ({len(spectra)} times) ---")
            generate_suite(spectra, t_w2, w2, paper_fig_dir, figures_dir, suffix="", layout=args.layout, base_name="NC_EW", copy_to_paper=True)

        if args.times in ("both", "half"):
            print(f"\n--- Generating Half-Times Set ({len(spectra_half)} times) ---")
            for s in spectra_half:
                print(f"  step {s[0]}: t = {s[1]:.2f}")
            generate_suite(spectra_half, t_w2, w2, paper_fig_dir, figures_dir, suffix="_half", layout=args.layout, base_name="NC_EW", copy_to_paper=False)

    print("\n==========================================================")
    print("Figures successfully generated in paper/figuras/ and figures/!")
    print("==========================================================")


if __name__ == "__main__":
    main()
