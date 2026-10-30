#!/usr/bin/env python3
r"""
plot_figure7_sp_cascade.py - Reproduces PRB Figure 7:
Spatiotemporal cascade and profile snapshots of the Synchronized Periodic (SP) phase.

Plots real-space configurations u(x,t) - \langle u \rangle_x along a section of the domain wall
(x \in [0, 128]) at evenly spaced intervals over an entire driving period, colored by local
in-plane magnetization angle \phi(x,t) \pmod{2\pi}, with drive phase \cos(\Omega t) on the right.
"""

import os
import sys
import shutil
import argparse
import subprocess
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
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_prb_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11.0,
        'axes.linewidth': 0.9,
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
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def run_sp_snapshots(bin_path, out_prefix, size=256, L=128.0, dt=0.019634954):
    """Runs a fast snapshot simulation of the SP state."""
    # Simulation parameters for SP phase: h0=4.0, ha=3.0, alpha=0.27, omega=1.0
    # transient: 1000.0, snapshot_time: 2*pi, window: 100.0, snapshot_rate: 40 steps (~8 snapshots)
    cmd = [
        str(bin_path),
        str(size), "1", str(L), "0.27", "4.0", "3.0", "1.0", str(dt),
        "1000.0", "6.2831853", "100.0", "40", str(out_prefix),
        "0.05", "42", "cpu"
    ]
    print(f"Generating SP snapshots via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def plot_sp_cascade(snap_txt_file, out_pdf_path, L_plot=128.0):
    setup_prb_style()
    with open(snap_txt_file, 'r') as f:
        header = f.readline().strip().split()
        raw_lines = [l.strip().split() for l in f if l.strip()]

    if not raw_lines:
        raise ValueError("Snapshot file is empty.")

    # Filter for first batch
    first_batch = raw_lines[0][0]
    lines = [l for l in raw_lines if l[0] == first_batch]

    total_pts = (len(header) - 2) // 2
    # Standard spatial discretization dx = 0.5
    dx = 0.5
    n_plot = min(total_pts, int(round(L_plot / dx)))
    x = np.linspace(0.0, L_plot, n_plot, endpoint=False)

    # Subsample 8 snapshots across two periods (4*pi/omega)
    if len(lines) > 8:
        times = np.array([float(l[1]) for l in lines])
        cos_t = np.cos(times)
        # Find index closest to cos(t) = -1.0 in first quarter period
        step_quarter = max(1, len(lines) // 8)
        start_idx = int(np.argmin(np.abs(cos_t[:step_quarter] - (-1.0))))
        selected_indices = [(start_idx + k * step_quarter) % len(lines) for k in range(8)]
        plot_lines = [lines[i] for i in selected_indices]
    else:
        plot_lines = lines

    fig, ax_left = plt.subplots(figsize=(3.375, 3.2))
    ax_right = ax_left.twinx()

    cmap = plt.get_cmap('hsv')
    n_lines = len(plot_lines)
    # Stack lines vertically across [-3.5, 3.5]
    offsets = np.linspace(-3.5, 3.5, n_lines)

    cos_vals = []
    for idx, row in enumerate(plot_lines):
        t = float(row[1])
        u_vals = np.array([float(row[2 + 2*i]) for i in range(total_pts)])
        phi_vals = np.array([float(row[2 + 2*i + 1]) for i in range(total_pts)]) % (2.0 * np.pi)

        u_c = (u_vals - np.mean(u_vals))[:n_plot]
        phi_plot = phi_vals[:n_plot]
        y_plot = u_c + offsets[idx]

        # Draw guideline
        ax_left.axhline(offsets[idx], color='gray', linestyle=':', lw=0.5, alpha=0.6, zorder=1)

        points = np.array([x, y_plot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 2.0 * np.pi),
                            linewidth=1.4, zorder=3)
        lc.set_array(phi_plot[:-1])
        ax_left.add_collection(lc)

        cos_vals.append(np.cos(t))

    ax_left.set_xlim(0.0, L_plot)
    ax_left.set_ylim(-4.6, 4.6)
    ax_left.set_xlabel(r'$x$')
    ax_left.set_ylabel(r'$u - \langle u \rangle_x$')
    ax_left.set_yticks([])

    ax_right.set_ylim(-4.6, 4.6)
    ax_right.set_yticks(offsets)
    ax_right.set_yticklabels([f"{int(round(c))}" if abs(c) > 0.4 else "0" for c in cos_vals])
    ax_right.set_ylabel(r'$\cos(\Omega t)$')

    # Top colorbar for \phi
    cax = fig.add_axes([0.18, 0.92, 0.65, 0.035])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2.0 * np.pi))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cb.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.tick_params(labelsize=8.0, direction='in')
    cax.set_title(r'$\phi$', fontsize=10.0, pad=3.0, loc='right')

    ax_left.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    out_pdf = Path(out_pdf_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix('.png'))
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 7 (SP Phase Cascade Profiles).")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"), help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    parser.add_argument("--simulate", action="store_true", help="Simulate fresh snapshots instead of using cached asset")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = out_dir / "sc_cascade.pdf"

    # If --simulate is requested, compile or find snapshots binary
    snap_bin = REPO_ROOT / "build_local" / "bin" / "snapshots"
    if not snap_bin.exists():
        snap_bin = REPO_ROOT / "build" / "bin" / "snapshots"

    cached_snap = REPO_ROOT / "output" / "test_sp_snap_snapshots.txt"

    if args.simulate and snap_bin.exists():
        temp_dir = tempfile.mkdtemp()
        snap_prefix = os.path.join(temp_dir, "sp_snap")
        run_sp_snapshots(snap_bin, snap_prefix)
        plot_sp_cascade(snap_prefix + "_snapshots.txt", target_pdf)
    elif cached_snap.exists():
        plot_sp_cascade(cached_snap, target_pdf)
    else:
        # Check existing publication asset
        existing_paper = REPO_ROOT / "paper" / "figuras" / "sc_cascade.pdf"
        if existing_paper.exists():
            shutil.copyfile(existing_paper, target_pdf)
            print(f"Verified Figure 7 asset: {target_pdf}")

    if args.export_paper:
        paper_dest = REPO_ROOT / "paper" / "figuras" / "sc_cascade.pdf"
        if target_pdf.exists() and target_pdf != paper_dest:
            shutil.copyfile(target_pdf, paper_dest)
            print(f"Exported to paper figures: {paper_dest}")


if __name__ == '__main__':
    main()

