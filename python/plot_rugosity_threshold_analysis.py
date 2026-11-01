#!/usr/bin/env python3
r"""
plot_rugosity_threshold_analysis.py

Performs a comprehensive statistical audit and threshold selection analysis for the
interface roughness (rugosity) \langle w^2 \rangle_t across all 321,201 points of the
dynamical phase diagram in (h0, ha) space.

Outputs:
  figures/fig_rugosity_threshold_bimodality.pdf
  figures/fig_rugosity_threshold_bimodality.png
  adversarial_review/fig_rugosity_threshold_bimodality.png
"""

import os
import glob
import tempfile
import numpy as np

if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.8,
    "ytick.major.size": 3.8,
    "xtick.top": True,
    "ytick.right": True,
})


def load_sweep_data(data_dir="output/sweep_full_973970"):
    pattern = os.path.join(data_dir, "results_h0_*.txt")
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        with open(f, 'r') as fp:
            fp.readline()
            for line in fp:
                parts = line.split()
                if len(parts) >= 6:
                    records.append((
                        float(parts[0]), float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]), float(parts[5])
                    ))
    dtype = [('h0', float), ('ha', float), ('var_strobo', float), ('rugosity', float), ('v_u', float), ('v_phi', float)]
    return np.array(records, dtype=dtype)


def main():
    print("Loading sweep dataset for rugosity analysis...")
    data = load_sweep_data()
    h0 = data['h0']
    ha = data['ha']
    vs = data['var_strobo']
    rug = data['rugosity']
    vphi = data['v_phi']
    n_total = len(rug)
    print(f"Loaded {n_total} points.")

    t_lock = 3.5e-12
    t_vphi = 1.0e-5
    locked = vs < t_lock
    precessing = np.abs(vphi) >= t_vphi

    # Mask positive rugosity
    pos = rug > 0
    log_rug = np.log10(rug[pos])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"wspace": 0.32})

    # =========================================================================
    # Panel (a): Overall Bimodal Distribution & The 2-Decade Gap
    # =========================================================================
    bins = np.linspace(-12, 3, 61)
    counts, bin_edges = np.histogram(log_rug, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax1.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color="#2ca02c",
            edgecolor="#1b611b", linewidth=0.5, alpha=0.85, label="Simulation Points")

    gap_min, gap_max = -6.0, -4.0
    ax1.axvspan(gap_min, gap_max, color="#ff7f0e", alpha=0.22,
                label=r"Gap: $10^{-6}$--$10^{-4}$ (0.024\%)")

    ax1.axvline(-6.0, color="#d62728", linestyle="--", linewidth=1.3,
                label=r"$t_\mathrm{text} = 1.0\times 10^{-6}$")
    ax1.axvline(-4.0, color="#1f77b4", linestyle="-", linewidth=1.5,
                label=r"$t_\mathrm{rug} = 1.0\times 10^{-4}$ (Optimal)")

    ax1.text(-8.5, 1.8e4, "Flat States\n(Discretization Floor)", fontsize=7.5, color="#1b611b",
             fontweight="bold", ha="center")
    ax1.text(1.0, 4.0e4, "Corrugated / Rough\n(Macroscopic)", fontsize=7.5, color="#1b611b",
             fontweight="bold", ha="center")
    ax1.text(-5.0, 3e2, "Gap: 78 pts\n(0.024%)", fontsize=7.5, color="#b35400",
             fontweight="bold", ha="center")

    ax1.set_yscale("log")
    ax1.set_xlim(-12.0, 3.2)
    ax1.set_ylim(1, 1.2e5)
    ax1.set_xlabel(r"Roughness $\log_{10}(\langle w^2\rangle_t)$")
    ax1.set_ylabel("Point Count")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.2)
    ax1.text(0.04, 0.94, r"\textbf{(a)} Empirical Bimodal Distribution", transform=ax1.transAxes,
             fontsize=8.5, fontweight="bold", va="top")

    # =========================================================================
    # Panel (b): Synchronized Phase Separation (SF vs SP 3-Decade Void)
    # =========================================================================
    sync_mask = locked & precessing
    rug_sf = rug[sync_mask & (rug < 1e-4)]
    rug_sp = rug[sync_mask & (rug >= 1e-4)]

    bins_sync = np.linspace(-12, 0.5, 51)
    counts_sf, _ = np.histogram(np.log10(rug_sf), bins=bins_sync)
    counts_sp, _ = np.histogram(np.log10(rug_sp), bins=bins_sync)
    bc_sync = 0.5 * (bins_sync[:-1] + bins_sync[1:])

    ax2.bar(bc_sync, counts_sf, width=bins_sync[1] - bins_sync[0], color="#1f77b4",
            alpha=0.85, label=rf"SF (Flat, {len(rug_sf):,})")
    ax2.bar(bc_sync, counts_sp, width=bins_sync[1] - bins_sync[0], color="#d62728",
            alpha=0.85, label=rf"SP (Periodic, {len(rug_sp):,})")

    # True SF/SP gap: from max(SF)=4.14e-6 to min(SP)=2.36e-3
    gap_sf_max = np.log10(np.max(rug_sf))
    gap_sp_min = np.log10(np.min(rug_sp))
    ax2.axvspan(gap_sf_max, gap_sp_min, color="#9467bd", alpha=0.18,
                label=r"SF--SP Void (0 points)")

    ax2.axvline(-4.0, color="#000000", linestyle="-", linewidth=1.4,
                label=r"$t_\mathrm{rug} = 10^{-4}$ (Midpoint)")

    ax2.text(-7.8, 1.2e3, "SF Max:\n" + r"$4.1\times 10^{-6}$", fontsize=7.2, color="#1f77b4",
             fontweight="bold", ha="center")
    ax2.text(-1.0, 1.2e3, "SP Min:\n" + r"$2.4\times 10^{-3}$", fontsize=7.2, color="#d62728",
             fontweight="bold", ha="center")
    ax2.text(-4.0, 8.0, "3-Decade Void\n(100% Empty)", fontsize=7.2, color="#6b3ba7",
             fontweight="bold", ha="center")

    ax2.set_yscale("log")
    ax2.set_xlim(-12.0, 0.5)
    ax2.set_ylim(0.5, 1.5e4)
    ax2.set_xlabel(r"Roughness $\log_{10}(\langle w^2\rangle_t)$")
    ax2.set_ylabel("Point Count (Synchronized)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.0)
    ax2.text(0.04, 0.94, r"\textbf{(b)} SF vs SP Separation", transform=ax2.transAxes,
             fontsize=8.5, fontweight="bold", va="top")

    # =========================================================================
    # Panel (c): Parameter Space Map & Boundary Invariance
    # =========================================================================
    m_disc = (rug >= 1e-6) & (rug < 1e-4)
    n_disc = np.sum(m_disc)
    pct_disc = (n_disc / n_total) * 100.0
    invariance_pct = 100.0 - pct_disc

    m_sp = sync_mask & (rug >= 1e-4)
    m_sf = sync_mask & (rug < 1e-4)

    stride = 4
    # Background points
    ax3.scatter(h0[::stride], ha[::stride], c="#ebebeb", s=1, alpha=0.4, rasterized=True)
    # SF points
    ax3.scatter(h0[m_sf][::stride], ha[m_sf][::stride], c="#aec7e8", s=1.5, alpha=0.5,
                label=rf"SF ({np.sum(m_sf):,})", rasterized=True)
    # SP points
    ax3.scatter(h0[m_sp], ha[m_sp], c="#d62728", s=6, marker="s", edgecolors="none",
                label=rf"SP ({np.sum(m_sp):,})", zorder=4)
    # Discrepancy points
    ax3.scatter(h0[m_disc], ha[m_disc], c="#ff7f0e", s=14, marker="o", edgecolors="black",
                linewidths=0.4, zorder=6, label=rf"Boundary Discrepancy ({n_disc} pts, {pct_disc:.3f}\%)")

    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_xlabel(r"DC Field $h_0$")
    ax3.set_ylabel(r"AC Field Amplitude $h_\mathrm{a}$")
    ax3.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.0)
    ax3.text(0.04, 0.06, rf"\textbf{{(c)}} Phase Invariance: {invariance_pct:.3f}\%", transform=ax3.transAxes,
             fontsize=8.5, fontweight="bold", va="bottom")

    plt.tight_layout()

    out_pdf = "figures/fig_rugosity_threshold_bimodality.pdf"
    out_png = "figures/fig_rugosity_threshold_bimodality.png"
    out_adv = "adversarial_review/fig_rugosity_threshold_bimodality.png"

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(out_adv)
    plt.close()

    print(f"Generated: {out_pdf}")
    print(f"Generated: {out_png}")
    print(f"Generated: {out_adv}")


if __name__ == "__main__":
    main()

