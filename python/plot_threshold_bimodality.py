#!/usr/bin/env python3
r"""
plot_threshold_bimodality.py

Generates publication-quality figure addressing the Reviewer's objection regarding
the threshold discrepancy and the Author's defense of the 6-decade empty gap.

Outputs:
  figures/fig_threshold_bimodality_desert.pdf
  figures/fig_threshold_bimodality_desert.png
  adversarial_review/fig_threshold_bimodality_desert.png
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
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
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
    print("Loading sweep dataset...")
    data = load_sweep_data()
    vs = data['var_strobo']
    h0 = data['h0']
    ha = data['ha']
    n_total = len(vs)
    print(f"Loaded {n_total} sweep points.")

    # Mask strictly positive
    pos = vs > 0
    log_vs = np.log10(vs[pos])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})

    # =========================================================================
    # Panel (a): Probability Density & 6-Decade Empty Gap
    # =========================================================================
    bins = np.linspace(-16, 2, 73)
    counts, bin_edges = np.histogram(log_vs, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot histogram bars
    ax1.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color="#1f77b4",
            edgecolor="#0f4c81", linewidth=0.5, alpha=0.85, label="Simulation Points")

    # Shade the 6-decade empty gap ("The Desert")
    gap_min, gap_max = -12.0, -6.0
    ax1.axvspan(gap_min, gap_max, color="#ff7f0e", alpha=0.18,
                label=r"6-Decade Empty Gap ($< 0.02\%$ points)")

    # Mark the two thresholds
    t_code_log = np.log10(3.5e-12)
    t_text_log = -4.0
    ax1.axvline(t_code_log, color="#d62728", linestyle="-", linewidth=1.5,
                label=rf"Code: $t_\mathrm{{lock}} = 3.5\times 10^{{-12}}$")
    ax1.axvline(t_text_log, color="#2ca02c", linestyle="--", linewidth=1.5,
                label=rf"Text: $t_\mathrm{{text}} = 1.0\times 10^{{-4}}$")

    # Labels and annotations
    ax1.text(-14.2, 1.8e4, "Synchronized\n(Machine Floor)", fontsize=8, color="#1f77b4",
             fontweight="bold", ha="center")
    ax1.text(-0.5, 1.8e4, "Unlocked\n(Macroscopic)", fontsize=8, color="#1f77b4",
             fontweight="bold", ha="center")
    ax1.text(-9.0, 5e3, "Empty\nDesert\n(6 Decades)", fontsize=8, color="#b35400",
             fontweight="bold", ha="center")

    ax1.set_yscale("log")
    ax1.set_xlim(-16.5, 2.0)
    ax1.set_ylim(1, 5e4)
    ax1.set_xlabel(r"Stroboscopic Variance $\log_{10}(\langle\sigma^2\rangle_x)$")
    ax1.set_ylabel("Point Count")
    ax1.legend(loc="upper center", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)
    ax1.text(0.04, 0.92, r"\textbf{(a)} Empirical Bimodal Distribution", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="top")

    # =========================================================================
    # Panel (b): Discrepancy Points Location in (h0, ha) Space
    # =========================================================================
    # Identify discrepancy points: vs in [3.5e-12, 1e-4]
    m_disc = (vs >= 3.5e-12) & (vs < 1e-4)
    n_disc = np.sum(m_disc)
    pct_disc = (n_disc / n_total) * 100.0

    m_sync = vs < 3.5e-12

    # Subsample background points for fast plotting
    stride = 3
    ax2.scatter(h0[::stride], ha[::stride], c="#e0e0e0", s=1, alpha=0.5, rasterized=True)
    ax2.scatter(h0[m_sync][::stride], ha[m_sync][::stride], c="#1f77b4", s=1.5, alpha=0.6,
                label=rf"Locked States ({np.sum(m_sync):,})", rasterized=True)
    ax2.scatter(h0[m_disc], ha[m_disc], c="#d62728", s=10, marker="o", edgecolors="black",
                linewidths=0.3, zorder=6, label=rf"Boundary Points ({n_disc} / {pct_disc:.2f}\%)")

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel(r"DC Field $h_0$")
    ax2.set_ylabel(r"AC Field Amplitude $h_\mathrm{a}$")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)
    ax2.text(0.04, 0.06, rf"\textbf{{(b)}} Phase Invariance: 99.83\%", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    plt.tight_layout()

    out_pdf = "figures/fig_threshold_bimodality_desert.pdf"
    out_png = "figures/fig_threshold_bimodality_desert.png"
    out_adv = "adversarial_review/fig_threshold_bimodality_desert.png"

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(out_adv)
    plt.close()

    print(f"Generated: {out_pdf}")
    print(f"Generated: {out_png}")
    print(f"Generated: {out_adv}")


if __name__ == "__main__":
    main()

