#!/usr/bin/env python3
r"""
plot_variance_precision.py

Generates publication-quality figure illustrating the Reviewer's objection regarding
catastrophic floating-point cancellation in the naive variance formula and the Author's
defense demonstrating the absolute stability of the two-pass algorithm.

Outputs:
  figures/fig_variance_precision_cancellation.pdf
  figures/fig_variance_precision_cancellation.png
  adversarial_review/fig_variance_precision_cancellation.png
"""

import os
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


def main():
    np.random.seed(42)
    N = 4096
    true_w2 = 5.0
    delta_u = np.random.normal(0, np.sqrt(true_w2), N)
    delta_u -= np.mean(delta_u)
    var_exact = np.mean(delta_u ** 2)

    mu_vals = np.logspace(0, 7.5, 60)
    err_naive_double = []
    err_twopass_double = []
    err_naive_single = []

    for mu in mu_vals:
        # Double precision
        u_d = delta_u + mu
        mean_d = np.mean(u_d)
        var_naive_d = np.mean(u_d ** 2) - mean_d ** 2
        var_twopass_d = np.mean((u_d - mean_d) ** 2)

        err_naive_double.append(max(abs(var_naive_d - var_exact) / var_exact, 1e-16))
        err_twopass_double.append(max(abs(var_twopass_d - var_exact) / var_exact, 1e-16))

        # Single precision simulation
        u_s = (delta_u + mu).astype(np.float32)
        mean_s = np.float32(np.mean(u_s))
        var_naive_s = np.float32(np.mean(u_s ** 2)) - mean_s ** 2
        err_naive_single.append(max(abs(float(var_naive_s) - var_exact) / var_exact, 1e-8))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})

    # =========================================================================
    # Panel (a): Relative Error vs Drift Displacement
    # =========================================================================
    ax1.plot(mu_vals, err_naive_single, color="#e377c2", linestyle=":", linewidth=1.2,
             label=r"Naive 1-Pass ($\langle u^2\rangle - \langle u\rangle^2$, Float32)")
    ax1.plot(mu_vals, err_naive_double, color="#d62728", linestyle="--", linewidth=1.4,
             label=r"Naive 1-Pass ($\langle u^2\rangle - \langle u\rangle^2$, Float64)")
    ax1.plot(mu_vals, err_twopass_double, color="#1f77b4", linestyle="-", linewidth=1.6,
             label=r"Two-Pass Algorithm ($\frac{1}{N}\sum (u-\bar{u})^2$, Float64)")

    # Mark physical drift range in simulations
    ax1.axvspan(1e2, 1e5, color="#1f77b4", alpha=0.10, label="Simulation Drift Range")
    ax1.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=0.8)
    ax1.text(3e5, 1.5, "100% Error Threshold", fontsize=7.5, color="#555555")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1, 3e7)
    ax1.set_ylim(1e-16, 1e2)
    ax1.set_xlabel(r"Center-of-Mass Drift $\langle u(t)\rangle_x$")
    ax1.set_ylabel(r"Relative Error in Roughness $|w^2 - w^2_0|/w^2_0$")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)
    ax1.text(0.04, 0.08, r"\textbf{(a)} Precision vs. Drift", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    # =========================================================================
    # Panel (b): Computed Roughness Spikes vs Stable Plateau
    # =========================================================================
    # Show computed value vs drift
    w2_naive_d = []
    w2_twopass_d = []
    for mu in mu_vals:
        u_d = delta_u + mu
        m_d = np.mean(u_d)
        w2_naive_d.append(np.mean(u_d ** 2) - m_d ** 2)
        w2_twopass_d.append(np.mean((u_d - m_d) ** 2))

    ax2.axhline(var_exact, color="black", linestyle=":", linewidth=1.2,
                label=rf"True Roughness $w^2 = {var_exact:.2f}$")
    ax2.plot(mu_vals, w2_naive_d, color="#d62728", linestyle="--", linewidth=1.3,
             label="Naive 1-Pass (Catastrophic Spikes)")
    ax2.plot(mu_vals, w2_twopass_d, color="#1f77b4", linestyle="-", linewidth=1.5,
             label="Two-Pass (Machine Floor)")

    ax2.set_xscale("log")
    ax2.set_xlim(1, 3e7)
    ax2.set_ylim(-2, 18)
    ax2.set_xlabel(r"Center-of-Mass Drift $\langle u(t)\rangle_x$")
    ax2.set_ylabel(r"Computed Roughness $w^2$")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)
    ax2.text(0.04, 0.08, r"\textbf{(b)} Algorithmic Stability", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    plt.tight_layout()

    out_pdf = "figures/fig_variance_precision_cancellation.pdf"
    out_png = "figures/fig_variance_precision_cancellation.png"
    out_adv = "adversarial_review/fig_variance_precision_cancellation.png"

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(out_adv)
    plt.close()

    print(f"Generated: {out_pdf}")
    print(f"Generated: {out_png}")
    print(f"Generated: {out_adv}")


if __name__ == "__main__":
    main()

