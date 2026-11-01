#!/usr/bin/env python3
r"""
plot_thermal_coherence.py

Generates publication-quality figure addressing the Reviewer's objection regarding
thermal spin-wave fluctuations and the Author's defense demonstrating finite-temperature
coherence lengths and condensation energies for the SP phase.

Outputs:
  figures/fig_thermal_coherence_scales.pdf
  figures/fig_thermal_coherence_scales.png
  adversarial_review/fig_thermal_coherence_scales.png
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
    kB = 1.380649e-23  # J/K
    mu0 = 4.0 * np.pi * 1e-7

    # 1. Co/Pt parameters
    A_ex_CoPt = 1.4e-11
    Ku_CoPt = 8.4e5
    Ms_CoPt = 9.1e5
    sigma_CoPt = 4.0 * np.sqrt(A_ex_CoPt * (Ku_CoPt - 0.5 * mu0 * Ms_CoPt ** 2))
    d_CoPt = 1.5e-9
    Gamma_CoPt = sigma_CoPt * d_CoPt
    Delta_CoPt = 6.6e-9
    L0_CoPt = 58.0e-9
    A0_CoPt = 0.385 * Delta_CoPt
    lambda_CoPt = (2.0 * np.pi / 0.834) * L0_CoPt

    # 2. Bi-YIG parameters
    A_ex_YIG = 3.7e-12
    Ku_YIG = 2.0e4
    Delta_YIG = np.sqrt(A_ex_YIG / Ku_YIG)
    sigma_YIG = 4.0 * np.sqrt(A_ex_YIG * Ku_YIG)
    d_YIG = 50e-9
    Gamma_YIG = sigma_YIG * d_YIG
    A0_YIG = 0.385 * Delta_YIG
    L0_YIG = 250e-9
    lambda_YIG = (2.0 * np.pi / 0.834) * L0_YIG

    T_vals = np.logspace(np.log10(2.0), np.log10(350.0), 100)

    # Coherence length xi_T = Gamma * A0^2 / (k_B * T)
    xi_CoPt = (Gamma_CoPt * A0_CoPt ** 2) / (kB * T_vals)
    xi_YIG = (Gamma_YIG * A0_YIG ** 2) / (kB * T_vals)

    # Condensation energy per wavelength E_cond = (1/2) * Gamma * (2*pi/lambda)^2 * A0^2 * lambda
    # E_cond = 2 * pi^2 * Gamma * A0^2 / lambda
    E_cond_CoPt = (2.0 * np.pi ** 2 * Gamma_CoPt * A0_CoPt ** 2) / lambda_CoPt
    E_cond_YIG = (2.0 * np.pi ** 2 * Gamma_YIG * A0_YIG ** 2) / lambda_YIG

    ratio_E_CoPt = E_cond_CoPt / (kB * T_vals)
    ratio_E_YIG = E_cond_YIG / (kB * T_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})

    # =========================================================================
    # Panel (a): Thermal Phase Coherence Length xi_T vs Temperature
    # =========================================================================
    ax1.plot(T_vals, xi_YIG * 1e6, color="#2ca02c", linestyle="-", linewidth=1.6,
             label=r"Bi-YIG Garnet ($d=50$ nm)")
    ax1.axhline(lambda_YIG * 1e6, color="#2ca02c", linestyle=":", linewidth=1.1,
                label=rf"Bi-YIG Wavelength $\lambda_\mathrm{{SP}} \approx {lambda_YIG*1e6:.2f}\,\mu$m")

    ax1.plot(T_vals, xi_CoPt * 1e6, color="#1f77b4", linestyle="-", linewidth=1.6,
             label=r"Co/Pt Multilayer ($d=1.5$ nm)")
    ax1.axhline(lambda_CoPt * 1e6, color="#1f77b4", linestyle="--", linewidth=1.1,
                label=rf"Co/Pt Wavelength $\lambda_\mathrm{{SP}} \approx {lambda_CoPt*1e9:.0f}$ nm")

    # Mark cryogenic points
    ax1.axvline(4.2, color="#7f7f7f", linestyle=":", linewidth=0.8)
    ax1.text(4.4, 0.03, "L-He (4.2 K)", fontsize=7.5, color="#555555", rotation=90)
    ax1.axvline(77.0, color="#7f7f7f", linestyle=":", linewidth=0.8)
    ax1.text(80.0, 0.03, "L-N$_2$ (77 K)", fontsize=7.5, color="#555555", rotation=90)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(2, 350)
    ax1.set_ylim(0.01, 100)
    ax1.set_xlabel(r"Temperature $T$ (K)")
    ax1.set_ylabel(r"Thermal Coherence Length $\xi_T$ ($\mu$m)")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.5)
    ax1.text(0.04, 0.08, r"\textbf{(a)} Spatial Coherence $\xi_T(T)$", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    # =========================================================================
    # Panel (b): Condensation Energy vs k_B T
    # =========================================================================
    ax2.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.0, label=r"Thermal Energy Floor $k_\mathrm{B} T$")
    ax2.plot(T_vals, ratio_E_YIG, color="#2ca02c", linestyle="-", linewidth=1.6,
             label=r"Bi-YIG Garnet ($E_\mathrm{cond}/k_\mathrm{B}T$)")
    ax2.plot(T_vals, ratio_E_CoPt, color="#1f77b4", linestyle="-", linewidth=1.6,
             label=r"Co/Pt Multilayer ($E_\mathrm{cond}/k_\mathrm{B}T$)")

    ax2.fill_between(T_vals, 1.0, 500, color="#2ca02c", alpha=0.08)
    ax2.text(10, 2.0, r"Thermally Stable Pattern" "\n" r"($E_\mathrm{cond} > k_\mathrm{B}T$)",
             fontsize=8, color="#1e7b1e", fontweight="bold")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(2, 350)
    ax2.set_ylim(0.05, 500)
    ax2.set_xlabel(r"Temperature $T$ (K)")
    ax2.set_ylabel(r"Barrier Ratio $E_\mathrm{cond} / (k_\mathrm{B} T)$")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)
    ax2.text(0.04, 0.08, r"\textbf{(b)} Pattern Condensation Energy", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    plt.tight_layout()

    out_pdf = "figures/fig_thermal_coherence_scales.pdf"
    out_png = "figures/fig_thermal_coherence_scales.png"
    out_adv = "adversarial_review/fig_thermal_coherence_scales.png"

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(out_adv)
    plt.close()

    print(f"Generated: {out_pdf}")
    print(f"Generated: {out_png}")
    print(f"Generated: {out_adv}")


if __name__ == "__main__":
    main()
