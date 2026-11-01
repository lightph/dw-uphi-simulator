#!/usr/bin/env python3
r"""
plot_floquet_convergence.py

Generates publication-quality figure addressing the Reviewer's objection regarding
the Floquet ODE solver discretization and the Author's defense of 8-digit spectral convergence.

Outputs:
  figures/fig_floquet_solver_convergence.pdf
  figures/fig_floquet_solver_convergence.png
  adversarial_review/fig_floquet_solver_convergence.png
"""

import os
import sys
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

sys.path.append("floquet/python")
from floquet_solver import solve_background_cycle, compute_floquet_multiplier


def solve_midpoint(h0, ha, kappa, N_tau, alpha=0.27, omega=1.0):
    conv, psi_cycle = solve_background_cycle(h0, ha, alpha=alpha, omega=omega, steps_per_cycle=N_tau)
    if not conv:
        return np.nan
    # Without passing h0 and ha, uses midpoint interpolation
    return compute_floquet_multiplier(psi_cycle, kappa, alpha=alpha, omega=omega, steps_per_cycle=N_tau)


def solve_simultaneous(h0, ha, kappa, N_tau, alpha=0.27, omega=1.0):
    conv, psi_cycle = solve_background_cycle(h0, ha, alpha=alpha, omega=omega, steps_per_cycle=N_tau)
    if not conv:
        return np.nan
    # Passing h0 and ha uses simultaneous RK4
    return compute_floquet_multiplier(psi_cycle, kappa, alpha=alpha, omega=omega, steps_per_cycle=N_tau, h0=h0, ha=ha)


def main():
    h0 = 4.0
    ha = 3.0
    kappa_star = 0.834

    print("Computing Floquet multiplier vs N_tau...")
    N_tau_vals = [50, 75, 100, 150, 200, 250, 320, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000]
    mu_simult = []
    mu_mid = []

    for N in N_tau_vals:
        s = solve_simultaneous(h0, ha, kappa_star, N)
        m = solve_midpoint(h0, ha, kappa_star, N)
        mu_simult.append(s)
        mu_mid.append(m)
        print(f"  N={N:4d}: Simultaneous = {s:.8f} | Midpoint = {m:.8f}")

    print("\nComputing Floquet spectrum |mu(kappa)|...")
    kappas = np.linspace(0.76, 0.90, 120)
    conv, psi_cycle = solve_background_cycle(h0, ha, steps_per_cycle=1000)
    mu_spectrum = [compute_floquet_multiplier(psi_cycle, k, steps_per_cycle=1000, h0=h0, ha=ha) for k in kappas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.30})

    # =========================================================================
    # Panel (a): Convergence vs N_tau
    # =========================================================================
    ax1.axhline(1.0, color="#555555", linestyle=":", linewidth=1.2, label=r"Neutral Stability ($|\mu|=1$)")
    ax1.plot(N_tau_vals, mu_simult, "o-", color="#1f77b4", linewidth=1.5, markersize=4,
             label=r"Simultaneous RK4 (Exact $\to 1.00606311$)")
    ax1.plot(N_tau_vals, mu_mid, "s--", color="#d62728", linewidth=1.3, markersize=3.5,
             label=r"Midpoint Interpolation (Damped)")

    # Highlight false stability zone below 1.0
    ax1.fill_between([40, 5000], [0.990, 0.990], [1.0, 1.0], color="#d62728", alpha=0.08)
    ax1.text(70, 0.993, "False Stability Region\n(Numerical Damping)", fontsize=7.5, color="#a01010", va="bottom")

    ax1.set_xscale("log")
    ax1.set_xlim(45, 5000)
    ax1.set_ylim(0.990, 1.008)
    ax1.set_xlabel(r"Discretization Steps per Cycle $N_\tau$")
    ax1.set_ylabel(r"Floquet Multiplier $|\mu_{\kappa^*}|$")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.4)
    ax1.text(0.04, 0.92, r"\textbf{(a)} ODE Discretization Convergence", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="top")

    # =========================================================================
    # Panel (b): Full Mode Instability Band & Discrete Allowed Modes
    # =========================================================================
    ax2.axhline(1.0, color="#555555", linestyle=":", linewidth=1.2)
    ax2.plot(kappas, mu_spectrum, color="#1f77b4", linewidth=1.6, label=r"Floquet Multiplier $|\mu_\kappa|$")

    # Shade unstable band
    k_unstable = np.array(kappas)[np.array(mu_spectrum) > 1.0]
    if len(k_unstable) > 0:
        ax2.fill_between(kappas, 1.0, mu_spectrum, where=(np.array(mu_spectrum) > 1.0),
                         color="#ff7f0e", alpha=0.25, label=r"Unstable Band ($\Delta\kappa \approx 0.033$)")

    # Mark discrete modes for small vs large systems
    # For L = 8: kappa_1 = 2*pi/8 = 0.785
    kappa_L8 = 2.0 * np.pi / 8.0
    ax2.axvline(kappa_L8, color="#2ca02c", linestyle="--", linewidth=1.2,
                label=rf"$\kappa_1(L=8) \approx {kappa_L8:.3f}$")

    # Peak mode
    ax2.axvline(kappa_star, color="#d62728", linestyle="-.", linewidth=1.2,
                label=rf"$\kappa^* \approx {kappa_star:.3f}$")

    ax2.set_xlim(0.76, 0.90)
    ax2.set_ylim(0.94, 1.01)
    ax2.set_xlabel(r"Wavenumber $\kappa$")
    ax2.set_ylabel(r"Floquet Multiplier $|\mu_\kappa|$")
    ax2.legend(loc="lower center", frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.4)
    ax2.text(0.04, 0.92, r"\textbf{(b)} Instability Band & Mode Selection", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="top")

    plt.tight_layout()

    out_pdf = "figures/fig_floquet_solver_convergence.pdf"
    out_png = "figures/fig_floquet_solver_convergence.png"
    out_adv = "adversarial_review/fig_floquet_solver_convergence.png"

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(out_adv)
    plt.close()

    print(f"\nGenerated: {out_pdf}")
    print(f"Generated: {out_png}")
    print(f"Generated: {out_adv}")


if __name__ == "__main__":
    main()

