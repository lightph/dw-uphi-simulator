#!/usr/bin/env python3
r"""
plot_roughness_saturation.py

Generates publication-quality figure for the steady-state interface roughness
saturation scaling w^2(L) vs system size L, confirming 1D Edwards-Wilkinson
Family-Vicsek dynamic scaling (zeta = 1/2, z = 2).

Outputs:
  figures/fig_roughness_saturation_scaling.pdf
  figures/fig_roughness_saturation_scaling.png
  adversarial_review/fig_roughness_saturation_scaling.png
"""

import os
import glob
import re
import numpy as np
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


def load_saturation_data():
    # 1. Load GPU saturation scaling sweep
    gpu_dirs = sorted(glob.glob("output/saturation_scaling_gpu/main_NC_sat_*"))
    gpu_results = []
    transient_curves = {}

    for d in gpu_dirs:
        m = re.search(r"main_NC_sat_([0-9.]+)_5\.0_3\.0_1\.0", d)
        if m:
            L = float(m.group(1))
            sum_f = os.path.join(d, "time_data_final_summary.txt")
            if os.path.exists(sum_f):
                with open(sum_f) as fp:
                    lines = [l.strip() for l in fp if l.strip() and not l.startswith("batch")]
                if lines:
                    parts = lines[0].split()
                    var_u = float(parts[2])
                    gpu_results.append((L, var_u))

            trans_f = os.path.join(d, "time_data_transient_metrics.txt")
            if os.path.exists(trans_f):
                try:
                    data = np.loadtxt(trans_f, skiprows=1)
                    if data.ndim == 2 and data.shape[0] > 2:
                        t = data[:, 2]
                        w2 = data[:, 3]
                        # Filter strictly positive values
                        valid = (w2 > 1e-10) & (t > 0)
                        if np.sum(valid) > 2:
                            transient_curves[L] = (t[valid], w2[valid])
                except Exception:
                    pass

    gpu_results.sort(key=lambda x: x[0])
    return gpu_results, transient_curves


def main():
    gpu_results, transient_curves = load_saturation_data()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("adversarial_review", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})

    # =========================================================================
    # Panel (a): Steady-State Roughness Saturation w_sat^2 vs L
    # =========================================================================
    L_all = np.array([r[0] for r in gpu_results])
    w2_all = np.array([r[1] for r in gpu_results])

    # Separate flat (small L) vs rough (L >= 32)
    m_flat = L_all < 32.0
    m_rough = L_all >= 32.0

    L_rough = L_all[m_rough]
    w2_rough = w2_all[m_rough]

    # Power law fit for rough regime
    logL = np.log(L_rough)
    logw2 = np.log(w2_rough)
    slope, intercept = np.polyfit(logL, logw2, 1)

    # Reference theoretical slope (EW: 2*zeta = 1.0)
    L_fit = np.logspace(np.log10(28), np.log10(1200), 100)
    w2_fit = np.exp(intercept) * (L_fit ** slope)
    # Theory line matching w2 at L=128
    ref_idx = np.where(L_rough == 128.0)[0]
    ref_w2 = w2_rough[ref_idx[0]] if len(ref_idx) > 0 else 4.67
    w2_theory = ref_w2 * (L_fit / 128.0) ** 1.0

    ax1.plot(L_fit, w2_theory, color="#7f7f7f", linestyle="--", linewidth=1.2,
             label=r"EW Theory: $\propto L^{1.00}$ ($2\zeta=1$)")
    ax1.plot(L_fit, w2_fit, color="#d62728", linestyle="-", linewidth=1.5,
             label=rf"Power-law Fit: $\propto L^{{{slope:.2f}}}$ ($2\zeta={slope:.2f}$)")

    # Plot rough data points
    ax1.scatter(L_rough, w2_rough, color="#1f77b4", s=35, zorder=5,
                edgecolors="black", linewidths=0.6, label="Simulation (NR Phase)")

    # Plot flat points at artificial floor for display
    floor_val = 2e-2
    ax1.scatter(L_all[m_flat], np.full(np.sum(m_flat), floor_val),
                color="#2ca02c", marker="v", s=35, zorder=5,
                edgecolors="black", linewidths=0.6, label=r"Flat State ($w^2 \sim 10^{-23}$)")

    ax1.axvspan(6, 24, color="#2ca02c", alpha=0.08, label="Discrete Mode Gap")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(6, 1500)
    ax1.set_ylim(1e-2, 35)
    ax1.set_xlabel(r"Domain Wall Length $L/L_0$")
    ax1.set_ylabel(r"Saturated Roughness $w_\mathrm{sat}^2$")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.5)
    ax1.text(0.04, 0.08, r"\textbf{(a)} Saturation Scaling", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    # =========================================================================
    # Panel (b): Dynamic Growth w^2(t) across System Sizes
    # =========================================================================
    sizes_to_plot = [32, 64, 128, 256, 512, 1024]
    cmap = matplotlib.colormaps.get_cmap("viridis")
    norm = matplotlib.colors.LogNorm(vmin=32, vmax=1024)

    for L in sizes_to_plot:
        if L in transient_curves:
            t, w2 = transient_curves[L]
            color = cmap(norm(L))
            ax2.plot(t, w2, color=color, linewidth=1.3, label=rf"$L={int(L)}$")

    # Growth power-law guide t^(1/2) (z=2, 2*zeta=1 -> 2*zeta/z = 1/2)
    t_guide = np.logspace(2.0, 4.5, 50)
    w2_guide = 0.04 * (t_guide ** 0.5)
    ax2.plot(t_guide, w2_guide, color="black", linestyle=":", linewidth=1.5,
             label=r"$\propto t^{1/2}$ ($2\zeta/z = 1/2$)")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e1, 1.2e6)
    ax2.set_ylim(1e-4, 50)
    ax2.set_xlabel(r"Time $t/T_0$")
    ax2.set_ylabel(r"Interface Roughness $w^2(t)$")
    ax2.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.2, columnspacing=0.8)
    ax2.text(0.04, 0.08, r"\textbf{(b)} Transient Saturation Dynamics", transform=ax2.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    plt.tight_layout()

    out_pdf = "figures/fig_roughness_saturation_scaling.pdf"
    out_png = "figures/fig_roughness_saturation_scaling.png"
    out_adv = "adversarial_review/fig_roughness_saturation_scaling.png"

    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(out_adv)
    plt.close()

    print(f"Generated: {out_pdf}")
    print(f"Generated: {out_png}")
    print(f"Generated: {out_adv}")


if __name__ == "__main__":
    main()

