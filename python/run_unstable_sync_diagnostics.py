#!/usr/bin/env python3
"""
run_unstable_sync_diagnostics.py

Investigates the physical mechanism of the synchronized phases that lie outside
the stability region (max_mu > 1.0) but inside the synchrony boundary (converged == True).

Performs:
  1. High-precision numerical time evolution for representative cases:
     - Case 1: Core Instability Peak (h0 = 3.70, ha = 3.0)
     - Case 2: Instability Onset (h0 = 3.50, ha = 3.0)
     - Case 3: Low-Amplitude Tongue (h0 = 3.80, ha = 2.10)
     - Case 4: Stable Synchronous Control Baseline (h0 = 4.50, ha = 3.0)
  2. Rugosity growth and saturation curves w^2(t).
  3. Spatio-temporal Hovmoller heatmaps (x vs t) revealing wave propagation.
  4. Instantaneous wave profiles u(x, t) and internal magnetization angles phi(x, t).
  5. Spatial power spectra |u(k)|^2 compared directly with Floquet-predicted critical wavenumbers k_mu.
  6. Smooth MP4 and GIF animated movies of the traveling pattern.
  7. Cross-case comparative summary figure.
"""

import os
import sys
import time
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path

# Configure writable matplotlib cache
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# PRB Typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 4.5,
    'ytick.major.size': 4.5,
    'xtick.minor.size': 2.5,
    'ytick.minor.size': 2.5,
    'legend.fontsize': 9.5,
    'legend.frameon': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ==============================================================================
# CONFIGURATION & EXECUTABLE
# ==============================================================================
EXECUTABLE = "./build_local/bin/movies"
OUT_BASE = "output/diagnostics_unstable_sync"
os.makedirs(OUT_BASE, exist_ok=True)

# System physical constants - exactly matching sweep parameters: L=8, N=16
SIZE = 16
L = 8.0
DX = L / SIZE
ALPHA = 0.27
OMEGA = 1.0
DT = 0.0196349540849362  # 2*pi / (320*omega)
T_DRIVE = 2.0 * np.pi / OMEGA  # Period of drive (~6.2831853)
T_LOCK = 4.0 * np.pi / OMEGA   # Period of synchronous cycle (~12.56637)
EPS = 0.05
SEED = 42

# Time spans: at least 10,000 driving periods
N_PERIODS = 10000
T_MAX = N_PERIODS * T_DRIVE  # 62831.85307 (~10,000 driving periods)
MOVIE_START = T_MAX - 4.0 * T_LOCK  # Final 4 cycles at the end of 10,000 periods
MOVIE_END = T_MAX
MOVIE_RATE = 5  # Record every 5 steps (~0.1 time units)

CASES = [
    {
        'id': 'case1_core_pattern',
        'title': 'Case 1: Core Instability Peak',
        'h0': 3.70, 'ha': 3.00,
        'floquet_mu': 1.074, 'floquet_k': 0.822,
        'desc': 'Strong parametric corrugation inside main Arnold tongue'
    },
    {
        'id': 'case2_onset_boundary',
        'title': 'Case 2: Instability Onset',
        'h0': 3.50, 'ha': 3.00,
        'floquet_mu': 1.261, 'floquet_k': 0.877,
        'desc': 'Near the onset boundary of parametric instability'
    },
    {
        'id': 'case3_low_ha_tongue',
        'title': 'Case 3: Low-Amplitude Tongue',
        'h0': 3.80, 'ha': 2.10,
        'floquet_mu': 1.226, 'floquet_k': 0.853,
        'desc': 'Strong instability at lower AC drive amplitude'
    },
    {
        'id': 'case_baseline_stable',
        'title': 'Baseline: Stable Synchronous',
        'h0': 3.15, 'ha': 7.30,
        'floquet_mu': 0.9995, 'floquet_k': 0.0,
        'desc': 'Strictly stable flat domain wall control (mu <= 1)'
    },
    {
        'id': 'case5_flat_unstable',
        'title': 'Case 5: Flat Sync (Outside Stab. Bnd.)',
        'h0': 3.45, 'ha': 3.675,
        'floquet_mu': 1.0863, 'floquet_k': 0.8529,
        'desc': 'Flat synchronized state outside stability boundary (discrete mode gap)'
    }
]

# ==============================================================================
# BINARY SNAPSHOT READER
# ==============================================================================
def read_binary_snapshots(snap_file, size):
    """Parses binary snapshot file written by movies executable."""
    record_bytes = 8 + 8 + size * 16  # uint64 batch, float64 time, size*complex128
    times = []
    u_profiles = []
    phi_profiles = []

    with open(snap_file, 'rb') as f:
        while True:
            chunk = f.read(record_bytes)
            if len(chunk) < record_bytes:
                break
            b_id = np.frombuffer(chunk[:8], dtype=np.uint64)[0]
            t = np.frombuffer(chunk[8:16], dtype=np.float64)[0]
            c_arr = np.frombuffer(chunk[16:], dtype=np.complex128)

            times.append(t)
            u_profiles.append(c_arr.real.copy())
            phi_profiles.append(-c_arr.imag.copy())

    return np.array(times), np.array(u_profiles), np.array(phi_profiles)

# ==============================================================================
# RUN SIMULATION
# ==============================================================================
def run_simulation(case, out_dir, force=None):
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "run")
    snap_file = prefix + "_snapshots.bin"
    metrics_file = prefix + "_metrics.txt"

    if force is None:
        force = '--force' in sys.argv
    if not force and os.path.exists(snap_file) and os.path.exists(metrics_file):
        rec_size = 16 + SIZE * 16
        if os.path.getsize(snap_file) % rec_size == 0 and os.path.getsize(snap_file) > 0:
            try:
                with open(metrics_file, 'r') as mf:
                    lines = [ln.strip() for ln in mf if ln.strip()]
                    if len(lines) > 1:
                        last_line = lines[-1].split()
                        t_last = float(last_line[2])
                        if t_last >= T_MAX - 10.0:
                            print(f"[{case['id']}] Found existing matching run data up to t={t_last:.1f} (~{t_last/T_DRIVE:.0f} periods) in {out_dir}. Loading...")
                            return snap_file, metrics_file
            except Exception:
                pass

    print(f"[{case['id']}] Running simulation (h0={case['h0']}, ha={case['ha']}) up to t={T_MAX:.1f} ({N_PERIODS} driving periods)...")
    cmd = [
        EXECUTABLE,
        str(SIZE), "1", str(L), str(ALPHA),
        str(case['h0']), str(case['ha']), str(OMEGA), str(DT),
        f"{T_MAX:.5f}", f"{T_LOCK:.5f}", f"{MOVIE_START:.5f}", f"{MOVIE_END:.5f}", str(MOVIE_RATE),
        prefix, str(EPS), str(SEED), "cpu"
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running simulation for {case['id']}:\n{res.stderr}")
        sys.exit(1)
    print(f"[{case['id']}] Completed in {time.time() - t0:.2f}s.")
    return snap_file, metrics_file

# ==============================================================================
# GENERATE DIAGNOSTICS FOR A CASE
# ==============================================================================
def process_case(case):
    case_dir = os.path.join(OUT_BASE, case['id'])
    os.makedirs(case_dir, exist_ok=True)

    snap_file, metrics_file = run_simulation(case, case_dir, force=False)

    # Load metrics time series
    # format: batch step time avg_var_u avg_u_dot avg_phi_dot avg_entropy
    metrics_data = np.loadtxt(metrics_file, skiprows=1)
    t_m = metrics_data[:, 2]
    var_u_m = metrics_data[:, 3]
    u_dot_m = metrics_data[:, 4]
    phi_dot_m = metrics_data[:, 5]

    # Load snapshots
    snap_t, u_prof, phi_prof = read_binary_snapshots(snap_file, SIZE)
    x = np.linspace(0, L, SIZE, endpoint=False)

    # -------------------------------------------------------------------------
    # 1. Plot Rugosity & Velocity Evolution
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.8), sharex=True)

    p_m = t_m / T_DRIVE  # Normalized to driving periods T_drive = 2*pi/omega
    min_rug = max(1e-25, float(np.min(var_u_m)))
    ax1.semilogy(p_m, np.maximum(var_u_m, 1e-26), color='#d95f02', lw=1.5, label=r'Rugosity $\langle w^2 \rangle_t$')
    ax1.set_ylabel(r'Rugosity $\langle w^2 \rangle$', fontsize=12)
    ax1.set_ylim(bottom=max(1e-20, min_rug * 0.1), top=max(1.0, var_u_m.max() * 2.0))
    ax1.axhline(1e-4, color='gray', linestyle=':', lw=1.0, label='Flat/Rough Threshold ($10^{-4}$)')
    ax1.legend(loc='upper left', fontsize=9.5)
    ax1.set_title(f"{case['title']} ($h_0={case['h0']}, h_\\mathrm{{a}}={case['ha']}$)\n"
                  f"Floquet: $\\mu={case['floquet_mu']:.3f}, k_\\mu={case['floquet_k']:.3f}$  |  $10^4$ Driving Periods",
                  fontsize=10.5)

    ax2.plot(p_m, phi_dot_m, color='#7570b3', lw=1.2, label=r'$\langle \dot{\phi} \rangle$')
    ax2.axhline(0.5, color='black', linestyle='--', lw=1.0, label=r'Synchronous locking $\omega/2 = 0.5$')
    ax2.plot(p_m, u_dot_m, color='#1b9e77', lw=1.0, alpha=0.8, label=r'Velocity $\langle \dot{u} \rangle$')
    ax2.set_xlabel(r'Driving Periods $t / T_\mathrm{drive}$', fontsize=12)
    ax2.set_ylabel(r'Velocities', fontsize=12)
    ax2.legend(loc='lower right', fontsize=8.5)

    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(os.path.join(case_dir, "evolution_rugosity_velocity.png"))
    fig.savefig(os.path.join(case_dir, "evolution_rugosity_velocity.pdf"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Spatio-Temporal Hovmoller Diagram (Space-Time Heatmap)
    # -------------------------------------------------------------------------
    # Restrict to 4 driving periods to clearly see wave motion
    n_cycles = 4
    t_span = n_cycles * T_LOCK
    t_end = snap_t[-1]
    t_start = t_end - t_span
    mask_window = (snap_t >= t_start) & (snap_t <= t_end)

    t_hov = snap_t[mask_window]
    u_hov = u_prof[mask_window]
    phi_hov = phi_prof[mask_window]

    # Center height around spatial mean to isolate wavy corrugation
    u_hov_centered = u_hov - u_hov.mean(axis=1, keepdims=True)

    fig, (ax_u, ax_phi) = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    X_grid, T_grid = np.meshgrid(x, t_hov)

    im_u = ax_u.pcolormesh(X_grid, T_grid, u_hov_centered, cmap='coolwarm', shading='nearest', rasterized=True)
    ax_u.set_xlabel(r'Position $x$', fontsize=11)
    ax_u.set_ylabel(r'Time $t$', fontsize=11)
    ax_u.set_title(r'(a) Relative Height $u - \langle u \rangle_x$', fontsize=11)
    cb_u = fig.colorbar(im_u, ax=ax_u, pad=0.03, fraction=0.046)
    cb_u.ax.tick_params(labelsize=8)

    im_phi = ax_phi.pcolormesh(X_grid, T_grid, phi_hov % (2 * np.pi), cmap='twilight', shading='nearest', rasterized=True)
    ax_phi.set_xlabel(r'Position $x$', fontsize=11)
    ax_phi.set_title(r'(b) Magnetization Angle $\phi(x, t)$', fontsize=11)
    cb_phi = fig.colorbar(im_phi, ax=ax_phi, pad=0.03, fraction=0.046)
    cb_phi.ax.tick_params(labelsize=8)

    for ax in (ax_u, ax_phi):
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(os.path.join(case_dir, "hovmoller_space_time.png"))
    fig.savefig(os.path.join(case_dir, "hovmoller_space_time.pdf"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Wave Profile Snapshots & Spatial Power Spectrum
    # -------------------------------------------------------------------------
    fig, (ax_prof, ax_spec) = plt.subplots(1, 2, figsize=(7.6, 3.2))

    # Pick 4 equidistant frames in a single driving cycle T_lock
    cycle_mask = (snap_t >= t_end - T_LOCK) & (snap_t <= t_end)
    cycle_indices = np.where(cycle_mask)[0]
    frame_picks = [cycle_indices[int(round(f * (len(cycle_indices) - 1)))] for f in [0.0, 0.25, 0.5, 0.75]]
    colors_prof = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

    for i, idx in enumerate(frame_picks):
        t_val = snap_t[idx]
        dt_rel = (t_val - (t_end - T_LOCK)) / T_LOCK
        u_c = u_prof[idx] - u_prof[idx].mean()
        x_plot = np.append(x, L)
        u_plot = np.append(u_c, u_c[0])
        ax_prof.plot(x_plot, u_plot, color=colors_prof[i], lw=1.4, marker='o', markersize=3, label=f'$t = t_0 + {dt_rel:.2f}T$')

    ax_prof.set_xlabel(r'Position $x$', fontsize=11)
    ax_prof.set_ylabel(r'Corrugation $u - \langle u \rangle_x$', fontsize=11)
    ax_prof.set_title('Spatial Wave Profiles', fontsize=11)
    ax_prof.legend(loc='upper right', fontsize=8.0, framealpha=0.9)

    # Power spectrum averaged over the steady-state snapshots
    fft_vals = np.fft.fft(u_prof[cycle_mask] - u_prof[cycle_mask].mean(axis=1, keepdims=True), axis=1)
    k_vals = 2 * np.pi * np.fft.fftfreq(SIZE, d=DX)
    pos_k = k_vals > 0
    k_pos = k_vals[pos_k]
    power_spectrum = np.mean(np.abs(fft_vals[:, pos_k]) ** 2 / SIZE, axis=0)

    ax_spec.semilogy(k_pos, power_spectrum + 1e-15, color='#2c7bb6', marker='o', markersize=3.5, lw=1.2, label='Simulation')
    if case['floquet_k'] > 0:
        ax_spec.axvline(case['floquet_k'], color='#d7191c', linestyle='--', lw=1.5,
                        label=f"Floquet $k_\\mu = {case['floquet_k']:.3f}$")

    # Annotate peak simulated wavenumber
    peak_idx = np.argmax(power_spectrum)
    peak_k = k_pos[peak_idx]
    ax_spec.plot(peak_k, power_spectrum[peak_idx], marker='*', color='#d7191c', markersize=9, zorder=5)

    ax_spec.set_xlabel(r'Wavenumber $k$', fontsize=11)
    ax_spec.set_ylabel(r'Power $|u(k)|^2$', fontsize=11)
    ax_spec.set_title(f'Power Spectrum (Peak $k={peak_k:.3f}$)', fontsize=11)
    ax_spec.set_xlim(0, 2.5)
    ax_spec.legend(loc='upper right', fontsize=8.5)

    for ax in (ax_prof, ax_spec):
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(os.path.join(case_dir, "profile_and_spectrum.png"))
    fig.savefig(os.path.join(case_dir, "profile_and_spectrum.pdf"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. Animated Video (MP4 and GIF)
    # -------------------------------------------------------------------------
    print(f"[{case['id']}] Rendering animated movie...")
    fig_anim, (ax_u_anim, ax_phi_anim) = plt.subplots(2, 1, figsize=(6.0, 4.4), sharex=True)

    u_min = (u_prof[cycle_mask] - u_prof[cycle_mask].mean(axis=1, keepdims=True)).min() * 1.2
    u_max = (u_prof[cycle_mask] - u_prof[cycle_mask].mean(axis=1, keepdims=True)).max() * 1.2
    if abs(u_max - u_min) < 1e-4:
        u_min, u_max = -0.1, 0.1

    ax_u_anim.set_xlim(0, L)
    ax_u_anim.set_ylim(u_min, u_max)
    ax_u_anim.set_ylabel(r'$u(x) - \langle u \rangle_x$', fontsize=11)
    line_u, = ax_u_anim.plot([], [], color='#d95f02', lw=1.8)

    ax_phi_anim.set_xlim(0, L)
    ax_phi_anim.set_ylim(0, 2 * np.pi)
    ax_phi_anim.set_yticks([0, np.pi, 2 * np.pi])
    ax_phi_anim.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax_phi_anim.set_ylabel(r'$\phi(x) \ (\mathrm{mod}\ 2\pi)$', fontsize=11)
    ax_phi_anim.set_xlabel(r'Position $x$', fontsize=11)
    line_phi, = ax_phi_anim.plot([], [], color='#7570b3', lw=1.8)

    title_text = ax_u_anim.set_title("", fontsize=10.5)

    # Subsample frames across cycle
    anim_indices = cycle_indices[::2]  # ~50-100 frames for smooth playback

    def init_anim():
        line_u.set_data([], [])
        line_phi.set_data([], [])
        title_text.set_text("")
        return line_u, line_phi, title_text

    def update_anim(frame_idx):
        t_val = snap_t[frame_idx]
        u_c = u_prof[frame_idx] - u_prof[frame_idx].mean()
        p_c = phi_prof[frame_idx] % (2 * np.pi)

        x_plot = np.append(x, L)
        u_plot = np.append(u_c, u_c[0])
        p_plot = np.append(p_c, p_c[0])
        line_u.set_data(x_plot, u_plot)
        line_phi.set_data(x_plot, p_plot)

        h_t = case['h0'] + case['ha'] * np.cos(OMEGA * t_val)
        rug_val = np.var(u_prof[frame_idx])
        title_text.set_text(
            f"{case['title']}  |  $t={t_val:.1f}$  |  $h(t)={h_t:.2f}$  |  $w^2={rug_val:.2e}$"
        )
        return line_u, line_phi, title_text

    anim = animation.FuncAnimation(
        fig_anim, update_anim, init_func=init_anim,
        frames=anim_indices, interval=50, blit=False
    )

    mp4_path = os.path.join(case_dir, "movie.mp4")
    gif_path = os.path.join(case_dir, "movie.gif")

    try:
        anim.save(mp4_path, writer='ffmpeg', fps=20, dpi=180)
        print(f"[{case['id']}] Saved MP4 video: {mp4_path}")
    except Exception as e:
        print(f"[{case['id']}] Note: could not render mp4 ({e}). Trying GIF...")

    try:
        anim.save(gif_path, writer='pillow', fps=15, dpi=120)
        print(f"[{case['id']}] Saved GIF animation: {gif_path}")
    except Exception as e:
        print(f"[{case['id']}] Could not save gif: {e}")

    plt.close(fig_anim)

    return {
        'case': case,
        't_m': t_m, 'var_u_m': var_u_m, 'phi_dot_m': phi_dot_m,
        'k_pos': k_pos, 'power_spectrum': power_spectrum,
        'peak_k': peak_k, 'peak_power': power_spectrum[peak_idx],
        'sat_rugosity': var_u_m[-1],
        'hov_t': t_hov, 'hov_u': u_hov_centered
    }

# ==============================================================================
# CROSS-CASE COMPARATIVE SUMMARY FIGURE
# ==============================================================================
def generate_comparative_summary(results):
    print("\nGenerating comprehensive comparative summary figure...")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))

    # (a) Rugosity Saturation over 10,000 Driving Periods
    ax_a = axes[0, 0]
    colors = ['#d95f02', '#e7298a', '#7570b3', '#2ca02c', '#1f78b4']
    for idx, res in enumerate(results):
        c = res['case']
        p_m = res['t_m'] / T_DRIVE
        ax_a.semilogy(p_m, np.maximum(res['var_u_m'], 1e-26),
                      color=colors[idx], lw=1.5,
                      label=f"{c['title']} ($h_0={c['h0']}, h_\\mathrm{{a}}={c['ha']}$)")

    ax_a.axhline(1e-4, color='gray', linestyle=':', lw=1.0)
    ax_a.set_xlabel(r'Driving Periods $t / T_\mathrm{drive}$', fontsize=11)
    ax_a.set_ylabel(r'Rugosity $\langle w^2 \rangle_t$', fontsize=11)
    ax_a.set_title(r'(a) Rugosity $w^2$ over $10^4$ Driving Periods', fontsize=11)
    ax_a.set_ylim(bottom=1e-25, top=2.0)
    ax_a.legend(loc='lower right', fontsize=7.5, framealpha=0.9)

    # (b) Synchronization Locking \phi_dot
    ax_b = axes[0, 1]
    for idx, res in enumerate(results):
        c = res['case']
        p_m = res['t_m'] / T_DRIVE
        ax_b.plot(p_m, res['phi_dot_m'], color=colors[idx], lw=1.2,
                  label=f"Case {idx+1}")
    ax_b.axhline(0.5, color='black', linestyle='--', lw=1.2, label=r'$\omega/2 = 0.5$ (Synchronous)')
    ax_b.set_xlabel(r'Driving Periods $t / T_\mathrm{drive}$', fontsize=11)
    ax_b.set_ylabel(r'Phase Velocity $\langle \dot{\phi} \rangle$', fontsize=11)
    ax_b.set_title(r'(b) Phase Locking $\langle \dot{\phi} \rangle$ over $10^4$ Periods', fontsize=11)
    ax_b.set_ylim(0.48, 0.52)
    ax_b.legend(loc='upper right', fontsize=8.0)

    # (c) Spatial Power Spectra & Mode Selection
    ax_c = axes[1, 0]
    for idx, res in enumerate(results):
        c = res['case']
        if res['sat_rugosity'] > 1e-5:
            ax_c.semilogy(res['k_pos'], res['power_spectrum'] + 1e-15,
                          color=colors[idx], marker='o', markersize=3, lw=1.2,
                          label=f"Case {idx+1} (Peak $k={res['peak_k']:.3f}$)")
            ax_c.axvline(c['floquet_k'], color=colors[idx], linestyle='--', lw=1.0, alpha=0.8)

    ax_c.set_xlabel(r'Wavenumber $k$', fontsize=11)
    ax_c.set_ylabel(r'Power $|u(k)|^2$', fontsize=11)
    ax_c.set_title(r'(c) Corrugated Modes vs Floquet $k_\mu$ (Dashed)', fontsize=11)
    ax_c.set_xlim(0, 2.5)
    ax_c.legend(loc='upper right', fontsize=7.8)

    # (d) Hovmoller Comparison of Saturated Wave Structure (Case 1)
    ax_d = axes[1, 1]
    res1 = results[0]
    X_grid, T_grid = np.meshgrid(np.linspace(0, L, SIZE), res1['hov_t'])
    im_d = ax_d.pcolormesh(X_grid, T_grid, res1['hov_u'], cmap='coolwarm', shading='nearest', rasterized=True)
    ax_d.set_xlabel(r'Position $x$', fontsize=11)
    ax_d.set_ylabel(r'Time $t$', fontsize=11)
    ax_d.set_title(r'(d) Space-Time Wave at $t \sim 10^4$ Periods (Case 1)', fontsize=11)
    cb_d = fig.colorbar(im_d, ax=ax_d, pad=0.02, fraction=0.046)
    cb_d.set_label(r'$u - \langle u \rangle_x$', fontsize=10)
    cb_d.ax.tick_params(labelsize=8)

    for ax in axes.flat:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[0, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[1, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    summary_png = os.path.join(OUT_BASE, "comparative_unstable_sync_summary.png")
    summary_pdf = os.path.join(OUT_BASE, "comparative_unstable_sync_summary.pdf")
    fig.savefig(summary_png, dpi=300)
    fig.savefig(summary_pdf)
    plt.close(fig)
    print(f"Summary figure saved: {summary_png} and {summary_pdf}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*75)
    print(f"  PHYSICAL INVESTIGATION: {N_PERIODS} DRIVING PERIODS DIAGNOSTICS")
    print("="*75)
    print(f"Target Directory: {OUT_BASE}")
    print(f"Extended System: Size={SIZE}, Length L={L}, dx={DX}")
    print(f"Simulation duration: t_max={T_MAX:.2f} ({N_PERIODS} driving periods)")
    print(f"Number of test cases: {len(CASES)}\n")

    # Step 1: Run all simulations in parallel
    print(f"Launching simulations in parallel across {len(CASES)} CPU workers...")
    t_start = time.time()
    force_flag = '--force' in sys.argv
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CASES)) as executor:
        futures = {executor.submit(run_simulation, c, os.path.join(OUT_BASE, c['id']), force_flag): c for c in CASES}
        for future in concurrent.futures.as_completed(futures):
            c = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Simulation failed for {c['id']}: {e}")
                sys.exit(1)

    print(f"\nAll simulations completed in {time.time() - t_start:.2f}s! Generating diagnostics, spectra, and movies...")
    results = []
    for c in CASES:
        print("-" * 65)
        print(f"Processing diagnostics for {c['title']}...")
        res = process_case(c)
        results.append(res)

    generate_comparative_summary(results)

    print("\n" + "="*75)
    print(f"ALL DIAGNOSTICS FOR {N_PERIODS} DRIVING PERIODS COMPLETED SUCCESSFULLY!")
    print(f"Outputs, figures, and videos stored in: {OUT_BASE}")
    print("="*75)

if __name__ == "__main__":
    main()
