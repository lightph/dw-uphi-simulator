import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import LogNorm


def load_vel_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".bin":
        raw = np.fromfile(filepath, dtype=np.float64).reshape(-1, 4)
        return pd.DataFrame(raw, columns=["h", "B", "h0", "AvgVelocity"])
    elif ext == ".csv":
        df = pd.read_csv(filepath)
        if len(df.columns) == 4 and df.columns[0] != "h":
            df.columns = ["h", "B", "h0", "AvgVelocity"]
        return df
    raise ValueError("File must be .csv or .bin")


def load_fft_slice(filepath, b_target, h0_target):
    # (Existing lazy loading logic remains same, ensuring columns are h, B, h0, Freq, Power)
    pass


def plot_interactive_sweep(df_vel, fft_file_path=None):
    has_fft = bool(fft_file_path and os.path.exists(fft_file_path))

    df_vel["B_round"] = df_vel["B"].round(4)
    df_vel["h0_round"] = df_vel["h0"].round(4)

    b_vals = np.sort(df_vel["B_round"].unique())
    h0_vals = np.sort(df_vel["h0_round"].unique())
    alpha_vals = np.linspace(0.01, 1.0, 100)

    vel_dict = {
        k: v.sort_values("h") for k, v in df_vel.groupby(["B_round", "h0_round"])
    }

    fig = plt.figure(figsize=(10, 10))
    ax_u = fig.add_axes((0.12, 0.605, 0.78, 0.345))
    ax_phi = fig.add_axes((0.12, 0.220, 0.78, 0.345), sharex=ax_u)
    ax_spec = fig.add_axes((0.12, 0.220, 0.78, 0.223), sharex=ax_u) if has_fft else None

    # Initial states
    b_init = b_vals[len(b_vals) // 2]
    h0_init = h0_vals[len(h0_vals) // 2]
    alpha_init = 0.5

    init_vel = vel_dict[(b_init, h0_init)]
    h_init = init_vel["h"]
    phiavg_init = init_vel["AvgVelocity"] * alpha_init

    # Setup Plots with Markers
    (uavg_line,) = ax_u.plot(
        h_init,
        (alpha_init**2 + 1) * h_init / 2 - (1 / alpha_init) * phiavg_init,
        marker=".",
        markersize=4,
        linestyle="-",
    )
    (phiavg_line,) = ax_phi.plot(
        h_init,
        phiavg_init,
        lw=2,
        color="#1f77b4",
        marker=".",
        markersize=4,
        linestyle="-",
    )

    ax_u.set_ylabel("$\\left\\langle \\dot u \\right\\rangle$")
    ax_phi.set_ylabel("$\\left\\langle \\dot \\phi \\right\\rangle$")
    ax_u.grid(True, linestyle="--", alpha=0.7)
    ax_phi.grid(True, linestyle="--", alpha=0.7)

    # Setup Sliders
    ax_alpha = plt.axes((0.15, 0.15, 0.65, 0.03))
    ax_B = plt.axes((0.15, 0.10, 0.65, 0.03))  # Replaces Omega
    ax_h0 = plt.axes((0.15, 0.05, 0.65, 0.03))

    fig.slider_alpha = Slider(ax_alpha, "$\\alpha$", 0.01, 1.0, valinit=alpha_init)
    fig.slider_B = Slider(
        ax_B,
        "$\\alpha/\\Omega$",
        b_vals.min(),
        b_vals.max(),
        valinit=b_init,
        valstep=b_vals,
    )
    fig.slider_h0 = Slider(
        ax_h0, "$h_0$", h0_vals.min(), h0_vals.max(), valinit=h0_init, valstep=h0_vals
    )

    mesh = None

    def update(val):
        nonlocal mesh
        alpha_curr = fig.slider_alpha.val
        b_curr = fig.slider_B.val
        h0_curr = fig.slider_h0.val

        # Calculate physical Omega for the title
        omega_phys = alpha_curr / b_curr
        key = (b_curr, h0_curr)

        if key in vel_dict:
            curr_vel = vel_dict[key]
            curr_h = curr_vel["h"]
            curr_phiavg = curr_vel["AvgVelocity"] * alpha_curr

            phiavg_line.set_xdata(curr_h)
            phiavg_line.set_ydata(curr_phiavg)

            u_avg = (1 / 2) * (alpha_curr**2 + 1) * curr_h - (
                1 / alpha_curr
            ) * curr_phiavg
            uavg_line.set_xdata(curr_h)
            uavg_line.set_ydata(u_avg)

            # Boundaries and Titles
            ax_u.set_xlim(curr_h.min(), curr_h.max())
            ax_phi.set_ylim(curr_phiavg.min() * 0.95, curr_phiavg.max() * 1.05)
            ax_u.set_ylim(u_avg.min() * 0.95, u_avg.max() * 1.05)

            ax_phi.set_title(
                f"Parámetros  |  $\\alpha$: {alpha_curr:.3f}  |  $\\Omega$: {omega_phys:.3f}  |  $h_0$: {h0_curr:.3f}"
            )

            # FFT Update (Logic remains same using b_curr, h0_curr)
            if has_fft and check.get_status()[0]:
                # ... FFT rendering code ...
                pass

        fig.canvas.draw_idle()

    fig.slider_alpha.on_changed(update)
    fig.slider_B.on_changed(update)
    fig.slider_h0.on_changed(update)

    plt.show()


if __name__ == "__main__":
    vel_file_path = (
        "output/phi3d/20260308_183745__phi_sweep_3D_h1000_Omega100_h0100.bin"
    )
    df_vel = load_vel_data(vel_file_path)
    plot_interactive_sweep(df_vel)
