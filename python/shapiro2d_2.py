import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import curve_fit
from scipy.special import jv


def style_ax(ax):
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6, color="#bdc3c7")
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_color("#7f8c8d")
    ax.spines.bottom.set_color("#7f8c8d")
    ax.tick_params(colors="#333333", labelsize=10)


title_font = {"fontsize": 14, "color": "black"}
label_font = {"fontsize": 12, "color": "black"}


def get_data_for_b(idx, C_array, step_array):
    return C_array, step_array[idx, :]


def show_interactive_plot(C_array, B_array, step_array, n):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.25)

    initial_idx = 0
    c_vals, step_vals = get_data_for_b(initial_idx, C_array, step_array)

    (line,) = ax.plot(c_vals, step_vals, lw=2.5, color="#2c3e50")
    ax.set_xlabel("$f_{AC}$", **label_font)
    ax.set_ylabel(f"Escalón {n}", **label_font)
    ax.set_title(f"B = {B_array.item(initial_idx):.4g}", pad=15, **title_font)
    ax.set_ylim(step_array.min(), step_array.max())
    style_ax(ax)

    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(
        ax=ax_slider,
        label="B",
        valmin=0,
        valmax=len(B_array) - 1,
        valinit=initial_idx,
        valstep=1,
        valfmt="%d",
        color="#e74c3c",
    )

    slider.valtext.set_text(f"{B_array.item(initial_idx):.4g}")

    def update(val):
        idx = int(slider.val)
        _, new_step_vals = get_data_for_b(idx, C_array, step_array)
        line.set_ydata(new_step_vals)
        ax.set_title(f"B = {B_array.item(idx):.4g}", pad=15, **title_font)
        slider.valtext.set_text(f"{B_array.item(idx):.4g}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return slider


def create_gif(C_array, B_array, step_array, n, fit_func, filename="output.gif"):
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

    initial_idx = 0
    c_vals, step_vals = get_data_for_b(initial_idx, C_array, step_array)

    # Initialize both the data line and the fit line
    (line_data,) = ax.plot(c_vals, step_vals, lw=2.5, color="#2c3e50", label="Data")
    (line_fit,) = ax.plot(
        c_vals,
        np.zeros_like(c_vals),
        lw=1.5,
        color="#e74c3c",
        linestyle="--",
        label="Fit",
    )

    ax.set_xlabel("$f_{AC}$", **label_font)
    ax.set_ylabel(f"Escalón {n}", **label_font)
    ax.set_title(f"B = {B_array.item(initial_idx):.4g}", pad=15, **title_font)
    ax.set_ylim(step_array.min(), step_array.max())
    style_ax(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    def update(idx):
        _, new_step_vals = get_data_for_b(idx, C_array, step_array)
        line_data.set_ydata(new_step_vals)
        ax.set_title(f"B = {B_array.item(idx):.4g}", pad=15, **title_font)

        # Attempt to fit the data for the current frame
        try:
            popt, _ = curve_fit(
                fit_func, c_vals, new_step_vals, p0=(1.0, 1.0), maxfev=5000
            )
            fit_vals = fit_func(c_vals, *popt)
            line_fit.set_ydata(fit_vals)
        except RuntimeError:
            # Hide the fit line if convergence fails for this specific frame
            line_fit.set_ydata(np.full_like(c_vals, np.nan))

        return line_data, line_fit

    ani = FuncAnimation(fig, update, frames=len(B_array), blit=False)
    ani.save(filename, writer=PillowWriter(fps=15))
    plt.close()


def plot_bessel_verification(idx, C_array, step_array, popt, fit_func, n, B_array):
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    c_vals, step_vals = get_data_for_b(idx, C_array, step_array)

    ax.plot(c_vals, step_vals, lw=1.5, color="#2c3e50", label="Data")
    ax.plot(
        c_vals,
        fit_func(c_vals, *popt),
        lw=1,
        color="#e74c3c",
        linestyle="--",
        label="Fit",
    )

    ax.set_xlabel("$f_{AC}$", **label_font)
    ax.set_ylabel(f"Escalón {n}", **label_font)
    ax.set_title(f"B = {B_array.item(idx):.4g}", pad=15, **title_font)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=10)
    return fig, ax


def get_bessel_fit(n):
    # Returns a function strictly configured for order n
    def bessel_fit(c, a, b):
        return np.abs(a * jv(n, b * c))

    return bessel_fit


# Iterate over the specified steps
for n in [0, 1, 2]:
    file_name = f"./output/shapiro2d/run_step{n}.csv"
    print(f"Procesando {file_name}...")

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Archivo {file_name} no encontrado. Omitiendo.")
        continue

    grid = df.pivot(index="B", columns="C", values="step")

    B_array = np.array(grid.index)
    C_array = np.array(grid.columns)
    step_array = np.array(grid.values)

    # Colormap Plot
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 6))
    mesh = ax.pcolormesh(C_array, B_array, step_array, shading="auto", cmap="magma")
    fig.colorbar(mesh)
    ax.set_xlabel("$f_{AC}$", **label_font)
    ax.set_ylabel("B", **label_font)
    ax.set_title(f"Escalón {n}", pad=15, **title_font)
    ax.set_aspect(1.0)
    style_ax(ax)
    ax.grid(False)
    plt.savefig(f"output/shapiro2d/colormap_step{n}.png", dpi=300)
    plt.show()
    plt.close("all")

    # Interactive Plot and GIF
    current_bessel_fit = get_bessel_fit(n)
    create_gif(
        C_array,
        B_array,
        step_array,
        n,
        current_bessel_fit,
        f"output/shapiro2d/bessel_step{n}.gif",
    )
    show_interactive_plot(C_array, B_array, step_array, n)
    # Curve Fitting
    correlations = []

    for i in range(len(B_array)):
        step_data = step_array[i, :]

        try:
            popt, _ = curve_fit(
                current_bessel_fit, C_array, step_data, p0=(1.0, 1.0), maxfev=5000
            )
            fit_data = current_bessel_fit(C_array, *popt)

            if (i == 1) or (i == 20) or (i == len(B_array) - 1):
                figbess, axbess = plot_bessel_verification(
                    i, C_array, step_array, popt, current_bessel_fit, n, B_array
                )
                plt.savefig(
                    f"output/shapiro2d/bessel_fit_step{n}_B={B_array[i]:.2f}.png",
                    dpi=300,
                )
                plt.close(figbess)

            r = np.corrcoef(step_data, fit_data)[0, 1]
            correlations.append(r)
        except RuntimeError:
            correlations.append(np.nan)

    # Correlation Plot
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

    ax.plot(
        B_array,
        correlations,
        marker="o",
        markersize=7,
        linestyle="-",
        linewidth=0.5,
        color="#2c3e50",
        markerfacecolor="#e74c3c",
        markeredgecolor="white",
        markeredgewidth=1.0,
    )

    ax.set_xlabel("B", **label_font)
    ax.set_ylabel("Correlación", **label_font)
    style_ax(ax)

    plt.savefig(f"output/shapiro2d/bessel_correlacion_step{n}.png", dpi=300)
    plt.show()
