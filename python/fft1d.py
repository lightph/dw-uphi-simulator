import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# Set publication-standard aesthetics
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_data(csv_filepath):
    data = np.loadtxt(csv_filepath, delimiter=",", skiprows=1)

    A_vals = data[:, 0]
    freq_vals = data[:, 3]
    power_vals = data[:, 4]

    unique_A = np.unique(A_vals)
    unique_Freq = np.unique(freq_vals)

    Z = power_vals.reshape(len(unique_A), len(unique_Freq))

    return unique_A, unique_Freq, Z


def plot_spectrogram(unique_A, unique_Freq, Z, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    min_val = np.min(Z[Z > 0]) if np.any(Z > 0) else 1e-10

    X, Y = np.meshgrid(unique_Freq * 2 * np.pi, unique_A)

    c = ax.pcolormesh(
        X,
        Y,
        Z,
        shading="nearest",
        cmap="inferno",
        norm=LogNorm(vmin=min_val),
    )

    ax.set_xlabel("Angular Frequency")
    ax.set_ylabel("Amplitude A")

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Power")

    fig.savefig(save_path)
    plt.close(fig)


def plot_noise_measure(unique_A, Z, save_path):
    noise_floor = np.median(Z, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(unique_A, noise_floor, color="indigo")

    ax.set_xlabel("Amplitude A")
    ax.set_ylabel("Median Power")
    ax.set_yscale("log")

    ax.grid(True, linestyle=":", alpha=0.7)

    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    target_file = (
        "output/fft1d/20260303_180608__fft_sweep_A0.000000-10.000000_pts10000.csv"
    )

    try:
        unique_A, unique_Freq, Z = load_data(target_file)

        out_dir = os.path.dirname(target_file)
        base_name = os.path.splitext(os.path.basename(target_file))[0]

        spectrogram_out = os.path.join(out_dir, f"{base_name}_spectrogram.png")
        median_out = os.path.join(out_dir, f"{base_name}_median.png")

        plot_spectrogram(unique_A, unique_Freq, Z, spectrogram_out)
        plot_noise_measure(unique_A, Z, median_out)

        print(f"Saved: {spectrogram_out}")
        print(f"Saved: {median_out}")

    except OSError:
        print(f"File not found or unreadable: {target_file}. Please check the path.")
