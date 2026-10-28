import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# --- Setup Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)

plot_dir = os.path.join(project_root, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Shared Physics Parameters matching the bash script
SIZE = 128
L = 64.0


def safe_load(filepath):
    """Safely load txt data, skipping strings in headers."""
    if not os.path.exists(filepath):
        print(f"[Skip] {filepath} not found.")
        return None, None
    try:
        with open(filepath, "r") as f:
            header = f.readline().strip().split()
        data = np.loadtxt(filepath, skiprows=1)
        if data.size == 0:
            return None, header
        # Ensure 2D array even if single row
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data, header
    except Exception as e:
        print(f"[Error] Could not process {filepath}: {e}")
        return None, None


def plot_time_series():
    data, cols = safe_load("test_main_time_series.txt")
    if data is None:
        return

    # Filter for batch 0 only to keep plots clean
    b0_mask = data[:, 0] == 0
    t = data[b0_mask, 1]
    mean_u = data[b0_mask, 2]
    u_dot = data[b0_mask, 3]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, mean_u, label="Mean $u$", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mean $u$", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(t, u_dot, label="$\dot{u}$", color="red", alpha=0.6)
    ax2.set_ylabel("$\dot{u}$", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Time Evolution of Observables (Batch 0)")
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "1_time_series.png"), dpi=150)
    plt.close()
    print("[Done] Time series plot.")


def plot_spectra_and_hist():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Power Spectrum
    ps_data, _ = safe_load("test_main_final_ps.txt")
    if ps_data is not None:
        b0_mask = ps_data[:, 0] == 0
        k = ps_data[b0_mask, 1]
        power = ps_data[b0_mask, 2]

        # Only plot positive k half for standard PS view
        pos_k = k[k >= 0]
        pos_power = power[k >= 0]

        ax1.loglog(
            pos_k, pos_power, marker="o", markersize=3, linestyle="-", color="purple"
        )
        ax1.set_xlabel("Wavenumber $k$")
        ax1.set_ylabel("Power $S(k)$")
        ax1.set_title("Final Power Spectrum")
        ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Histogram
    hist_data, _ = safe_load("test_main_final_height_hist.txt")
    if hist_data is not None:
        b0_mask = hist_data[:, 0] == 0
        bin_center = hist_data[b0_mask, 1]
        pdf = hist_data[b0_mask, 3]

        ax2.plot(bin_center, pdf, drawstyle="steps-mid", color="green")
        ax2.fill_between(bin_center, pdf, step="mid", color="green", alpha=0.3)
        ax2.set_xlabel("Normalized Height (Z)")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("Final Height Distribution")
        ax2.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "2_spectra_hist.png"), dpi=150)
    plt.close()
    print("[Done] Spectra and histogram plot.")


def plot_moments():
    data, cols = safe_load("test_moments_moments.txt")
    if data is None:
        return

    t = data[:, 0]
    mu2 = data[:, 2]  # Variance
    skew = data[:, 10]
    kurt = data[:, 11]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, mu2, color="orange")
    axs[0].set_ylabel("Variance ($\mu_2$)")
    axs[0].set_title("Evolution of Global Moments")
    axs[0].grid(alpha=0.5)

    axs[1].plot(t, skew, color="teal")
    axs[1].set_ylabel("Skewness")
    axs[1].grid(alpha=0.5)

    axs[2].plot(t, kurt, color="maroon")
    axs[2].set_ylabel("Excess Kurtosis")
    axs[2].set_xlabel("Time")
    axs[2].grid(alpha=0.5)

    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3_moments.png"), dpi=150)
    plt.close()
    print("[Done] Moments evolution plot.")


def plot_structure_factor():
    data, cols = safe_load("test_moments_S_q_t.txt")
    if data is None:
        return

    times = np.unique(data[:, 0])
    qs = np.unique(data[:, 1])

    if len(times) > 1 and len(qs) > 1:
        # Reshape data into 2D grid for heatmap
        # Ensure data is sorted by time then q
        data_sorted = data[np.lexsort((data[:, 1], data[:, 0]))]
        S_qt = data_sorted[:, 2].reshape(len(times), len(qs))

        fig, ax = plt.subplots(figsize=(8, 6))
        # LogNorm enhances contrast for power spectrums dropping exponentially
        c = ax.pcolormesh(
            qs, times, S_qt, shading="nearest", cmap="inferno", norm=mcolors.LogNorm()
        )
        ax.set_xlabel("Wavenumber $q$")
        ax.set_ylabel("Time $t$")
        ax.set_title("Structure Factor $S(q,t)$ Evolution")
        fig.colorbar(c, ax=ax, label="$S(q,t)$")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "4_S_q_t_heatmap.png"), dpi=150)
        plt.close()
        print("[Done] Structure factor heatmap.")


def animate_binary_movie():
    filepath = "test_movies_snapshots.bin"
    if not os.path.exists(filepath):
        print(f"[Skip] {filepath} not found.")
        return

    frames_u = []
    frames_phi = []
    times = []

    # 8 bytes for size_t (batch), 8 bytes for double (time)
    header_fmt = "Qd"
    header_size = struct.calcsize(header_fmt)
    # 16 bytes per std::complex<double>
    payload_size = SIZE * 16

    try:
        with open(filepath, "rb") as f:
            while True:
                header = f.read(header_size)
                if len(header) < header_size:
                    break
                batch, t = struct.unpack(header_fmt, header)
                payload = f.read(payload_size)

                # Only animate batch 0 to avoid overlapping lines
                if batch == 0:
                    # np.frombuffer reads the bytes directly into an array of floats
                    arr = np.frombuffer(payload, dtype=np.float64)
                    u = arr[0::2]  # Real part
                    phi = arr[1::2]  # Imaginary part
                    frames_u.append(u)
                    frames_phi.append(phi)
                    times.append(t)
    except Exception as e:
        print(f"[Error] Failed to read binary movie: {e}")
        return

    if not frames_u:
        print(
            "[Warning] Binary movie read successfully but contained no batch 0 frames."
        )
        return

    # To keep GIF generation fast and file size small, limit to max 300 frames
    stride = max(1, len(frames_u) // 300)
    frames_u = frames_u[::stride]
    frames_phi = frames_phi[::stride]
    times = times[::stride]

    x = np.linspace(0, L, SIZE, endpoint=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Fixed y-limits slightly wider than min/max of the data
    u_min, u_max = np.min(frames_u), np.max(frames_u)
    p_min, p_max = np.min(frames_phi), np.max(frames_phi)

    # Add 10% padding
    u_pad = (u_max - u_min) * 0.1 if u_max != u_min else 1.0
    p_pad = (p_max - p_min) * 0.1 if p_max != p_min else 1.0

    ax1.set_ylim(u_min - u_pad, u_max + u_pad)
    ax2.set_ylim(p_min - p_pad, p_max + p_pad)
    ax1.set_xlim(0, L)

    ax1.set_ylabel("$u(x)$")
    ax2.set_ylabel("$\phi(x)$")
    ax2.set_xlabel("Position $x$")

    (line_u,) = ax1.plot(x, frames_u[0], color="navy")
    (line_phi,) = ax2.plot(x, frames_phi[0], color="darkred")
    title = ax1.set_title(f"Time: {times[0]:.2f}")

    def update(frame):
        line_u.set_ydata(frames_u[frame])
        line_phi.set_ydata(frames_phi[frame])
        title.set_text(f"Time: {times[frame]:.2f}")
        return line_u, line_phi, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_u), interval=50, blit=True
    )

    out_file = os.path.join(plot_dir, "5_movie_animation.gif")
    # Use pillow to generate the GIF so it doesn't require FFMPEG installations
    ani.save(out_file, writer="pillow", fps=15)
    plt.close()
    print("[Done] Binary movie animation saved to GIF.")


def main():
    print("========================================")
    print("Generating Simulation Visualizations")
    print(f"Reading from: {project_root}")
    print("========================================\n")

    plot_time_series()
    plot_spectra_and_hist()
    plot_moments()
    plot_structure_factor()
    animate_binary_movie()

    print(f"\n========================================")
    print(f"SUCCESS: All visualizations saved to:\n{plot_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
