import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def setup_prb_style():
    """Configures matplotlib to match Physical Review B single-column style."""
    plt.rcParams.update({
        'figure.figsize': (3.375, 2.5),  # 3.375 inches is the standard 1-column PRB width
        'font.size': 10,
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.8,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def find_prefix(data_dir):
    """Finds the prefix of the data files in the given directory."""
    files = list(data_dir.glob("*_transient_metrics.txt"))
    if not files:
        raise FileNotFoundError(f"No *_transient_metrics.txt found in {data_dir}")
    # Extract the prefix by removing the known suffix
    return files[0].name.replace("_transient_metrics.txt", "")

def plot_time_series(data_dir, output_dir, prefix):
    file_path = data_dir / f"{prefix}_time_series.txt"
    if not file_path.exists():
        print(f"Skipping time series: {file_path.name} not found.")
        return

    data = np.loadtxt(file_path, skiprows=1)
    time = data[:, 0]
    mean_u = data[:, 1]
    u_dot = data[:, 2]

    # Plot mean_u
    fig, ax = plt.subplots()
    ax.plot(time, mean_u, color='black', rasterized=True, linewidth=0.5)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'$\langle u \rangle_{x,t}$')
    fig.savefig(output_dir / "time_series_mean_u.pdf")
    plt.close(fig)

    # Plot u_dot
    fig, ax = plt.subplots()
    ax.plot(time, u_dot, color='tab:red', rasterized=True, linewidth=0.5)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Velocity $\langle \dot{u} \rangle$')
    fig.savefig(output_dir / "time_series_u_dot.pdf")
    plt.close(fig)

    # Plot Temporal Power Spectrum of u_dot
    if len(time) > 1:
        dt_ts = time[1] - time[0]
        N = len(time)
        
        # Subtract mean to remove the large DC component at omega=0
        u_dot_centered = u_dot - np.mean(u_dot)
        
        # Compute FFT and angular frequencies (omega = 2 * pi * f)
        fft_u_dot = np.fft.fft(u_dot_centered)
        freqs = np.fft.fftfreq(N, d=dt_ts)
        omega = 2 * np.pi * freqs
        
        # Filter for positive frequencies
        pos_mask = freqs > 0
        omega_pos = omega[pos_mask]
        power_u_dot = np.abs(fft_u_dot[pos_mask])**2 / N
        
        fig, ax = plt.subplots()
        ax.loglog(omega_pos, power_u_dot, color='tab:purple', linewidth=0.5, rasterized=True)
        ax.set_xlabel(r'Angular Frequency $\omega$')
        ax.set_ylabel(r'Power Spectrum of $\langle \dot{u} \rangle$')
        fig.savefig(output_dir / "time_series_u_dot_ps.pdf")
        plt.close(fig)

def plot_transient_metrics(data_dir, output_dir, prefix):
    file_path = data_dir / f"{prefix}_transient_metrics.txt"
    if not file_path.exists():
        print(f"Skipping transient metrics: {file_path.name} not found.")
        return

    data = np.loadtxt(file_path, skiprows=1)
    if data.size == 0:
        print("Transient metrics file is empty.")
        return

    # Handle 1D array if only one row exists
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[:, 1]
    avg_var_u = data[:, 2]
    avg_u_dot = data[:, 3]
    avg_phi_dot = data[:, 4]
    avg_entropy = data[:, 5]

    # Read final summary values to plot as horizontal asymptotes
    summary_path = data_dir / f"{prefix}_final_summary.txt"
    final_vals = None
    if summary_path.exists():
        try:
            summary_data = np.loadtxt(summary_path, skiprows=1)
            if summary_data.size > 0:
                # Ensure it's 1D
                if summary_data.ndim > 1:
                    summary_data = summary_data[0]
                final_vals = {
                    'var_u': summary_data[1],
                    'u_dot': summary_data[2],
                    'phi_dot': summary_data[3],
                    'entropy': summary_data[4]
                }
        except Exception as e:
            print(f"Could not parse final summary for asymptotes: {e}")

    # Plot Variance
    fig, ax = plt.subplots()
    ax.plot(time, avg_var_u, marker='o', markersize=3, linestyle='-', color='tab:blue', label='Transient')
    if final_vals:
        ax.axhline(final_vals['var_u'], color='black', linestyle='--', linewidth=1, label='Final Avg')
        ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Variance $\sigma_u^2$')
    fig.savefig(output_dir / "transient_variance.pdf")
    plt.close(fig)

    # Plot u_dot
    fig, ax = plt.subplots()
    ax.plot(time, avg_u_dot, marker='^', markersize=3, linestyle='-', color='tab:orange', label='Transient')
    if final_vals:
        ax.axhline(final_vals['u_dot'], color='black', linestyle='--', linewidth=1, label='Final Avg')
        ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Avg Velocity $\langle \dot{u} \rangle$')
    fig.savefig(output_dir / "transient_u_dot.pdf")
    plt.close(fig)

    # Plot phi_dot
    fig, ax = plt.subplots()
    ax.plot(time, avg_phi_dot, marker='v', markersize=3, linestyle='-', color='tab:green', label='Transient')
    if final_vals:
        ax.axhline(final_vals['phi_dot'], color='black', linestyle='--', linewidth=1, label='Final Avg')
        ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Avg Phase Vel. $\langle \dot{\phi} \rangle$')
    fig.savefig(output_dir / "transient_phi_dot.pdf")
    plt.close(fig)

    # Plot Spectral Entropy
    fig, ax = plt.subplots()
    ax.plot(time, avg_entropy, marker='s', markersize=3, linestyle='-', color='tab:purple', label='Transient')
    if final_vals:
        ax.axhline(final_vals['entropy'], color='black', linestyle='--', linewidth=1, label='Final Avg')
        ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Spectral Entropy $S$')
    fig.savefig(output_dir / "transient_entropy.pdf")
    plt.close(fig)

def plot_power_spectrum(data_dir, output_dir, prefix):
    file_path = data_dir / f"{prefix}_final_ps.txt"
    if not file_path.exists():
        print(f"Skipping power spectrum: {file_path.name} not found.")
        return

    data = np.loadtxt(file_path, skiprows=1)
    k = data[:, 0]
    power = data[:, 1]

    # Filter out k=0 for log-log plotting
    mask = k > 0
    k_pos = k[mask]
    power_pos = power[mask]

    fig, ax = plt.subplots()
    ax.loglog(k_pos, power_pos, color='black', linewidth=1.2)
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'Structure Factor $S(k)$')
    
    fig.savefig(output_dir / "power_spectrum.pdf")
    plt.close(fig)

def plot_height_histogram(data_dir, output_dir, prefix):
    file_path = data_dir / f"{prefix}_final_height_hist.txt"
    if not file_path.exists():
        print(f"Skipping histogram: {file_path.name} not found.")
        return

    data = np.loadtxt(file_path, skiprows=1)
    bin_center = data[:, 0]
    pdf = data[:, 2]

    fig, ax = plt.subplots()
    ax.plot(bin_center, pdf, color='tab:blue', linestyle='-', linewidth=1.2)
    ax.fill_between(bin_center, pdf, color='tab:blue', alpha=0.3)
    
    # Standard normal distribution overlay for reference
    std_normal = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * bin_center**2)
    ax.plot(bin_center, std_normal, color='black', linestyle='--', linewidth=1.0, label='Gaussian')
    
    ax.set_xlabel(r'Normalized Height $(u - \langle u \rangle)/\sigma_u$')
    ax.set_ylabel(r'Probability Density $P$')
    ax.legend()
    
    fig.savefig(output_dir / "final_height_hist.pdf")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate PRB style plots from simulation data.")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the simulation text files.")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Path to save the PDF figures. Defaults to ../../figures relative to data_dir.")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    
    if not data_dir.is_dir():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        # Defaults to the parent of the parent of the data directory, then /figures
        output_dir = data_dir.parent.parent / "figures"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        prefix = find_prefix(data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Reading data from: {data_dir}")
    print(f"Using file prefix: {prefix}")
    print(f"Saving figures to: {output_dir}")

    setup_prb_style()

    plot_time_series(data_dir, output_dir, prefix)
    plot_transient_metrics(data_dir, output_dir, prefix)
    plot_power_spectrum(data_dir, output_dir, prefix)
    plot_height_histogram(data_dir, output_dir, prefix)
    
    print("Done generating plots.")

if __name__ == "__main__":
    main()