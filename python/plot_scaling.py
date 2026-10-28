import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def setup_prb_style():
    plt.rcParams.update({
        'figure.figsize': (3.375, 2.5),
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

def get_prefix(data_dir):
    files = list(data_dir.glob("*_transient_metrics.txt"))
    if not files:
        raise FileNotFoundError(f"No _transient_metrics.txt found in {data_dir}")
    return files[0].name.replace("_transient_metrics.txt", "")

def read_parameters(data_dir):
    param_file = data_dir / "parameters.txt"
    params = {}
    if param_file.exists():
        with open(param_file, 'r') as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    params[key.strip()] = val.strip()
    return params

def plot_individual_system(data_dir, output_dir, prefix):
    # Time Series
    ts_file = data_dir / f"{prefix}_time_series.txt"
    if ts_file.exists():
        data = np.genfromtxt(ts_file, names=True)
        if data.size > 0:
            if data.ndim == 0: data = data.reshape(1)
            if 'batch' in data.dtype.names:
                data = data[data['batch'] == 0]
            time = data['time'] if 'time' in data.dtype.names else data[data.dtype.names[0]]
            mean_u = data['mean_u'] if 'mean_u' in data.dtype.names else data[data.dtype.names[1]]
            u_dot = data['u_dot'] if 'u_dot' in data.dtype.names else data[data.dtype.names[2]]
            
            fig, ax = plt.subplots()
            ax.plot(time, mean_u, color='black', rasterized=True, linewidth=0.5)
            ax.set_xlabel(r'Time $t$')
            ax.set_ylabel(r'Mean height $\langle u \rangle$')
            fig.savefig(output_dir / "time_series_mean_u.pdf")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(time, u_dot, color='tab:red', rasterized=True, linewidth=0.5)
            ax.set_xlabel(r'Time $t$')
            ax.set_ylabel(r'Velocity $\langle \dot{u} \rangle$')
            fig.savefig(output_dir / "time_series_u_dot.pdf")
            plt.close(fig)

            # 1) Time Fourier Transform of mean_udot
            if len(time) > 1:
                dt_ts = time[1] - time[0]
                N = len(time)
                
                # Subtract mean to remove the large DC component at omega=0
                u_dot_centered = u_dot - np.mean(u_dot)
                
                # Compute FFT and angular frequencies
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

    # Transient Metrics
    trans_file = data_dir / f"{prefix}_transient_metrics.txt"
    sum_file = data_dir / f"{prefix}_final_summary.txt"
    final_vals = None
    if sum_file.exists():
        try:
            s_data = np.genfromtxt(sum_file, names=True)
            if s_data.size > 0:
                if s_data.ndim > 0 and 'batch' in s_data.dtype.names:
                    s_data = s_data[s_data['batch'] == 0]
                row = s_data[0] if s_data.ndim > 0 else s_data
                final_vals = {
                    'var_u': row['var_u'] if 'var_u' in row.dtype.names else float(row[1]),
                    'u_dot': row['u_dot'] if 'u_dot' in row.dtype.names else float(row[2]),
                    'phi_dot': row['phi_dot'] if 'phi_dot' in row.dtype.names else float(row[3]),
                    'entropy': row['spectral_entropy'] if 'spectral_entropy' in row.dtype.names else float(row[4])
                }
        except Exception as e:
            print(f"Could not parse final summary: {e}")

    if trans_file.exists():
        data = np.genfromtxt(trans_file, names=True)
        if data.size > 0:
            if data.ndim == 0: data = data.reshape(1)
            if 'batch' in data.dtype.names:
                data = data[data['batch'] == 0]
            t = data['time'] if 'time' in data.dtype.names else data[data.dtype.names[1]]
            var = data['avg_var_u'] if 'avg_var_u' in data.dtype.names else data[data.dtype.names[2]]
            u_dot_trans = data['avg_u_dot'] if 'avg_u_dot' in data.dtype.names else data[data.dtype.names[3]]
            phi_dot = data['avg_phi_dot'] if 'avg_phi_dot' in data.dtype.names else data[data.dtype.names[4]]
            ent = data['avg_entropy'] if 'avg_entropy' in data.dtype.names else data[data.dtype.names[5]]

            plots = [
                ("transient_variance.pdf", var, r'Variance $\sigma_u^2$', 'tab:blue', 'var_u'),
                ("transient_u_dot.pdf", u_dot_trans, r'Avg Velocity $\langle \dot{u} \rangle$', 'tab:orange', 'u_dot'),
                ("transient_phi_dot.pdf", phi_dot, r'Avg Phase Vel. $\langle \dot{\phi} \rangle$', 'tab:green', 'phi_dot'),
                ("transient_entropy.pdf", ent, r'Spectral Entropy $S$', 'tab:purple', 'entropy')
            ]

            for fname, ydata, ylabel, color, key in plots:
                fig, ax = plt.subplots()
                ax.plot(t, ydata, marker='o', markersize=3, linestyle='-', color=color, label='Transient')
                if final_vals:
                    ax.axhline(final_vals[key], color='black', linestyle='--', linewidth=1, label='Final Avg')
                    ax.legend()
                ax.set_xscale('log')
                ax.set_xlabel(r'Time $t$')
                ax.set_ylabel(ylabel)
                fig.savefig(output_dir / fname)
                plt.close(fig)

    # Power Spectrum
    ps_file = data_dir / f"{prefix}_final_ps.txt"
    if ps_file.exists():
        data = np.genfromtxt(ps_file, names=True)
        if data.size > 0:
            if data.ndim == 0: data = data.reshape(1)
            if 'batch' in data.dtype.names:
                data = data[data['batch'] == 0]
            k = data['k'] if 'k' in data.dtype.names else data[data.dtype.names[0]]
            power = data['power'] if 'power' in data.dtype.names else data[data.dtype.names[1]]
            mask = k > 0
            k, power = k[mask], power[mask]
            fig, ax = plt.subplots()
            ax.loglog(k, power, color='black', linewidth=1.2)
            ax.set_xlabel(r'Wavenumber $k$')
            ax.set_ylabel(r'Structure Factor $S(k)$')
            fig.savefig(output_dir / "power_spectrum.pdf")
            plt.close(fig)

    # Height Histogram
    hist_file = data_dir / f"{prefix}_final_height_hist.txt"
    if hist_file.exists():
        data = np.genfromtxt(hist_file, names=True)
        if data.size > 0:
            if data.ndim == 0: data = data.reshape(1)
            if 'batch' in data.dtype.names:
                data = data[data['batch'] == 0]
            bin_center = data['bin_center'] if 'bin_center' in data.dtype.names else data[data.dtype.names[0]]
            pdf = data['probability_density'] if 'probability_density' in data.dtype.names else data[data.dtype.names[-1]]
            fig, ax = plt.subplots()
            ax.plot(bin_center, pdf, color='tab:blue', linestyle='-', linewidth=1.2)
            ax.fill_between(bin_center, pdf, color='tab:blue', alpha=0.3)
            std_normal = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * bin_center**2)
            ax.plot(bin_center, std_normal, color='black', linestyle='--', linewidth=1.0, label='Gaussian')
            ax.set_xlabel(r'Normalized Height $(u - \langle u \rangle)/\sigma_u$')
            ax.set_ylabel(r'Probability Density $P$')
            ax.legend()
            fig.savefig(output_dir / "final_height_hist.pdf")
            plt.close(fig)

def plot_scaling_analysis(scaling_data, output_dir):
    L = np.array([d['L'] for d in scaling_data])
    var_u = np.array([d['var_u'] for d in scaling_data])
    u_dot = np.array([d['u_dot'] for d in scaling_data])
    phi_dot = np.array([d['phi_dot'] for d in scaling_data])
    entropy = np.array([d['entropy'] for d in scaling_data])
    k_max = np.array([d['k_max'] for d in scaling_data])
    s_max = np.array([d['s_max'] for d in scaling_data])

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Variance vs L
    fig, ax = plt.subplots()
    ax.loglog(L, var_u, marker='o', color='tab:blue', linestyle='-')
    ax.set_xlabel(r'System Length $L$')
    ax.set_ylabel(r'Variance $\sigma_u^2$')
    fig.savefig(output_dir / "scaling_variance.pdf")
    plt.close(fig)

    # 2. Velocity vs L
    fig, ax = plt.subplots()
    ax.semilogx(L, u_dot, marker='s', color='tab:red', linestyle='-')
    ax.set_xlabel(r'System Length $L$')
    ax.set_ylabel(r'Velocity $\langle \dot{u} \rangle$')
    fig.savefig(output_dir / "scaling_u_dot.pdf")
    plt.close(fig)

    # 3. Peak Wavenumber (k_max) vs L
    fig, ax = plt.subplots()
    valid = k_max > 0
    ax.loglog(L[valid], k_max[valid], marker='^', color='tab:green', linestyle='-', zorder=3)
    
    if np.any(valid):
        ref_L = L[valid]
        ax.loglog(ref_L, 2 * np.pi / ref_L, color='black', linestyle='--', linewidth=1, label=r'$2\pi/L$', zorder=3)
    
    # 2) Instability markers and bounds
    ax.axhline(0.833375, color='tab:red', linestyle=':', linewidth=1.5, label='Max Floquet', zorder=2)

    # Apply 90% min buffer to X-axis
    if len(L) > 0:
        ax.set_xlim(left=0.9 * k_max[valid].min())

    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\kappa_{max}$')
    
    # Place legend outside or find best location to avoid covering data
    ax.legend(loc='best')
    fig.savefig(output_dir / "scaling_k_max.pdf")
    plt.close(fig)

    # 4. Peak Structure Factor S(k_max) vs L
    fig, ax = plt.subplots()
    ax.loglog(L[valid], s_max[valid], marker='v', color='tab:orange', linestyle='-')
    ax.set_xlabel(r'System Length $L$')
    ax.set_ylabel(r'Peak Structure Factor $S(k_{max})$')
    fig.savefig(output_dir / "scaling_s_max.pdf")
    plt.close(fig)

    # 5. Overlaid Power Spectra
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    log_L = np.log2(L)
    norm = plt.Normalize(vmin=log_L.min(), vmax=log_L.max())
    
    for d in scaling_data:
        k = d['spectra_k']
        s = d['spectra_s']
        if len(k) > 0:
            c = cmap(norm(np.log2(d['L'])))
            ax.loglog(k, s, color=c, alpha=0.7, linewidth=0.8)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$\log_2(L)$')
    
    ax.set_xlabel(r'Wavenumber $k$')
    ax.set_ylabel(r'Structure Factor $S(k)$')
    fig.savefig(output_dir / "scaling_spectra_overlay.pdf")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Process scaling array output.")
    parser.add_argument("base_dir", type=str, help="Path to the scaling base directory (e.g., output/scaling).")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.is_dir():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    setup_prb_style()
    scaling_data = []

    print(f"Scanning {base_dir} for dataset folders...")
    for data_dir in base_dir.glob("main_SC_scaling_*"):
        if not data_dir.is_dir():
            continue
        
        try:
            prefix = get_prefix(data_dir)
        except FileNotFoundError:
            continue

        params = read_parameters(data_dir)
        L = float(params.get("L", 0))
        if L == 0:
            print(f"Could not determine L for {data_dir.name}, skipping.")
            continue

        print(f"Processing L = {L} ...")
        
        # 1. Generate individual plots inside the dataset directory
        indiv_fig_dir = data_dir / "figures"
        indiv_fig_dir.mkdir(exist_ok=True)
        plot_individual_system(data_dir, indiv_fig_dir, prefix)

        # 2. Extract metrics for scaling analysis
        sys_data = {'L': L, 'var_u': np.nan, 'u_dot': np.nan, 'phi_dot': np.nan, 'entropy': np.nan, 'k_max': np.nan, 's_max': np.nan, 'spectra_k': [], 'spectra_s': []}
        
        sum_file = data_dir / f"{prefix}_final_summary.txt"
        if sum_file.exists():
            try:
                s_data = np.genfromtxt(sum_file, names=True)
                if s_data.size > 0:
                    if s_data.ndim > 0 and 'batch' in s_data.dtype.names:
                        s_data = s_data[s_data['batch'] == 0]
                    row = s_data[0] if s_data.ndim > 0 else s_data
                    sys_data['var_u'] = row['var_u'] if 'var_u' in row.dtype.names else float(row[1])
                    sys_data['u_dot'] = row['u_dot'] if 'u_dot' in row.dtype.names else float(row[2])
                    sys_data['phi_dot'] = row['phi_dot'] if 'phi_dot' in row.dtype.names else float(row[3])
                    sys_data['entropy'] = row['spectral_entropy'] if 'spectral_entropy' in row.dtype.names else float(row[4])
            except Exception as e:
                print(f"Could not parse final summary for scaling: {e}")

        ps_file = data_dir / f"{prefix}_final_ps.txt"
        if ps_file.exists():
            try:
                p_data = np.genfromtxt(ps_file, names=True)
                if p_data.size > 0:
                    if p_data.ndim > 0 and 'batch' in p_data.dtype.names:
                        p_data = p_data[p_data['batch'] == 0]
                    k = p_data['k'] if 'k' in p_data.dtype.names else p_data[p_data.dtype.names[0]]
                    power = p_data['power'] if 'power' in p_data.dtype.names else p_data[p_data.dtype.names[1]]
                    mask = k > 0
                    k = k[mask]
                    power = power[mask]
                    sys_data['spectra_k'] = k
                    sys_data['spectra_s'] = power
                    if len(k) > 0:
                        max_idx = np.argmax(power)
                        sys_data['k_max'] = k[max_idx]
                        sys_data['s_max'] = power[max_idx]
            except Exception as e:
                print(f"Could not parse final ps for scaling: {e}")

        scaling_data.append(sys_data)

    if not scaling_data:
        print("No complete datasets found for scaling analysis.")
        return

    # Sort data by system size L
    scaling_data.sort(key=lambda x: x['L'])

    # 3. Generate global scaling plots
    print("Generating global scaling plots...")
    scaling_fig_dir = base_dir / "figures_scaling"
    plot_scaling_analysis(scaling_data, scaling_fig_dir)
    print(f"All done! Scaling plots saved to {scaling_fig_dir}")

if __name__ == "__main__":
    main()