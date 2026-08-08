import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.collections import LineCollection

# =============================================================================
# CUSTOMIZATION VARIABLES
# =============================================================================

# --- General ---
SUMMARY_FILE_SUFFIX = "_time_summary.txt"
SPATIAL_PS_DIR = "time_power_spectra"
TEMPORAL_PS_DIR = "time_power_spectra"
REAL_STATE_DIR = "real_state_plots"
HISTOGRAM_DIR = "u_histograms"

# --- New Instantaneous & Transient Settings ---
TRANSIENT_FILE_SUFFIX = "_transient_w2.txt"
INST_SPATIAL_PS_DIR = "inst_power_spectra"
INST_HISTOGRAM_DIR = "inst_u_histograms"
VIDEO_INST_SPATIAL_NAME = "inst_spatial_evolution"
VIDEO_INST_HIST_NAME = "inst_histogram_evolution"

# --- Performance Settings ---
MAX_PLOT_POINTS = 4096 # Aggressively downsample large arrays for memory/rendering speed

# --- Video Settings ---
MAKE_VIDEO = True
VIDEO_FPS = 4
VIDEO_FORMAT = "mp4" # Set to "gif" if you don't have ffmpeg installed
VIDEO_SPATIAL_NAME = "spatial_evolution"
VIDEO_TEMPORAL_NAME = "temporal_evolution"
VIDEO_REAL_NAME = "real_state_evolution"
VIDEO_HIST_NAME = "histogram_evolution"
FIX_AXES_FOR_VIDEO = True # Locks the X and Y bounds so the frame doesn't jump

# --- Plot Titles ---
TITLE_ENTROPY = "Spectral Entropy vs Time"
TITLE_VARIANCE = "Spatial Variance vs Time"
TITLE_U_DOT = "Average u_dot vs Time"
TITLE_PHI_DOT = "Average phi_dot vs Time"
TITLE_SPATIAL_PS = "Spatial Power Spectrum at step = {step}"
TITLE_TEMPORAL_PS = "Time Power Spectrum of u_dot at step = {step}"

# --- Axis Labels ---
LABEL_TIME = "Total Steps"
LABEL_ENTROPY = "Spectral Entropy (log scale)"
LABEL_VARIANCE = "Spatial Variance (log scale)"
LABEL_U_DOT = "Average u velocity (u_dot)"
LABEL_PHI_DOT = "Average phi velocity (phi_dot)"
LABEL_SPATIAL_K = "Wavenumber (k)"
LABEL_SPATIAL_POWER = "Power"
LABEL_TEMPORAL_W = "Temporal Frequency (omega)"
LABEL_TEMPORAL_POWER = "Power"

# --- Reference Power Law Line Settings ---
LABEL_REF_LINE = "k^-2 Power Law"
LABEL_REF_LINE_TEMPORAL = "omega^-2 Power Law"
COLOR_REF_LINE = "red"
STYLE_REF_LINE = "--"

POWER_LAW_OFFSET_SPATIAL = 2.0
POWER_LAW_OFFSET_TEMPORAL = 0.01

# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_time.py <base_prefix> <k_cutoff>")
        sys.exit(1)

    # 1. Resolve absolute path of the input data BEFORE changing directories
    raw_prefix = sys.argv[1]
    base_prefix = os.path.abspath(raw_prefix)
    
    # 2. Change the working directory to where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    k_cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    summary_file = f"{base_prefix}{SUMMARY_FILE_SUFFIX}"

    if not os.path.exists(summary_file):
        print(f"Error: Summary file {summary_file} not found.")
        sys.exit(1)

    # 3. Determine output figures directory based on data directory name
    data_dir = os.path.dirname(base_prefix)
    data_dir_name = os.path.basename(data_dir) # e.g., "time_evo_100000.0(1)"
    figures_dir = os.path.abspath(os.path.join("..", "figures", data_dir_name))
    
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving figures and videos to: {figures_dir}")

    try:
        # Use fast C engine and exact space separator
        df_summary = pd.read_csv(summary_file, sep=' ', engine='c')
    except pd.errors.EmptyDataError:
        print("The summary file is empty. Please wait for the first point to finish.")
        sys.exit(0)

    if df_summary.empty:
        print("The summary file has a header but no data yet. Please wait.")
        sys.exit(0)

    df_summary = df_summary.sort_values(by='total_steps').reset_index(drop=True)
    print(f"Loaded {len(df_summary)} time step evaluations.")

    # 1. Plot Summary Metrics
    def plot_summary(y_col, ylabel, title, log_y=False, marker='o', color='b'):
        plt.figure(figsize=(8, 6))
        plt.plot(df_summary['total_steps'], df_summary[y_col], marker=marker, linestyle='-', color=color)
        plt.xscale('log', base=2)
        if log_y:
            plt.yscale('log')
        plt.xlabel(LABEL_TIME)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"time_{y_col}.png"))
        plt.close()

    plot_summary('spectral_entropy', LABEL_ENTROPY, TITLE_ENTROPY, log_y=True, marker='o', color='b')
    plot_summary('var_u', LABEL_VARIANCE, TITLE_VARIANCE, log_y=True, marker='s', color='r')
    plot_summary('u_dot', LABEL_U_DOT, TITLE_U_DOT, log_y=False, marker='^', color='g')
    plot_summary('phi_dot', LABEL_PHI_DOT, TITLE_PHI_DOT, log_y=False, marker='v', color='m')

    spatial_data = {}
    temporal_data = {}
    real_state_data = {}
    hist_data = {}
    inst_spatial_data = {}
    inst_hist_data = {}

    # Track global limits for efficient video generation
    lims = {
        'k_min': float('inf'), 'k_max': float('-inf'),
        'p_min': float('inf'), 'p_max': float('-inf'),
        'w_min': float('inf'), 'w_max': float('-inf'),
        'wp_min': float('inf'), 'wp_max': float('-inf'),
        'x_min': float('inf'), 'x_max': float('-inf'),
        'u_min': float('inf'), 'u_max': float('-inf'),
        'pdf_max': 0.5
    }

    # Create subdirectories inside the new figures directory
    spatial_ps_out_dir = os.path.join(figures_dir, SPATIAL_PS_DIR)
    temporal_ps_out_dir = os.path.join(figures_dir, TEMPORAL_PS_DIR)
    real_state_out_dir = os.path.join(figures_dir, REAL_STATE_DIR)
    hist_out_dir = os.path.join(figures_dir, HISTOGRAM_DIR)
    inst_spatial_ps_out_dir = os.path.join(figures_dir, INST_SPATIAL_PS_DIR)
    inst_hist_out_dir = os.path.join(figures_dir, INST_HISTOGRAM_DIR)

    os.makedirs(spatial_ps_out_dir, exist_ok=True)
    os.makedirs(temporal_ps_out_dir, exist_ok=True)
    os.makedirs(real_state_out_dir, exist_ok=True)
    os.makedirs(hist_out_dir, exist_ok=True)
    os.makedirs(inst_spatial_ps_out_dir, exist_ok=True)
    os.makedirs(inst_hist_out_dir, exist_ok=True)

    os.makedirs(spatial_ps_out_dir, exist_ok=True)
    os.makedirs(temporal_ps_out_dir, exist_ok=True)
    os.makedirs(real_state_out_dir, exist_ok=True)
    os.makedirs(hist_out_dir, exist_ok=True)
    
    # --- Process Transient Variance (1D EW t^(1/2) Scaling) ---
    transient_file = f"{base_prefix}{TRANSIENT_FILE_SUFFIX}"
    if os.path.exists(transient_file):
        try:
            df_trans = pd.read_csv(transient_file, sep=' ', engine='c')
            if not df_trans.empty and len(df_trans) > 1:
                df_trans = df_trans[df_trans['time'] > 0]
                plt.figure(figsize=(8, 6))
                plt.loglog(df_trans['time'], df_trans['var_u'], color='r', label="Simulated Variance (W^2)")
                
                t_vals = df_trans['time'].values
                var_vals = df_trans['var_u'].values
                if len(t_vals) > 0 and var_vals[-1] > 0:
                    # Align the reference slope with the final simulated point
                    c_t = var_vals[-1] / (t_vals[-1]**0.5)
                    plt.loglog(t_vals, c_t * (t_vals**0.5), color='k', linestyle='--', label="t^(1/2) EW Scaling")
                
                plt.xlabel("Time (t)")
                plt.ylabel("Spatial Variance W^2")
                plt.title("Transient Spatial Variance vs Time (Log-Log)")
                plt.legend()
                plt.grid(True, which='both', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, "transient_variance_scaling.png"))
                plt.close()
        except Exception as e:
            print(f"Skipping transient plot: {e}")

    # 2. Extract and Plot Individual Power Spectra
    for step in df_summary['total_steps']:
        step_int = int(step)

        # --- Process Spatial Power Spectrum ---
        ps_file = f"{base_prefix}_ps_step_{step_int}.txt"
        if os.path.exists(ps_file):
            df_ps = pd.read_csv(ps_file, sep=' ', engine='c')
            df_ps = df_ps[(df_ps['k'] > 0) & (df_ps['k'] <= k_cutoff) & (df_ps['power'] > 0)]
            df_ps = df_ps.sort_values(by='k')

            if not df_ps.empty:
                k_vals = df_ps['k'].values
                power_vals = df_ps['power'].values
                spatial_data[step_int] = (k_vals, power_vals)

                lims['k_min'] = min(lims['k_min'], k_vals.min())
                lims['k_max'] = max(lims['k_max'], k_vals.max())
                lims['p_min'] = min(lims['p_min'], power_vals.min())
                lims['p_max'] = max(lims['p_max'], power_vals.max())

                plt.figure(figsize=(8, 6))
                plt.loglog(k_vals, power_vals, color='k', alpha=0.8, label="Data")
                
                c = (power_vals[0] * k_vals[0]**2) * POWER_LAW_OFFSET_SPATIAL
                ref_power = c * (k_vals**-2)
                plt.loglog(k_vals, ref_power, color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
                
                plt.xlabel(LABEL_SPATIAL_K)
                plt.ylabel(LABEL_SPATIAL_POWER)
                
                title_str = TITLE_SPATIAL_PS.format(step=step_int)
                if k_cutoff != float('inf'):
                    title_str += f' (k cutoff = {k_cutoff})'
                plt.title(title_str)
                plt.legend()
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(spatial_ps_out_dir, f'ps_step_{step_int}.png'))
                plt.close()

        # --- Process Temporal Power Spectrum ---
        ts_file = f"{base_prefix}_mean_u_ts_step_{step_int}.txt"
        if os.path.exists(ts_file):
            df_ts = pd.read_csv(ts_file, sep=' ', engine='c')
            
            if 'mean_u' in df_ts.columns and len(df_ts) > 1:
                dt_step = df_ts['time'].iloc[1] - df_ts['time'].iloc[0]
                n_samples = len(df_ts)
                
                window = np.hanning(n_samples)
                fft_vals = np.fft.rfft(df_ts['mean_u'].values * window)
                
                fft_freqs_hz = np.fft.rfftfreq(n_samples, d=dt_step)
                omega = 2.0 * np.pi * fft_freqs_hz
                
                P_omega = ((np.abs(fft_vals) / n_samples)**2 * (8.0 / 3.0)) * (omega**2)
                
                valid_idx = (omega > 0) & (P_omega > 0)
                omega = omega[valid_idx]
                P_omega = P_omega[valid_idx]
                
                if len(omega) > 0:
                    temporal_data[step_int] = (omega, P_omega)

                    lims['w_min'] = min(lims['w_min'], omega.min())
                    lims['w_max'] = max(lims['w_max'], omega.max())
                    lims['wp_min'] = min(lims['wp_min'], P_omega.min())
                    lims['wp_max'] = max(lims['wp_max'], P_omega.max())

                    plt.figure(figsize=(8, 6))
                    plt.loglog(omega, P_omega, color='purple', alpha=0.8, label="Data")
                    
                    c_temp = (P_omega[0] * omega[0]**2) * POWER_LAW_OFFSET_TEMPORAL
                    ref_P_omega = c_temp * (omega**-2)
                    plt.loglog(omega, ref_P_omega, color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE_TEMPORAL)

                    plt.xlabel(LABEL_TEMPORAL_W)
                    plt.ylabel(LABEL_TEMPORAL_POWER)
                    plt.title(TITLE_TEMPORAL_PS.format(step=step_int))
                    plt.legend()
                    plt.grid(True, which='both', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(temporal_ps_out_dir, f'time_ps_u_dot_step_{step_int}.png'))
                    plt.close()

        # --- Process Real State ---
        state_file = f"{base_prefix}_real_state_step_{step_int}.txt"
        if os.path.exists(state_file):
            df_state = pd.read_csv(state_file, sep=' ', engine='c')
            if not df_state.empty:
                x = df_state['x'].values
                u_real = df_state['Re(u)'].values
                u_imag = df_state['Im(u)'].values
                
                # Downsample large arrays to save memory and speed up Matplotlib
                if len(x) > MAX_PLOT_POINTS:
                    stride = max(1, len(x) // MAX_PLOT_POINTS)
                    x = x[::stride]
                    u_real = u_real[::stride]
                    u_imag = u_imag[::stride]
                
                u_centered = u_real - np.mean(u_real)
                phase = np.arctan2(u_imag, u_centered)
                
                real_state_data[step_int] = (x, u_centered, phase)

                lims['x_min'] = min(lims['x_min'], x.min())
                lims['x_max'] = max(lims['x_max'], x.max())
                lims['u_min'] = min(lims['u_min'], u_centered.min())
                lims['u_max'] = max(lims['u_max'], u_centered.max())
                
                points = np.array([x, u_centered]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                norm = plt.Normalize(-np.pi, np.pi)
                lc = LineCollection(segments, cmap='hsv', norm=norm)
                lc.set_array(phase[:-1])
                lc.set_linewidth(1.5)
                
                ax.add_collection(lc)
                ax.set_xlim(x.min(), x.max())
                
                y_margin = max(np.abs(u_centered.min()), np.abs(u_centered.max())) * 0.1
                y_margin = 0.1 if y_margin == 0 else y_margin
                ax.set_ylim(u_centered.min() - y_margin, u_centered.max() + y_margin)
                
                cbar = fig.colorbar(lc, ax=ax, ticks=[-np.pi, 0, np.pi])
                cbar.ax.set_yticklabels([r'$-\pi$', '0', '$\pi$'])
                cbar.set_label('Phase')
                
                ax.set_xlabel('x')
                ax.set_ylabel('Re(u) - mean(Re(u))')
                ax.set_title(f'Real Space State at step = {step_int}')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(real_state_out_dir, f'real_state_step_{step_int}.png'))
                plt.close(fig)
                
                

        # --- Process Pre-computed Histogram ---
        hist_file = f"{base_prefix}_height_hist_step_{step_int}.txt"
        if os.path.exists(hist_file):
            df_hist = pd.read_csv(hist_file, sep=' ', engine='c')
            if not df_hist.empty:
                bin_centers = df_hist['bin_center'].values
                pdf = df_hist['probability_density'].values
                hist_data[step_int] = (bin_centers, pdf)

                lims['pdf_max'] = max(lims['pdf_max'], pdf.max() * 1.1)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
                ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Simulation Data', align='center')
                
                x_gauss = np.linspace(-5, 5, 200)
                y_gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)
                ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
                
                ax.set_xlim(-5, 5)
                ax.set_xlabel('(Re(u) - mean) / std')
                ax.set_ylabel('Density')
                ax.set_title(f'Normalized Histogram of Re(u) at step = {step_int}')
                ax.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(hist_out_dir, f'hist_step_{step_int}.png'))
                plt.close(fig)
        # --- Process Instantaneous Spatial Power Spectrum ---
        inst_ps_file = f"{base_prefix}_inst_ps_step_{step_int}.txt"
        if os.path.exists(inst_ps_file):
            df_inst_ps = pd.read_csv(inst_ps_file, sep=' ', engine='c')
            df_inst_ps = df_inst_ps[(df_inst_ps['k'] > 0) & (df_inst_ps['k'] <= k_cutoff) & (df_inst_ps['power'] > 0)]
            df_inst_ps = df_inst_ps.sort_values(by='k')

            if not df_inst_ps.empty:
                k_vals = df_inst_ps['k'].values
                power_vals = df_inst_ps['power'].values
                inst_spatial_data[step_int] = (k_vals, power_vals)

                plt.figure(figsize=(8, 6))
                plt.loglog(k_vals, power_vals, color='k', alpha=0.8, label="Instantaneous Data")
                
                c = (power_vals[0] * k_vals[0]**2) * POWER_LAW_OFFSET_SPATIAL
                ref_power = c * (k_vals**-2)
                plt.loglog(k_vals, ref_power, color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
                
                plt.xlabel(LABEL_SPATIAL_K)
                plt.ylabel(LABEL_SPATIAL_POWER)
                
                title_str = f"Instantaneous Spatial PS at step = {step_int}"
                if k_cutoff != float('inf'):
                    title_str += f' (k cutoff = {k_cutoff})'
                plt.title(title_str)
                plt.legend()
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(inst_spatial_ps_out_dir, f'inst_ps_step_{step_int}.png'))
                plt.close()

        # --- Process Instantaneous Histogram ---
        inst_hist_file = f"{base_prefix}_inst_height_hist_step_{step_int}.txt"
        if os.path.exists(inst_hist_file):
            df_inst_hist = pd.read_csv(inst_hist_file, sep=' ', engine='c')
            if not df_inst_hist.empty:
                bin_centers = df_inst_hist['bin_center'].values
                pdf = df_inst_hist['probability_density'].values
                inst_hist_data[step_int] = (bin_centers, pdf)

                fig, ax = plt.subplots(figsize=(8, 6))
                width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
                ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Instantaneous Data', align='center')
                
                x_gauss = np.linspace(-5, 5, 200)
                y_gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)
                ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
                
                ax.set_xlim(-5, 5)
                ax.set_xlabel('(Re(u) - mean) / std')
                ax.set_ylabel('Density')
                ax.set_title(f'Instantaneous Histogram of Re(u) at step = {step_int}')
                ax.legend(loc='upper right')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(inst_hist_out_dir, f'inst_hist_step_{step_int}.png'))
                plt.close(fig)

    print(f"Saved summary plots and individual step plots.")

    # 3. Generate Videos
    if not MAKE_VIDEO:
        return

    print(f"Generating videos in {VIDEO_FORMAT} format...")

    # -- Spatial Video --
    if spatial_data:
        valid_steps = sorted(list(spatial_data.keys()))
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.loglog([], [], color='k', alpha=0.8, label="Data")
        ref_line, = ax.loglog([], [], color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
        
        if FIX_AXES_FOR_VIDEO:
            ax.set_xlim(lims['k_min'], lims['k_max'])
            ax.set_ylim(lims['p_min'] * 0.1, lims['p_max'] * 10.0)

        ax.set_xlabel(LABEL_SPATIAL_K)
        ax.set_ylabel(LABEL_SPATIAL_POWER)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        def update_spat(frame):
            step_int = valid_steps[frame]
            k, p = spatial_data[step_int]
            line.set_data(k, p)
            c = (p[0] * k[0]**2) * POWER_LAW_OFFSET_SPATIAL
            ref_line.set_data(k, c * (k**-2))
            
            title_str = TITLE_SPATIAL_PS.format(step=step_int)
            if k_cutoff != float('inf'):
                title_str += f' (k cutoff = {k_cutoff})'
            ax.set_title(title_str)
            return line, ref_line

        ani = animation.FuncAnimation(fig, update_spat, frames=len(valid_steps), blit=False)
        out_file = os.path.join(figures_dir, f"{VIDEO_SPATIAL_NAME}.{VIDEO_FORMAT}")
        writer = 'ffmpeg' if VIDEO_FORMAT == "mp4" else 'pillow'
        ani.save(out_file, writer=writer, fps=VIDEO_FPS)
        plt.close(fig)
        print(f"Saved {out_file}")

    # -- Temporal Video --
    if temporal_data:
        valid_steps = sorted(list(temporal_data.keys()))
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.loglog([], [], color='purple', alpha=0.8, label="Data")
        ref_line, = ax.loglog([], [], color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE_TEMPORAL)
        
        if FIX_AXES_FOR_VIDEO:
            ax.set_xlim(lims['w_min'], lims['w_max'])
            ax.set_ylim(lims['wp_min'] * 0.1, lims['wp_max'] * 10.0)

        ax.set_xlabel(LABEL_TEMPORAL_W)
        ax.set_ylabel(LABEL_TEMPORAL_POWER)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        def update_temp(frame):
            step_int = valid_steps[frame]
            w, p = temporal_data[step_int]
            line.set_data(w, p)
            c = (p[0] * w[0]**2) * POWER_LAW_OFFSET_TEMPORAL
            ref_line.set_data(w, c * (w**-2))
            ax.set_title(TITLE_TEMPORAL_PS.format(step=step_int))
            return line, ref_line

        ani = animation.FuncAnimation(fig, update_temp, frames=len(valid_steps), blit=False)
        out_file = os.path.join(figures_dir, f"{VIDEO_TEMPORAL_NAME}.{VIDEO_FORMAT}")
        writer = 'ffmpeg' if VIDEO_FORMAT == "mp4" else 'pillow'
        ani.save(out_file, writer=writer, fps=VIDEO_FPS)
        plt.close(fig)
        print(f"Saved {out_file}")

    # -- Real State Video --
    if real_state_data:
        valid_steps = sorted(list(real_state_data.keys()))
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if FIX_AXES_FOR_VIDEO:
            margin = max(abs(lims['u_min']), abs(lims['u_max'])) * 0.1
            margin = 0.1 if margin == 0 else margin
            ax.set_xlim(lims['x_min'], lims['x_max'])
            ax.set_ylim(lims['u_min'] - margin, lims['u_max'] + margin)
        
        norm = plt.Normalize(-np.pi, np.pi)
        lc = LineCollection([], cmap='hsv', norm=norm)
        ax.add_collection(lc)
        
        cbar = fig.colorbar(lc, ax=ax, ticks=[-np.pi, 0, np.pi])
        cbar.ax.set_yticklabels([r'$-\pi$', '0', '$\pi$'])
        cbar.set_label('Phase')
        
        ax.set_xlabel('x')
        ax.set_ylabel('Re(u) - mean(Re(u))')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        def update_real(frame):
            step_int = valid_steps[frame]
            x, u_c, phase = real_state_data[step_int]
            points = np.array([x, u_c]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(phase[:-1])
            ax.set_title(f'Real Space State at step = {step_int}')
            
            if not FIX_AXES_FOR_VIDEO:
                ax.set_xlim(x.min(), x.max())
                y_margin = max(abs(u_c.min()), abs(u_c.max())) * 0.1
                ax.set_ylim(u_c.min() - y_margin, u_c.max() + y_margin)
                
            return lc,
            
        ani = animation.FuncAnimation(fig, update_real, frames=len(valid_steps), blit=False)
        out_file = os.path.join(figures_dir, f"{VIDEO_REAL_NAME}.{VIDEO_FORMAT}")
        writer = 'ffmpeg' if VIDEO_FORMAT == "mp4" else 'pillow'
        ani.save(out_file, writer=writer, fps=VIDEO_FPS)
        plt.close(fig)
        print(f"Saved {out_file}")

    # -- Histogram Video --
    if hist_data:
        valid_steps = sorted(list(hist_data.keys()))
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_gauss = np.linspace(-5, 5, 200)
        y_gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)
        
        def update_hist(frame):
            ax.clear()
            step_int = valid_steps[frame]
            bin_centers, pdf = hist_data[step_int]
            
            width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
            ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Simulation Data', align='center')
            ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
            
            ax.set_xlim(-5, 5)
            y_max = lims['pdf_max'] if FIX_AXES_FOR_VIDEO else max(0.5, pdf.max() * 1.1)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('(Re(u) - mean) / std')
            ax.set_ylabel('Density')
            ax.set_title(f'Normalized Histogram of Re(u) at step = {step_int}')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        ani = animation.FuncAnimation(fig, update_hist, frames=len(valid_steps), blit=False)
        out_file = os.path.join(figures_dir, f"{VIDEO_HIST_NAME}.{VIDEO_FORMAT}")
        writer = 'ffmpeg' if VIDEO_FORMAT == "mp4" else 'pillow'
        ani.save(out_file, writer=writer, fps=VIDEO_FPS)
        plt.close(fig)
        print(f"Saved {out_file}")
        
    # -- Instantaneous Spatial Video --
    if inst_spatial_data:
        valid_steps = sorted(list(inst_spatial_data.keys()))
        fig, ax = plt.subplots(figsize=(8, 6))
        line, = ax.loglog([], [], color='k', alpha=0.8, label="Instantaneous Data")
        ref_line, = ax.loglog([], [], color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
        
        if FIX_AXES_FOR_VIDEO:
            ax.set_xlim(lims['k_min'], lims['k_max'])
            ax.set_ylim(lims['p_min'] * 0.1, lims['p_max'] * 10.0)

        ax.set_xlabel(LABEL_SPATIAL_K)
        ax.set_ylabel(LABEL_SPATIAL_POWER)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        def update_inst_spat(frame):
            step_int = valid_steps[frame]
            k, p = inst_spatial_data[step_int]
            line.set_data(k, p)
            c = (p[0] * k[0]**2) * POWER_LAW_OFFSET_SPATIAL
            ref_line.set_data(k, c * (k**-2))
            
            title_str = f"Instantaneous Spatial PS at step = {step_int}"
            if k_cutoff != float('inf'):
                title_str += f' (k cutoff = {k_cutoff})'
            ax.set_title(title_str)
            return line, ref_line

        ani = animation.FuncAnimation(fig, update_inst_spat, frames=len(valid_steps), blit=False)
        out_file = os.path.join(figures_dir, f"{VIDEO_INST_SPATIAL_NAME}.{VIDEO_FORMAT}")
        writer = 'ffmpeg' if VIDEO_FORMAT == "mp4" else 'pillow'
        ani.save(out_file, writer=writer, fps=VIDEO_FPS)
        plt.close(fig)
        print(f"Saved {out_file}")

    # -- Instantaneous Histogram Video --
    if inst_hist_data:
        valid_steps = sorted(list(inst_hist_data.keys()))
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_gauss = np.linspace(-5, 5, 200)
        y_gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)
        
        def update_inst_hist(frame):
            ax.clear()
            step_int = valid_steps[frame]
            bin_centers, pdf = inst_hist_data[step_int]
            
            width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
            ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Instantaneous Data', align='center')
            ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
            
            ax.set_xlim(-5, 5)
            y_max = lims['pdf_max'] if FIX_AXES_FOR_VIDEO else max(0.5, pdf.max() * 1.1)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('(Re(u) - mean) / std')
            ax.set_ylabel('Density')
            ax.set_title(f'Instantaneous Histogram of Re(u) at step = {step_int}')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        ani = animation.FuncAnimation(fig, update_inst_hist, frames=len(valid_steps), blit=False)
        out_file = os.path.join(figures_dir, f"{VIDEO_INST_HIST_NAME}.{VIDEO_FORMAT}")
        writer = 'ffmpeg' if VIDEO_FORMAT == "mp4" else 'pillow'
        ani.save(out_file, writer=writer, fps=VIDEO_FPS)
        plt.close(fig)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    main()