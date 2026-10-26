import sys
import os
import gc
import concurrent.futures
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection

# Turn off interactive plotting to save overhead
plt.ioff()

# =============================================================================
# CUSTOMIZATION VARIABLES
# =============================================================================

# --- BUG FIX TOGGLE ---
# Set to True to apply the N^2 * L correction to old data that suffers from the 
# GPU double-normalization and missing continuous scale bugs.
APPLY_AFTER_THE_FACT_FIX = True 

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
POWER_LAW_OFFSET_TEMPORAL = 1.0

TARGET_COLS = ['k', 'power', 'step', 'time', 'mean_u', 'u_dot', 'x', 'Re(u)', 'Im(u)', 'bin_center', 'probability_density', 'var_u']

# =============================================================================
# PARALLEL WORKER FUNCTION
# =============================================================================
def process_step_task(params):
    """Processes a single time step completely independently."""
    step_int, base_prefix, out_dirs, k_cutoff, override, make_video = params
    
    # Results dictionary to return extracted raw data back to main thread
    res = {
        'step': step_int, 'spatial': None, 'temporal': None, 
        'real_state': None, 'hist': None, 'inst_spatial': None, 'inst_hist': None,
        'lims': {
            'k_min': float('inf'), 'k_max': float('-inf'),
            'p_min': float('inf'), 'p_max': float('-inf'),
            'w_min': float('inf'), 'w_max': float('-inf'),
            'wp_min': float('inf'), 'wp_max': float('-inf'),
            'x_min': float('inf'), 'x_max': float('-inf'),
            'u_min': float('inf'), 'u_max': float('-inf'),
            'pdf_max': 0.5
        }
    }

    # Helper Gaussian for histograms
    x_gauss = np.linspace(-5, 5, 200).astype(np.float32)
    y_gauss = ((1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)).astype(np.float32)

    try:
        # --- 1. Spatial Power Spectrum ---
        spat_file = f"{base_prefix}_ps_step_{step_int}.txt"
        if os.path.exists(spat_file):
            df = pd.read_csv(spat_file, sep=' ', engine='c', dtype=np.float32, usecols=lambda c: c in TARGET_COLS)
            if not df.empty and 'k' in df.columns and 'power' in df.columns:
                
                # --- AFTER-THE-FACT BUG FIX ---
                N_grid = len(df)
                positive_k = df.loc[df['k'] > 0, 'k']
                dk = positive_k.min() if not positive_k.empty else 1.0
                L = 2.0 * np.pi / dk
                correction = L * (N_grid ** 2) if APPLY_AFTER_THE_FACT_FIX else 1.0
                # ------------------------------

                df = df[(df['k'] > 0) & (df['k'] <= k_cutoff) & (df['power'] > 0)].sort_values(by='k')
                k, p = df['k'].values, df['power'].values * correction
                res['spatial'] = (k, p)
                
                if len(k) > 0:
                    res['lims']['k_min'], res['lims']['k_max'] = k.min(), k.max()
                    res['lims']['p_min'], res['lims']['p_max'] = p.min(), p.max()

                out_file = os.path.join(out_dirs['spatial_ps'], f'ps_step_{step_int}.png')
                if override or not os.path.exists(out_file):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.loglog(k, p, 'k', alpha=0.8, label="Data")
                    if len(p) > 0:
                        c = (p[0] * k[0]**2) * POWER_LAW_OFFSET_SPATIAL
                        ax.loglog(k, c * (k**-2), color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
                    ax.set_xlabel(LABEL_SPATIAL_K)
                    ax.set_ylabel(LABEL_SPATIAL_POWER)
                    t_str = TITLE_SPATIAL_PS.format(step=step_int) + (f' (k cutoff = {k_cutoff})' if k_cutoff != float('inf') else '')
                    ax.set_title(t_str)
                    ax.legend(loc='upper right')
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    fig.tight_layout()
                    fig.savefig(out_file)
                    plt.close(fig)

# --- 2. Temporal Power Spectrum ---
        ts_file = f"{base_prefix}_mean_u_ts_step_{step_int}.txt"
        out_file = os.path.join(out_dirs['temporal_ps'], f'time_ps_u_dot_step_{step_int}.png')
        if os.path.exists(ts_file) and (make_video or override or not os.path.exists(out_file)):
            df = pd.read_csv(ts_file, sep=' ', engine='c', dtype=np.float32, usecols=lambda c: c in TARGET_COLS)
            if 'u_dot' in df.columns and len(df) > 1:
                dt_step = (df['time'].iloc[-1] - df['time'].iloc[0]) / (len(df) - 1)
                if dt_step > 0:
                    n_samples = len(df)
                    
                    # Directly FFT the exact u_dot signal without a window
                    fft_vals = np.fft.rfft(df['u_dot'].values)
                    
                    # Frequencies
                    freqs_hz = np.fft.rfftfreq(n_samples, d=dt_step).astype(np.float32)
                    omega = 2.0 * np.pi * freqs_hz
                    
                    # Proper One-Sided Power Spectral Density (PSD)
                    # No omega^2 multiplier needed since we are FFTing the velocity directly
                    P_omega = (2.0 * dt_step / n_samples) * (np.abs(fft_vals)**2)
                    
                    valid_idx = (omega > 0) & (P_omega > 0)
                    omega, P_omega = omega[valid_idx], P_omega[valid_idx]
                    
                    if len(omega) > 0:
                        res['temporal'] = (omega, P_omega)
                        res['lims']['w_min'], res['lims']['w_max'] = omega.min(), omega.max()
                        res['lims']['wp_min'], res['lims']['wp_max'] = P_omega.min(), P_omega.max()

                        if override or not os.path.exists(out_file):
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.loglog(omega, P_omega, color='purple', alpha=0.8, label="Data")
                            c_temp = (P_omega[0] * omega[0]**2) * POWER_LAW_OFFSET_TEMPORAL
                            ax.loglog(omega, c_temp * (omega**-2), color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE_TEMPORAL)
                            ax.set_xlabel(LABEL_TEMPORAL_W)
                            ax.set_ylabel(LABEL_TEMPORAL_POWER)
                            ax.set_title(TITLE_TEMPORAL_PS.format(step=step_int))
                            ax.legend(loc='upper right')
                            ax.grid(True, which='both', linestyle='--', alpha=0.5)
                            fig.tight_layout()
                            fig.savefig(out_file)
                            plt.close(fig)

        # --- 3. Real State ---
        real_file = f"{base_prefix}_real_state_step_{step_int}.txt"
        out_file = os.path.join(out_dirs['real_state'], f'real_state_step_{step_int}.png')
        if os.path.exists(real_file) and (make_video or override or not os.path.exists(out_file)):
            df = pd.read_csv(real_file, sep=' ', engine='c', dtype=np.float32, usecols=lambda c: c in TARGET_COLS)
            if not df.empty and 'x' in df.columns:
                x, u_real, u_imag = df['x'].values, df['Re(u)'].values, df['Im(u)'].values
                u_centered = u_real - np.mean(u_real)
                phase = np.arctan2(u_imag, u_centered)
                res['real_state'] = (x, u_centered, phase)
                
                if len(x) > 0:
                    res['lims']['x_min'], res['lims']['x_max'] = x.min(), x.max()
                    res['lims']['u_min'], res['lims']['u_max'] = u_centered.min(), u_centered.max()

                if override or not os.path.exists(out_file):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    points = np.array([x, u_centered]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(-np.pi, np.pi)
                    lc = LineCollection(segments, cmap='hsv', norm=norm)
                    lc.set_array(phase[:-1])
                    lc.set_linewidth(1.5)
                    ax.add_collection(lc)
                    
                    ax.set_xlim(x.min(), x.max())
                    y_margin = max(abs(u_centered.min()), abs(u_centered.max())) * 0.1
                    y_margin = 0.1 if y_margin == 0 else y_margin
                    ax.set_ylim(u_centered.min() - y_margin, u_centered.max() + y_margin)
                    
                    ax.set_xlabel('x')
                    ax.set_ylabel('Re(u) - mean(Re(u))')
                    ax.set_title(f'Real Space State at step = {step_int}')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    fig.tight_layout()
                    fig.savefig(out_file)
                    plt.close(fig)

        # --- 4. Pre-computed Histogram ---
        hist_file = f"{base_prefix}_height_hist_step_{step_int}.txt"
        out_file = os.path.join(out_dirs['hist'], f'hist_step_{step_int}.png')
        if os.path.exists(hist_file) and (make_video or override or not os.path.exists(out_file)):
            df = pd.read_csv(hist_file, sep=' ', engine='c', dtype=np.float32, usecols=lambda c: c in TARGET_COLS)
            if not df.empty and 'bin_center' in df.columns:
                bin_centers, pdf = df['bin_center'].values, df['probability_density'].values
                res['hist'] = (bin_centers, pdf)
                if len(pdf) > 0:
                    res['lims']['pdf_max'] = max(res['lims']['pdf_max'], pdf.max() * 1.1)

                if override or not os.path.exists(out_file):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
                    ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Simulation Data')
                    ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
                    ax.set_xlim(-5, 5)
                    ax.set_xlabel('(Re(u) - mean) / std')
                    ax.set_ylabel('Density')
                    ax.set_title(f'Normalized Histogram of Re(u) at step = {step_int}')
                    ax.legend(loc='upper right')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    fig.tight_layout()
                    fig.savefig(out_file)
                    plt.close(fig)

        # --- 5. Instantaneous Spatial PS ---
        inst_spat_file = f"{base_prefix}_inst_ps_step_{step_int}.txt"
        out_file = os.path.join(out_dirs['inst_spatial_ps'], f'inst_ps_step_{step_int}.png')
        if os.path.exists(inst_spat_file):
            df = pd.read_csv(inst_spat_file, sep=' ', engine='c', dtype=np.float32, usecols=lambda c: c in TARGET_COLS)
            if not df.empty and 'k' in df.columns:
                
                # --- AFTER-THE-FACT BUG FIX ---
                N_grid = len(df)
                positive_k = df.loc[df['k'] > 0, 'k']
                dk = positive_k.min() if not positive_k.empty else 1.0
                L = 2.0 * np.pi / dk
                correction = L * (N_grid ** 2) if APPLY_AFTER_THE_FACT_FIX else 1.0
                # ------------------------------

                df = df[(df['k'] > 0) & (df['k'] <= k_cutoff) & (df['power'] > 0)].sort_values(by='k')
                k, p = df['k'].values, df['power'].values * correction
                res['inst_spatial'] = (k, p)

                if override or not os.path.exists(out_file):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.loglog(k, p, 'k', alpha=0.8, label="Instantaneous Data")
                    if len(p) > 0:
                        c = (p[0] * k[0]**2) * POWER_LAW_OFFSET_SPATIAL
                        ax.loglog(k, c * (k**-2), color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
                    ax.set_xlabel(LABEL_SPATIAL_K)
                    ax.set_ylabel(LABEL_SPATIAL_POWER)
                    t_str = f"Instantaneous Spatial PS at step = {step_int}" + (f' (k cutoff = {k_cutoff})' if k_cutoff != float('inf') else '')
                    ax.set_title(t_str)
                    ax.legend(loc='upper right')
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    fig.tight_layout()
                    fig.savefig(out_file)
                    plt.close(fig)

        # --- 6. Instantaneous Histogram ---
        inst_hist_file = f"{base_prefix}_inst_height_hist_step_{step_int}.txt"
        out_file = os.path.join(out_dirs['inst_hist'], f'inst_hist_step_{step_int}.png')
        if os.path.exists(inst_hist_file) and (make_video or override or not os.path.exists(out_file)):
            df = pd.read_csv(inst_hist_file, sep=' ', engine='c', dtype=np.float32, usecols=lambda c: c in TARGET_COLS)
            if not df.empty and 'bin_center' in df.columns:
                bin_centers, pdf = df['bin_center'].values, df['probability_density'].values
                res['inst_hist'] = (bin_centers, pdf)

                if override or not os.path.exists(out_file):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
                    ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Instantaneous Data')
                    ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
                    ax.set_xlim(-5, 5)
                    ax.set_xlabel('(Re(u) - mean) / std')
                    ax.set_ylabel('Density')
                    ax.set_title(f'Instantaneous Histogram of Re(u) at step = {step_int}')
                    ax.legend(loc='upper right')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    fig.tight_layout()
                    fig.savefig(out_file)
                    plt.close(fig)

    except Exception as e:
        print(f"Worker Error processing step {step_int}: {e}")
        
    gc.collect()
    return res

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    args = sys.argv[1:]
    
    override = False
    if "--override" in args:
        override = True
        args.remove("--override")
        
    outdir_override = None
    if "--outdir" in args:
        idx = args.index("--outdir")
        if idx + 1 < len(args):
            outdir_override = args[idx + 1]
            args.pop(idx)
            args.pop(idx)
        else:
            print("Error: --outdir requires a path argument.")
            sys.exit(1)

    if len(args) < 1:
        print("Usage: python plot_time.py <base_prefix> [k_cutoff] [--override] [--outdir <path>]")
        sys.exit(1)

    # 1. Paths & Validations
    raw_prefix = args[0]
    base_prefix = os.path.abspath(raw_prefix)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    k_cutoff = float(args[1]) if len(args) > 1 else float('inf')
    summary_file = f"{base_prefix}{SUMMARY_FILE_SUFFIX}"

    data_dir = os.path.dirname(base_prefix)
    prefix_basename = os.path.basename(base_prefix)
    if not os.path.exists(summary_file):
        print(f"Error: Summary file {summary_file} not found.")
        sys.exit(1)

    data_dir_name = os.path.basename(data_dir)
    base_figures_dir = os.path.abspath(outdir_override) if outdir_override else os.path.abspath(os.path.join("..", "figures"))
    figures_dir = os.path.join(base_figures_dir, data_dir_name)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving figures and videos to: {figures_dir}")
    print(f"Override mode: {'ON (Re-plotting all)' if override else 'OFF (Skipping existing plots)'}")

    # Read summary file
    try:
        df_summary = pd.read_csv(summary_file, sep=' ', engine='c', dtype=np.float32)
    except pd.errors.EmptyDataError:
        print("The summary file is empty. Please wait for the first point to finish.")
        sys.exit(0)

    if df_summary.empty:
        sys.exit(0)

    df_summary = df_summary.sort_values(by='total_steps').reset_index(drop=True)
    print(f"Loaded {len(df_summary)} time step evaluations.")

    # Core Summary Metric Plots
    for y_col, ylabel, title, logy, c, m in [
        ('spectral_entropy', LABEL_ENTROPY, TITLE_ENTROPY, True, 'b', 'o'),
        ('var_u', LABEL_VARIANCE, TITLE_VARIANCE, True, 'r', 's'),
        ('u_dot', LABEL_U_DOT, TITLE_U_DOT, False, 'g', '^'),
        ('phi_dot', LABEL_PHI_DOT, TITLE_PHI_DOT, False, 'm', 'v')
    ]:
        if y_col in df_summary.columns:
            plt.figure(figsize=(8, 6))
            plt.plot(df_summary['total_steps'], df_summary[y_col], marker=m, linestyle='-', color=c)
            plt.xscale('log', base=2)
            if logy: plt.yscale('log')
            plt.xlabel(LABEL_TIME); plt.ylabel(ylabel); plt.title(title)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"time_{y_col}.png"))
            plt.close()

    # Create subdirectories
    out_dirs = {
        'spatial_ps': os.path.join(figures_dir, SPATIAL_PS_DIR),
        'temporal_ps': os.path.join(figures_dir, TEMPORAL_PS_DIR),
        'real_state': os.path.join(figures_dir, REAL_STATE_DIR),
        'hist': os.path.join(figures_dir, HISTOGRAM_DIR),
        'inst_spatial_ps': os.path.join(figures_dir, INST_SPATIAL_PS_DIR),
        'inst_hist': os.path.join(figures_dir, INST_HISTOGRAM_DIR)
    }
    for d in out_dirs.values(): os.makedirs(d, exist_ok=True)

    # 2. Parallel Processing execution for all time steps
    spatial_data, temporal_data, real_state_data, hist_data, inst_spatial_data, inst_hist_data = {}, {}, {}, {}, {}, {}
    global_lims = {
        'k_min': float('inf'), 'k_max': float('-inf'), 'p_min': float('inf'), 'p_max': float('-inf'),
        'w_min': float('inf'), 'w_max': float('-inf'), 'wp_min': float('inf'), 'wp_max': float('-inf'),
        'x_min': float('inf'), 'x_max': float('-inf'), 'u_min': float('inf'), 'u_max': float('-inf'),
        'pdf_max': 0.5
    }

    # Cap workers to 2 to prevent OOM
    cpu_count = min(2, multiprocessing.cpu_count())
    tasks = [(int(step), base_prefix, out_dirs, k_cutoff, override, MAKE_VIDEO) for step in df_summary['total_steps']]
    
    print(f"Processing individual plots concurrently across {cpu_count} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        for res in executor.map(process_step_task, tasks):
            step = res['step']
            if res['spatial']: spatial_data[step] = res['spatial']
            if res['temporal']: temporal_data[step] = res['temporal']
            if res['real_state']: real_state_data[step] = res['real_state']
            if res['hist']: hist_data[step] = res['hist']
            if res['inst_spatial']: inst_spatial_data[step] = res['inst_spatial']
            if res['inst_hist']: inst_hist_data[step] = res['inst_hist']
            
            for k in global_lims.keys():
                if '_max' in k: global_lims[k] = max(global_lims[k], res['lims'][k])
                else: global_lims[k] = min(global_lims[k], res['lims'][k])

    print("Completed concurrent parsing and plotting.")

    # --- Family-Vicsek Scaling Collapse ---
    step_to_time = {}
    if 'time' in df_summary.columns:
        step_to_time.update(dict(zip(df_summary['total_steps'].astype(int).values, df_summary['time'].values)))

    if not df_summary.empty:
        zeta, z = 0.5, 2.0
        exponent_y, exponent_x = (1.0 + 2.0 * zeta) / z, 1.0 / z

        def plot_ew_collapse(data_dict, title, filename):
            valid_times = [step_to_time[s] for s in data_dict.keys() if s in step_to_time and step_to_time[s] > 0]
            if not valid_times: return
            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = plt.get_cmap('plasma')
            norm = mcolors.LogNorm(vmin=min(valid_times), vmax=max(valid_times))
            sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

            for step_int, (k_vals, power_vals) in data_dict.items():
                if step_int in step_to_time and step_to_time[step_int] > 0:
                    t = step_to_time[step_int]
                    ax.loglog(k_vals * (t ** exponent_x), power_vals / (t ** exponent_y), alpha=0.2 + 0.8 * norm(t), color=cmap(norm(t)))
            
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Time (t)')
            ax.set_xlabel(r'$\kappa t^{1/z}$'); ax.set_ylabel(r'$S_\kappa(t) / t^{(1+2\zeta)/z}$')
            ax.set_title(title); ax.grid(True, which='both', linestyle='--', alpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(figures_dir, filename))
            plt.close(fig)

        plot_ew_collapse(inst_spatial_data, r'Instantaneous EW Data Collapse (1D EW: $\zeta=0.5$, $z=2.0$)', 'inst_ew_data_collapse.png')
        plot_ew_collapse(spatial_data, r'Time-Averaged EW Data Collapse (1D EW: $\zeta=0.5$, $z=2.0$)', 'avg_ew_data_collapse.png')

    # --- Generate Videos ---
    # We fully clear and redraw the canvas on EVERY frame to guarantee correct rendering (fixes Pillow empty data bugs)
    if MAKE_VIDEO:
        vid_fmt = VIDEO_FORMAT
        writer = 'ffmpeg' if vid_fmt == "mp4" else 'pillow'
        if vid_fmt == "mp4" and not animation.writers.is_available('ffmpeg'):
            print("Warning: ffmpeg is not available. Falling back to GIF format using Pillow.")
            vid_fmt = "gif"; writer = 'pillow'

        print(f"Generating videos in {vid_fmt} format...")

        x_gauss = np.linspace(-5, 5, 200).astype(np.float32)
        y_gauss = ((1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)).astype(np.float32)

        # SPATIAL VIDEO
        if spatial_data:
            valid_steps = sorted(list(spatial_data.keys()))
            fig, ax = plt.subplots(figsize=(8, 6))

            def update_spat(frame):
                ax.clear()
                step_int = valid_steps[frame]
                k, p = spatial_data[step_int]
                ax.loglog(k, p, color='k', alpha=0.8, label="Data")
                if len(p) > 0 and len(k) > 0:
                    c = (p[0] * k[0]**2) * POWER_LAW_OFFSET_SPATIAL
                    ax.loglog(k, c * (k**-2), color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE)
                if FIX_AXES_FOR_VIDEO and global_lims['p_min'] < global_lims['p_max']:
                    ax.set_xlim(max(1e-10, global_lims['k_min']), global_lims['k_max'])
                    ax.set_ylim(max(1e-10, global_lims['p_min']) * 0.1, global_lims['p_max'] * 10.0)
                ax.set_xlabel(LABEL_SPATIAL_K); ax.set_ylabel(LABEL_SPATIAL_POWER); ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend(loc='upper right')
                t_str = TITLE_SPATIAL_PS.format(step=step_int) + (f' (k cutoff = {k_cutoff})' if k_cutoff != float('inf') else '')
                ax.set_title(t_str)

            ani = animation.FuncAnimation(fig, update_spat, frames=len(valid_steps), blit=False)
            out_file = os.path.join(figures_dir, f"{VIDEO_SPATIAL_NAME}.{vid_fmt}")
            ani.save(out_file, writer=writer, fps=VIDEO_FPS)
            plt.close(fig); print(f"Saved {out_file}")

        # TEMPORAL VIDEO
        if temporal_data:
            valid_steps = sorted(list(temporal_data.keys()))
            fig, ax = plt.subplots(figsize=(8, 6))

            def update_temp(frame):
                ax.clear()
                step_int = valid_steps[frame]
                w, p = temporal_data[step_int]
                ax.loglog(w, p, color='purple', alpha=0.8, label="Data")
                if len(p) > 0 and len(w) > 0:
                    c = (p[0] * w[0]**2) * POWER_LAW_OFFSET_TEMPORAL
                    ax.loglog(w, c * (w**-2), color=COLOR_REF_LINE, linestyle=STYLE_REF_LINE, label=LABEL_REF_LINE_TEMPORAL)
                if FIX_AXES_FOR_VIDEO and global_lims['wp_min'] < global_lims['wp_max']:
                    ax.set_xlim(max(1e-10, global_lims['w_min']), global_lims['w_max'])
                    ax.set_ylim(max(1e-10, global_lims['wp_min']) * 0.1, global_lims['wp_max'] * 10.0)
                ax.set_xlabel(LABEL_TEMPORAL_W); ax.set_ylabel(LABEL_TEMPORAL_POWER); ax.grid(True, which='both', linestyle='--', alpha=0.5)
                ax.legend(loc='upper right'); ax.set_title(TITLE_TEMPORAL_PS.format(step=step_int))

            ani = animation.FuncAnimation(fig, update_temp, frames=len(valid_steps), blit=False)
            out_file = os.path.join(figures_dir, f"{VIDEO_TEMPORAL_NAME}.{vid_fmt}")
            ani.save(out_file, writer=writer, fps=VIDEO_FPS)
            plt.close(fig); print(f"Saved {out_file}")

        # REAL STATE VIDEO
        if real_state_data:
            valid_steps = sorted(list(real_state_data.keys()))
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Setup static colorbar once to avoid duplicate bars
            norm = plt.Normalize(-np.pi, np.pi)
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='hsv'), ax=ax, ticks=[-np.pi, 0, np.pi])
            cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$']); cbar.set_label('Phase')

            def update_real(frame):
                ax.clear()
                step_int = valid_steps[frame]
                x, u_c, phase = real_state_data[step_int]
                points = np.array([x, u_c]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, cmap='hsv', norm=norm)
                lc.set_array(phase[:-1])
                ax.add_collection(lc)
                
                if FIX_AXES_FOR_VIDEO and global_lims['u_max'] > global_lims['u_min']:
                    margin = max(abs(global_lims['u_min']), abs(global_lims['u_max'])) * 0.1
                    margin = 0.1 if margin == 0 else margin
                    ax.set_xlim(global_lims['x_min'], global_lims['x_max'])
                    ax.set_ylim(global_lims['u_min'] - margin, global_lims['u_max'] + margin)
                else:
                    ax.set_xlim(x.min(), x.max())
                    y_margin = max(abs(u_c.min()), abs(u_c.max())) * 0.1
                    y_margin = 0.1 if y_margin == 0 else y_margin
                    ax.set_ylim(u_c.min() - y_margin, u_c.max() + y_margin)

                ax.set_xlabel('x'); ax.set_ylabel('Re(u) - mean(Re(u))')
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(f'Real Space State at step = {step_int}')

            ani = animation.FuncAnimation(fig, update_real, frames=len(valid_steps), blit=False)
            out_file = os.path.join(figures_dir, f"{VIDEO_REAL_NAME}.{vid_fmt}")
            ani.save(out_file, writer=writer, fps=VIDEO_FPS)
            plt.close(fig); print(f"Saved {out_file}")

        # HISTOGRAM VIDEO
        if hist_data:
            valid_steps = sorted(list(hist_data.keys()))
            fig, ax = plt.subplots(figsize=(8, 6))

            def update_hist(frame):
                ax.clear()
                step_int = valid_steps[frame]
                bin_centers, pdf = hist_data[step_int]
                width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
                ax.bar(bin_centers, pdf, width=width, alpha=0.6, color='blue', label='Simulation Data', align='center')
                ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label='Standard Gaussian')
                ax.set_xlim(-5, 5)
                ax.set_ylim(0, global_lims['pdf_max'] if FIX_AXES_FOR_VIDEO else max(0.5, pdf.max() * 1.1))
                ax.set_xlabel('(Re(u) - mean) / std'); ax.set_ylabel('Density'); ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper right'); ax.set_title(f'Normalized Histogram of Re(u) at step = {step_int}')

            ani = animation.FuncAnimation(fig, update_hist, frames=len(valid_steps), blit=False)
            out_file = os.path.join(figures_dir, f"{VIDEO_HIST_NAME}.{vid_fmt}")
            ani.save(out_file, writer=writer, fps=VIDEO_FPS)
            plt.close(fig); print(f"Saved {out_file}")

    # --- Transient Variance Vectorized Plotting ---
    transient_file = f"{base_prefix}{TRANSIENT_FILE_SUFFIX}"
    if os.path.exists(transient_file):
        try:
            valid_steps_set = set(df_summary['total_steps'].astype(int).values)
            # Load fully and vectorize constraints rather than chunking with skipped strides
            df_trans = pd.read_csv(transient_file, sep=' ', engine='c', dtype=np.float32, usecols=['step', 'time', 'var_u'])
            
            mask = (df_trans['time'] > 0) & df_trans['step'].isin(valid_steps_set)
            df_trans = df_trans[mask].sort_values(by='time').reset_index(drop=True)

            if not df_trans.empty and len(df_trans) > 1:
                out_path = os.path.join(figures_dir, "transient_variance_scaling.png")
                if override or not os.path.exists(out_path):
                    plt.figure(figsize=(8, 6))
                    plt.loglog(df_trans['time'], df_trans['var_u'], color='r', label="Simulated Variance (W^2)")
                    
                    t_vals, var_vals = df_trans['time'].values, df_trans['var_u'].values
                    if len(t_vals) > 0 and var_vals[-1] > 0:
                        c_t = var_vals[-1] / (t_vals[-1]**0.5)
                        plt.loglog(t_vals, c_t * (t_vals**0.5), color='k', linestyle='--', label="t^(1/2) EW Scaling")
                    
                    plt.xlabel("Time (t)"); plt.ylabel("Spatial Variance W^2"); plt.title("Transient Spatial Variance vs Time (Log-Log)")
                    plt.legend(loc='upper right'); plt.grid(True, which='both', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(out_path)
                    plt.close()
            del df_trans
        except Exception as e:
            print(f"Skipping transient plot: {e}")

if __name__ == "__main__":
    main()