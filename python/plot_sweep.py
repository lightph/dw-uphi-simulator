# python/plot_sweep.py
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, ListedColormap

# ==========================================
# PRB PUBLICATION-READY PLOT SETTINGS
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.fontsize': 12,
    'legend.frameon': False,
    'figure.figsize': (7, 5),  # Professional aspect ratio
})

def get_first_major_gap_threshold(series, min_gap_orders=2.0, default_val=1e-8):
    """
    Scans from the smallest valid values upwards in log space.
    Returns the midpoint of the FIRST gap that exceeds min_gap_orders.
    This separates the numerical noise floor from macroscopic physics.
    """
    valid = series[series > 0].dropna().values
    if len(valid) < 2:
        return default_val
        
    log_vals = np.log10(np.sort(valid))
    gaps = np.diff(log_vals)
    
    for i, gap in enumerate(gaps):
        if gap > min_gap_orders:
            # Place the threshold directly in the center of the gap
            return 10 ** (log_vals[i] + gap / 2.0)
            
    return default_val

def get_largest_gap_threshold(series, max_search_val=1e-1, default_val=1e-8, gap_fraction=0.5):
    """
    Finds the largest logarithmic gap in the data below a maximum physical threshold.
    gap_fraction (0.0 to 1.0): 
        0.1 = Very Strict (threshold placed just above the noise floor)
        0.5 = Balanced (geometric center of the gap)
        0.9 = Very Loose (threshold placed just below macroscopic physics)
    """
    valid = series[(series > 0) & (series < max_search_val)].dropna().values
    if len(valid) < 2:
        return default_val
        
    log_vals = np.log10(np.sort(valid))
    gaps = np.diff(log_vals)
    
    if len(gaps) == 0:
        return default_val
        
    max_gap_idx = np.argmax(gaps)
    # Slide the threshold up from the bottom of the gap according to the fraction
    return 10 ** (log_vals[max_gap_idx] + gaps[max_gap_idx] * gap_fraction)

# ==========================================
# FLOQUET BOUNDARY OVERLAY SETTINGS
# ==========================================
# Adjusted to match the C++ output naming pattern (e.g., L = 10)
FLOQUET_BIN_FILE = "/home/luz.perez/inv1/stability/floquet_transient_omega_1.000_L_1024.bin"
FLOQUET_NUM_H = 1000
FLOQUET_NUM_HAC = 1000
FLOQUET_H_MAX = 10.0
FLOQUET_HAC_MAX = 20.0

# Synchrony Region Boundary (converged)
SYNC_COLOR = 'white'
SYNC_STYLE = 'solid'
SYNC_WIDTH = 2.0

# Stability Region Boundary (max_mu = 1.0)
STAB_COLOR = 'black'
STAB_STYLE = 'dashed'
STAB_WIDTH = 2.0
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Plot comprehensive 2D sweep results.")
    parser.add_argument("input_dir", help="Directory containing results_h0_*.txt files")
    parser.add_argument("output_dir", help="Directory to save the plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.input_dir, "results_h0_*.txt"))
    
    if not files:
        print(f"No results files found in {args.input_dir}")
        return
        
    # Efficiently load and concatenate all valid data
    dfs = [pd.read_csv(f, sep=r"\s+") for f in files if os.path.getsize(f) > 0]
    if not dfs:
        print("No valid data could be extracted from the files.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Round parameters to prevent floating point mismatch during pivoting
    df['h0'] = df['h0'].round(5)
    df['ha'] = df['ha'].round(5)

    # ==========================================
    # LOAD FLOQUET BINARY DATA
    # ==========================================
    floquet_data_loaded = False
    H_GRID = HAC_GRID = converged_grid = mu_grid = None
    
    if os.path.exists(FLOQUET_BIN_FILE):
        # Updated dtype matching the new C++ Result struct layout (with explicit alignment/padding if necessary)
        dtype = np.dtype([
            ('h', '<f8'),
            ('h_ac', '<f8'),
            ('omega', '<f8'),
            ('x', '<f8'),
            ('max_k_mu', '<f8'),
            ('max_mu', '<f8'),
            ('max_k_sigma', '<f8'),
            ('max_sigma', '<f8'),
            ('converged', '<?')
        ], align=True)
        
        try:
            raw_data = np.fromfile(FLOQUET_BIN_FILE, dtype=dtype)
            
            if len(raw_data) == FLOQUET_NUM_H * FLOQUET_NUM_HAC:
                converged_grid = raw_data['converged'].reshape((FLOQUET_NUM_H, FLOQUET_NUM_HAC))
                mu_grid = raw_data['max_mu'].reshape((FLOQUET_NUM_H, FLOQUET_NUM_HAC))
                
                mu_grid = np.where(converged_grid, mu_grid, np.nan)
                
                h_vals = np.linspace(0, FLOQUET_H_MAX, FLOQUET_NUM_H)
                hac_vals = np.linspace(0, FLOQUET_HAC_MAX, FLOQUET_NUM_HAC)
                # Note: indexing 'ij' corresponds to rows=h (h0), cols=h_ac (ha) matching pivot tables
                H_GRID, HAC_GRID = np.meshgrid(h_vals, hac_vals, indexing='ij')
                
                floquet_data_loaded = True
                print(f"Successfully loaded {FLOQUET_BIN_FILE} for overlays.")
            else:
                print(f"Warning: Floquet bin file size mismatch. Expected {FLOQUET_NUM_H * FLOQUET_NUM_HAC}, got {len(raw_data)}. Skipping overlays.")
        except Exception as e:
            print(f"Warning: Failed to read {FLOQUET_BIN_FILE}: {e}")

    # ==========================================
    # DATA-DRIVEN THRESHOLD DISCOVERY
    # ==========================================
    THRESH_LOCK = get_largest_gap_threshold(df['mean_var_strobo'], max_search_val=1e-1, default_val=1e-10, gap_fraction=0.1)
    THRESH_RUG  = get_largest_gap_threshold(df['rugosity'], max_search_val=1e-1, default_val=1e-7, gap_fraction=0.2)
    THRESH_VPHI = get_largest_gap_threshold(df['v_phi'].abs(), max_search_val=1e-1, default_val=1e-6, gap_fraction=0.2)

    is_locked = df['mean_var_strobo'] < THRESH_LOCK
    is_flat = df['rugosity'] <= THRESH_RUG
    is_pinned = is_locked & (df['v_phi'].abs() < THRESH_VPHI)

    df['phase_locked'] = is_locked.astype(int)
    df['structure'] = np.where(is_flat, 0, np.where(is_locked, 1, 2))
    df['binarized_rugosity'] = (~is_flat).astype(int)

    conditions = [
        is_pinned,                             # 0: Pinned (P)
        is_locked & is_flat & ~is_pinned,      # 1: Phase Locked Flat (SF)
        is_locked & ~is_flat,                  # 2: Phase Locked Periodic (SP)
        ~is_locked & is_flat,                  # 3: Unlocked Flat (NF)
        ~is_locked & ~is_flat                  # 4: Unlocked Rough (NR)
    ]
    df['phase_diagram'] = np.select(conditions, [0, 1, 2, 3, 4], default=-1)
    df['wall_q'] = np.where(df['structure'] == 1, df['max_q'], 0.0)
    
    unique_ha, unique_h0 = df['ha'].nunique(), df['h0'].nunique()
    is_1d = (unique_ha == 1) or (unique_h0 == 1)

    # ==========================================
    # METRICS CONFIGURATION FOR PLOTTING
    # ==========================================
    metrics = {
        'phase_diagram': {
            'cmap': ListedColormap(['green', 'blue', 'cyan', 'red', 'darkred']), 
            'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 4.5,
            'ticks': [0, 1, 2, 3, 4], 
            'labels': ['P', 'SF', 'SP', 'NF', 'NR'],
            'display_name': 'Phase'
        },
        'binarized_rugosity': {
            'cmap': ListedColormap(['#4575b4', '#d73027']), 
            'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 1.5,
            'ticks': [0, 1], 'labels': ['Flat (0)', 'Rough (1)'],
            'display_name': 'Binarized Rugosity'
        },
        'phase_locked': {
            'cmap': ListedColormap(['#d73027', '#1a9850']), 
            'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 1.5,
            'ticks': [0, 1], 'labels': ['False (0)', 'True (1)'],
            'display_name': 'Phase Locked'
        },
        'structure': {
            'cmap': ListedColormap(['#4575b4', '#ffffbf', '#d73027']), 
            'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 2.5,
            'ticks': [0, 1, 2], 'labels': ['Flat (0)', 'Periodic (1)', 'Corrugated (2)'],
            'display_name': 'Spatial Structure'
        },
        'mean_var_strobo': {'cmap': 'magma', 'log': True, 'cat': False, 'display_name': r'$\langle \sigma^2 \rangle$'},
        'spectral_entropy': {'cmap': 'plasma', 'log': False, 'cat': False, 'display_name': 'Spectral Entropy'},
        'rugosity': {'cmap': 'inferno', 'log': True, 'cat': False, 'display_name': 'Rugosity'},
        'v_u': {'cmap': 'coolwarm', 'log': False, 'cat': False, 'display_name': r'$v_u$'},
        'v_phi': {'cmap': 'coolwarm', 'log': False, 'cat': False, 'display_name': r'$v_\phi$'},
        'max_q': {'cmap': 'viridis', 'log': False, 'cat': False, 'display_name': r'Max $q$'},
        'wall_q': {'cmap': 'viridis', 'log': False, 'cat': False, 'display_name': r'Wall $q$'}
    }

    # LaTeX-formatted axis labels
    label_h0 = r'$h_0$'
    label_ha = r'$h_\mathrm{a}$'

    for metric, props in metrics.items():
        if metric not in df.columns:
            continue
            
        plt.figure() # Uses PRB size configured in rcParams
        
        if is_1d:
            x_col = 'h0' if unique_ha == 1 else 'ha'
            x_label = label_h0 if x_col == 'h0' else label_ha
            df_sorted = df.dropna(subset=[metric]).sort_values(by=x_col)
            
            if df_sorted.empty:
                plt.close()
                continue
            
            if props['log']:
                plt.semilogy(df_sorted[x_col], df_sorted[metric] + 1e-15, marker='o', linestyle='-', color='black', markersize=4)
            else:
                plt.plot(df_sorted[x_col], df_sorted[metric], marker='o', linestyle='-', color='black', markersize=4)
                
            plt.xlabel(x_label)
            plt.ylabel(props.get('display_name', metric))
            
            # Use categorical y-ticks if it is a phase / categorical plot
            if props['cat']:
                plt.yticks(props['ticks'], props['labels'])
                
        else:
            pivot = df.pivot_table(index='ha', columns='h0', values=metric)
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            Z = np.ma.masked_invalid(pivot.values)
            
            if Z.count() == 0:
                plt.close()
                continue
                
            kwargs = {'shading': 'nearest', 'cmap': props['cmap']}
            
            if props['cat']:
                kwargs.update({'vmin': props['vmin'], 'vmax': props['vmax']})
                mesh = plt.pcolormesh(X, Y, Z, **kwargs)
                
                # Special handling for phase diagram legend
                if metric == 'phase_diagram':
                    colors = props['cmap'].colors
                    patches = [mpatches.Patch(color=colors[i], label=props['labels'][i]) for i in range(len(colors))]
                    plt.legend(handles=patches, loc='best', framealpha=0.9, edgecolor='black')
                else:
                    cbar = plt.colorbar(mesh, ticks=props['ticks'])
                    cbar.set_ticklabels(props['labels'])
                    cbar.set_label(props.get('display_name', metric))
            else:
                if props['log']:
                    Z_plot = np.clip(Z, a_min=1e-15, a_max=None)
                    kwargs['norm'] = LogNorm(vmin=Z_plot.min(), vmax=Z_plot.max())
                    mesh = plt.pcolormesh(X, Y, Z_plot, **kwargs)
                else:
                    mesh = plt.pcolormesh(X, Y, Z, **kwargs)
                
                cbar = plt.colorbar(mesh)
                cbar.set_label(props.get('display_name', metric))
                
            plt.xlabel(label_h0)
            plt.ylabel(label_ha)
            
            # Draw Floquet overlays if available
            if floquet_data_loaded:
                xlims = plt.xlim()
                ylims = plt.ylim()

                plt.contour(H_GRID, HAC_GRID, converged_grid, levels=[0.5],
                            colors=[SYNC_COLOR], linestyles=[SYNC_STYLE], 
                            linewidths=[SYNC_WIDTH], alpha=0.9)
                
                if np.nanmin(mu_grid) <= 1.0 <= np.nanmax(mu_grid):
                    plt.contour(H_GRID, HAC_GRID, mu_grid, levels=[1.0001],
                                colors=[STAB_COLOR], linestyles=[STAB_STYLE], 
                                linewidths=[STAB_WIDTH], alpha=0.9)
                
                plt.xlim(xlims)
                plt.ylim(ylims)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()