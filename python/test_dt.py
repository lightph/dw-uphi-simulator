import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
# Macroscopic variables to analyze
VARIABLES = ['var_u', 'u_dot', 'phi_dot', 'spectral_entropy']
TITLES = {
    'var_u': 'Spatial Variance',
    'u_dot': 'Mean u_dot',
    'phi_dot': 'Mean phi_dot',
    'spectral_entropy': 'Spectral Entropy'
}

# Use log scale on the Y-axis for these specific time series plots
LOG_Y_VARS = ['var_u', 'spectral_entropy']

# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_dt_dependence.py <data_directory>")
        sys.exit(1)

    # 1. Resolve absolute path of the input data BEFORE changing directories[cite: 4]
    raw_dir = sys.argv[1]
    base_dir = os.path.abspath(raw_dir)
    
    # 2. Change the working directory to where this script is located[cite: 4]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        sys.exit(1)

    # 3. Determine output figures directory based on data directory name[cite: 4]
    data_dir_name = os.path.basename(base_dir)
    figures_dir = os.path.abspath(os.path.join("..", "figures", data_dir_name))
    
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving figures to: {figures_dir}")

    # Locate all dt folders
    dt_folders = glob.glob(os.path.join(base_dir, "dt_*"))
    if not dt_folders:
        print(f"No 'dt_' folders found in {base_dir}.")
        sys.exit(1)

    dt_data = {}

    # Extract and load data
    print("Loading data...")
    for folder in dt_folders:
        dt_str = os.path.basename(folder).replace("dt_", "")
        try:
            dt_val = float(dt_str)
        except ValueError:
            continue
        
        summary_file = os.path.join(folder, "time_data_time_summary.txt")
        if os.path.exists(summary_file):
            try:
                # Use fast C engine and exact space separator[cite: 4]
                df = pd.read_csv(summary_file, sep=' ', engine='c')
                
                if df.empty:
                    print(f"Warning: The summary file in {folder} has a header but no data yet.")
                    continue
                    
                # Map steps to physical time
                df['time'] = df['total_steps'] * dt_val
                dt_data[dt_val] = {'folder': folder, 'df': df}
                
            except pd.errors.EmptyDataError:
                print(f"Warning: The summary file in {folder} is empty. Skipping.")
        else:
            print(f"Warning: No summary file found in {folder}")

    if not dt_data:
        print("No valid data loaded. Exiting.")
        sys.exit(1)

    # Sort dt values in descending order for legend consistency
    dt_keys = sorted(dt_data.keys(), reverse=True)

    # -------------------------------------------------------------------------
    # PLOT 1: Macroscopic Variables vs Time (Overlaying all dt runs)
    # -------------------------------------------------------------------------
    print("Generating time series plots...")
    for var in VARIABLES:
        plt.figure(figsize=(10, 6))
        
        for dt in dt_keys:
            df = dt_data[dt]['df']
            if var in df.columns:
                plt.plot(df['time'], df[var], marker='o', markersize=4, linestyle='-', label=f"dt = {dt:.5f}")
        
        plt.xscale('log', base=2)
        if var in LOG_Y_VARS:
            plt.yscale('log')
            
        plt.xlabel("Total Time")
        plt.ylabel(TITLES[var])
        plt.title(f"{TITLES[var]} vs Time for Different Time Steps")
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_file = os.path.join(figures_dir, f"time_series_{var}.png")
        plt.savefig(out_file, dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # PLOT 2: Final Spatial Power Spectra (Overlaying all dt runs)
    # -------------------------------------------------------------------------
    print("Generating final power spectra comparison...")
    plt.figure(figsize=(10, 6))
    
    for dt in dt_keys:
        folder = dt_data[dt]['folder']
        df = dt_data[dt]['df']
        
        max_step = int(df['total_steps'].max())
        ps_file = os.path.join(folder, f"time_data_ps_step_{max_step}.txt")
        
        if os.path.exists(ps_file):
            try:
                ps_df = pd.read_csv(ps_file, sep=' ', engine='c')
                ps_df = ps_df[ps_df['k'] > 0]
                
                plt.loglog(ps_df['k'], ps_df['power'], label=f"dt = {dt:.5f}", alpha=0.8)
            except pd.errors.EmptyDataError:
                pass
            
    plt.xlabel("Wavenumber (k)")
    plt.ylabel("Power")
    plt.title("Final Spatial Power Spectrum Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    ps_out_file = os.path.join(figures_dir, "final_power_spectra_comparison.png")
    plt.savefig(ps_out_file, dpi=150)
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 3: Converged Final Values vs dt
    # -------------------------------------------------------------------------
    print("Generating convergence plots vs dt...")
    
    # Sort dt in ascending order (smallest dt on the left)
    dt_asc_keys = sorted(dt_data.keys())
    
    for var in VARIABLES:
        dt_list = []
        final_vals = []
        
        for dt in dt_asc_keys:
            df = dt_data[dt]['df']
            if var in df.columns:
                dt_list.append(dt)
                final_vals.append(df[var].iloc[-1])
                
        if dt_list:
            plt.figure(figsize=(8, 6))
            plt.plot(dt_list, final_vals, marker='s', color='b', linestyle='-', markersize=6)
            
            plt.xscale('log', base=2)
            
            plt.xlabel("Time step (dt)")
            plt.ylabel(f"Final {TITLES[var]}")
            plt.title(f"Convergence of {TITLES[var]} with respect to dt")
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            conv_out_file = os.path.join(figures_dir, f"converged_{var}_vs_dt.png")
            plt.savefig(conv_out_file, dpi=150)
            plt.close()

    print(f"Done! All plots saved to {figures_dir}")

if __name__ == "__main__":
    main()