import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sweep.py <base_prefix> [k_cutoff]")
        sys.exit(1)

    # 1. Resolve absolute path of the input data BEFORE changing directories
    raw_prefix = sys.argv[1]
    base_prefix = os.path.abspath(raw_prefix)
    
    # 2. Change the working directory to where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Optional cutoff frequency argument (defaults to infinity if not provided)
    k_cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    
    summary_file = f"{base_prefix}_summary.txt"

    if not os.path.exists(summary_file):
        print(f"Error: Summary file {summary_file} not found.")
        sys.exit(1)

    # 3. Determine output figures directory based on data directory name
    data_dir = os.path.dirname(base_prefix)
    data_dir_name = os.path.basename(data_dir) # e.g., "sweep_1000.0(1)"
    figures_dir = os.path.abspath(os.path.join("..", "figures", data_dir_name))
    
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving figures to: {figures_dir}")

    try:
        df_summary = pd.read_csv(summary_file, sep='\s+')
    except pd.errors.EmptyDataError:
        print("The summary file is empty. Please wait for the first point to finish.")
        sys.exit(0)

    if df_summary.empty:
        print("The summary file has a header but no data yet. Please wait.")
        sys.exit(0)

    df_summary = df_summary.sort_values(by='h0').reset_index(drop=True)
    print(f"Loaded {len(df_summary)} parameter points.")

    # 1. Plot Spectral Entropy (Log Scale)
    plt.figure(figsize=(8, 6))
    plt.plot(df_summary['h0'], df_summary['spectral_entropy'], marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('h0')
    plt.ylabel('Spectral Entropy (log scale)')
    plt.title('Spectral Entropy vs h0')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sweep_entropy.png"))
    plt.close()

    # 2. Plot Spatial Variance (Log Scale)
    plt.figure(figsize=(8, 6))
    plt.plot(df_summary['h0'], df_summary['var_u'], marker='s', linestyle='-', color='r')
    plt.yscale('log')
    plt.xlabel('h0')
    plt.ylabel('Spatial Variance (log scale)')
    plt.title('Spatial Variance vs h0')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sweep_variance.png"))
    plt.close()

    # Plot u_dot
    plt.figure(figsize=(8, 6))
    plt.plot(df_summary['h0'], df_summary['u_dot'], marker='^', linestyle='-', color='g')
    plt.xlabel('h0')
    plt.ylabel('Average u velocity (u_dot)')
    plt.title('Average u_dot vs h0')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sweep_u_dot.png"))
    plt.close()

    # Plot phi_dot
    plt.figure(figsize=(8, 6))
    plt.plot(df_summary['h0'], df_summary['phi_dot'], marker='v', linestyle='-', color='m')
    plt.xlabel('h0')
    plt.ylabel('Average phi velocity (phi_dot)')
    plt.title('Average phi_dot vs h0')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sweep_phi_dot.png"))
    plt.close()

    # 3. Process Power Spectra with cutoff frequency
    ps_dir = os.path.join(figures_dir, "power_spectra")
    os.makedirs(ps_dir, exist_ok=True)
    
    ps_files = glob.glob(f"{base_prefix}_ps_h0_*.txt")

    for h0 in df_summary['h0']:
        matched_file = None
        
        for f in ps_files:
            try:
                num_str = f.split('_ps_h0_')[-1].replace('.txt', '')
                f_h0 = float(num_str)
                if abs(f_h0 - h0) < 1e-4:
                    matched_file = f
                    break
            except ValueError:
                continue

        if not matched_file:
            print(f"Warning: Missing power spectrum file for h0 = {h0}")
            continue

        df_ps = pd.read_csv(matched_file, sep='\s+')
        
        # Filter out k=0, negative frequencies, and apply high-frequency cutoff
        df_ps = df_ps[(df_ps['k'] > 0) & (df_ps['k'] <= k_cutoff)]
        df_ps = df_ps.sort_values(by='k')

        if df_ps.empty:
            print(f"Warning: Cutoff {k_cutoff} filtered out all frequencies for h0 = {h0}")
            continue

        plt.figure(figsize=(8, 6))
        plt.plot(df_ps['k'], df_ps['power'], color='k', alpha=0.8)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Wavenumber (k)')
        plt.ylabel('Power')
        
        title_str = f'Power Spectrum at h0 = {h0}'
        if k_cutoff != float('inf'):
            title_str += f' (k cutoff = {k_cutoff})'
            
        plt.title(title_str)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_name = os.path.join(ps_dir, f'ps_h0_{h0}.png')
        plt.savefig(out_name)
        plt.close()

    print(f"Saved summary plots and populated power spectrum plots in directory: {ps_dir}")

if __name__ == "__main__":
    main()