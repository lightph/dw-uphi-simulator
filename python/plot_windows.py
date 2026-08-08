import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_convergence(filename):
    df = pd.read_csv(filename, sep=' ')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    
    # --- Spatial Variance ---
    axes[0, 0].plot(df['time'], df['mean_variance'], color='tab:blue')
    axes[0, 0].set_ylabel('Mean Spatial Variance')
    axes[0, 0].set_title('Macroscopic Scalar Tracking')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 0].plot(df['time'], df['var_drift'], color='tab:blue')
    axes[1, 0].set_ylabel('Variance Drift')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, which='both', linestyle='--', alpha=0.7)

    # --- Spectral Entropy ---
    axes[0, 1].plot(np.array(df['time'])[10,], np.array(df['spectral_entropy'])[10,], color='tab:purple')
    axes[0, 1].set_ylabel('Spectral Entropy (Normalized)')
    axes[0, 1].set_title('Fourier Space Disorder Tracking')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    axes[1, 1].plot(np.array(df['time'])[10,], np.array(df['ent_drift'])[10,], color='tab:purple')
    axes[1, 1].set_ylabel('Entropy Drift')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("windows.png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metric comparison.")
    parser.add_argument("filename", type=str, help="Path to the output file")
    args = parser.parse_args()
    
    plot_convergence(args.filename)
