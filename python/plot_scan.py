/import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot domain wall parameter scan output.")
    parser.add_argument("--input", default="scan_output.txt", help="Input file path")
    args = parser.parse_args()

    # Load data using whitespace delimiter
    df = pd.read_csv(args.input, sep=r'\s+')
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # Subplot 1: Time to settle
    axs[0].plot(df['h0'], df['time_to_settle'], marker='o', markersize=5, linestyle='-', color='tab:red')
    axs[0].set_ylabel('Time to Settle (t)')
    axs[0].set_title('Settling Time vs h0')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Average velocities
    axs[1].plot(df['h0'], df['avg_u_dot'], marker='s', markersize=5, linestyle='-', label='<u_dot>', color='tab:blue')
    axs[1].plot(df['h0'], df['avg_phi_dot'], marker='^', markersize=5, linestyle='-', label='<phi_dot>', color='tab:orange')
    axs[1].set_ylabel('Average Velocities')
    axs[1].set_title('Phase Space Velocities vs h0')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 3: Variance of u
    axs[2].plot(df['h0'], df['var_u'], marker='d', markersize=5, linestyle='-', color='tab:green')
    axs[2].set_xlabel('h0')
    axs[2].set_ylabel('Variance of u')
    axs[2].set_title('Spatial Variance of Real Part vs h0')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('scan_results.png')
    plt.show()

if __name__ == "__main__":
    main()
