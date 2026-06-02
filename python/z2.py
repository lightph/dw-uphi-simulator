import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import math
import matplotlib.colors as colors

# Configure matplotlib for publication-quality plots
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_simulation_data():
    all_data = []
    file_paths = glob.glob("build/output_data_heatmap3_h0_*/*.bin")

    if not file_paths:
        return pd.DataFrame()

    for filepath in file_paths:
        if os.path.getsize(filepath) == 0:
            continue

        raw_array = np.fromfile(filepath, dtype=np.float64)

        complete_elements = (len(raw_array) // 8) * 8
        if complete_elements == 0:
            continue

        valid_array = raw_array[:complete_elements].reshape(-1, 8)

        df = pd.DataFrame(
            valid_array,
            columns=[
                "system_size",
                "alpha",
                "h",
                "h0",
                "Omega",
                "time_avg_var_re",
                "speed_mean_re",
                "speed_mean_im",
            ],
        )
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df["h"] = combined_df["h"].round(3)
    combined_df["system_size"] = combined_df["system_size"].round(1)
    combined_df["h0"] = combined_df["h0"].round(2)

    # Multiply phi data by -1
    combined_df["speed_mean_im"] = combined_df["speed_mean_im"] * -1

    # Calculate the nearest integer metric (epsilon_phi) with 0.25 normalization
    normalized_speed = combined_df["speed_mean_im"] / 0.25
    combined_df["epsilon_phi"] = np.abs(normalized_speed - np.round(normalized_speed))

    return combined_df


def get_grid_data(current_df, target_h0, metric, sys_sizes, h_vals):
    subset = current_df[np.isclose(current_df["h0"], target_h0, atol=1e-3)]
    if subset.empty:
        return np.full((len(sys_sizes), len(h_vals)), np.nan)

    pivot_df = subset.pivot_table(
        index="system_size", columns="h", values=metric, aggfunc="mean"
    )
    pivot_df = pivot_df.reindex(index=sys_sizes, columns=h_vals)
    return pivot_df.values


def main():
    df = load_simulation_data()
    if df.empty:
        print("No data found. Exiting.")
        return

    metrics = {
        "time_avg_var_re": "$\\omega ^2$",
        "speed_mean_re": "$\\left\\langle \\dot u \\right \\rangle$",
        "speed_mean_im": "$\\left\\langle \\dot \\phi \\right \\rangle$",
        "epsilon_phi": "$\\varepsilon_\\phi$",
    }

    h0_vals = np.sort(df["h0"].unique())
    h_vals = np.sort(df["h"].unique())
    sys_sizes = np.sort(df["system_size"].unique())

    my_cmap = plt.colormaps["magma"].copy()
    my_cmap.set_bad(color="lightgray")

    os.makedirs("figures", exist_ok=True)

    n_plots = len(h0_vals)
    # Dynamically calculate grid to prevent extremely long horizontal plots
    cols = min(3, n_plots)  # Maximum 3 columns wide
    rows = math.ceil(n_plots / cols) if n_plots > 0 else 1

    # --- Part 1: Combined figures for each metric across all h0 values ---
    for metric, metric_name in metrics.items():
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True
        )

        # Flatten axes array for easy iteration, handle case where there's only 1 plot
        if n_plots == 1:
            axes_flat = np.array([axes])
        else:
            axes_flat = axes.flatten()

        # Set appropriate norm and colorbar settings
        if metric == "time_avg_var_re":
            # Maximum contrast between 0.1 and 10 (Linear clamped scale)
            norm = colors.Normalize(vmin=0.1, vmax=10)
            cbar_extend = "both"
        elif metric == "epsilon_phi":
            # Strict 0 to 0.5 scale for distance to nearest integer
            norm = colors.Normalize(vmin=0, vmax=0.5)
            cbar_extend = "neither"
        else:
            vmin = np.nanmin(df[metric])
            vmax = np.nanmax(df[metric])
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cbar_extend = "neither"

        X, Y = np.meshgrid(h_vals, sys_sizes)
        mesh = None

        for i, h0 in enumerate(h0_vals):
            ax = axes_flat[i]
            Z = get_grid_data(df, h0, metric, sys_sizes, h_vals)

            mesh = ax.pcolormesh(X, Y, Z, cmap=my_cmap, shading="nearest", norm=norm)

            ax.set_yscale("log")
            ax.set_xlabel("h")
            ax.set_ylabel("$L/L_0$")
            ax.set_title(f"$h_0 = {h0}$")

        # Hide any unused subplots if the grid is larger than n_plots
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Add a single colorbar for the entire figure using the active axes
        if mesh is not None:
            cbar = fig.colorbar(
                mesh, ax=axes_flat[:n_plots].tolist(), shrink=0.8, extend=cbar_extend
            )
            cbar.set_label(metric_name)

        filename = f"figures/combined_{metric}.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close(fig)

    # --- Part 2: Separate plots for selected metrics for individual h0 values ---
    separate_metrics = ["time_avg_var_re", "speed_mean_re", "epsilon_phi"]

    for metric in separate_metrics:
        metric_name = metrics[metric]

        if metric == "time_avg_var_re":
            norm = colors.Normalize(vmin=0.1, vmax=10)
            cbar_extend = "both"
        elif metric == "epsilon_phi":
            # Strict 0 to 0.5 scale for distance to nearest integer
            norm = colors.Normalize(vmin=0, vmax=0.5)
            cbar_extend = "neither"
        else:
            vmin = np.nanmin(df[metric])
            vmax = np.nanmax(df[metric])
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cbar_extend = "neither"

        for h0 in h0_vals:
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            Z = get_grid_data(df, h0, metric, sys_sizes, h_vals)
            X, Y = np.meshgrid(h_vals, sys_sizes)

            mesh = ax.pcolormesh(X, Y, Z, cmap=my_cmap, shading="nearest", norm=norm)

            ax.set_yscale("log")
            ax.set_xlabel("h")
            ax.set_ylabel("$L/L_0$")
            ax.set_title(f"$h_0 = {h0}$")

            cbar = fig.colorbar(mesh, ax=ax, extend=cbar_extend)
            cbar.set_label(metric_name)

            filename = f"figures/single_{metric}_h0_{h0}.png"
            plt.savefig(filename)
            print(f"Saved: {filename}")
            plt.close(fig)


if __name__ == "__main__":
    main()
