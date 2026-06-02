import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

from matplotlib.widgets import Slider, CheckButtons, Button

# --- Global Font Size Configuration ---
# Sizes used while viewing the interactive plot
FONT_LABEL_UI = 12
FONT_TICK_UI = 10
FONT_TEXT_UI = 12

# Sizes used when saving the PDF
FONT_LABEL_SAVE = 30
FONT_TICK_SAVE = 22
FONT_TEXT_SAVE = 30

# --------------------------------------


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

    # Multiply phi data by -1 to match previous scripts
    combined_df["speed_mean_im"] = combined_df["speed_mean_im"] * -1

    # Calculate the new sin^2 metric
    # Calculate the nearest integer metric (epsilon_phi)
    normalized_speed = 2 * combined_df["speed_mean_im"] / 0.5
    combined_df["epsilon_phi"] = np.abs(normalized_speed - np.round(normalized_speed))

    # Prevent log(0) errors
    # Clip omega^2 data to a minimum of 10^-3
    combined_df["time_avg_var_re"] = combined_df["time_avg_var_re"].clip(lower=1e-3)

    return combined_df


def main():
    df = load_simulation_data()
    if df.empty:
        print("No data found. Exiting.")
        return

    # Extract unique values for sliders
    h0_vals = np.sort(df["h0"].unique())
    sys_sizes = np.sort(df["system_size"].unique())

    left_metrics = ["time_avg_var_re", "speed_mean_re", "speed_mean_im"]
    left_labels = [
        "$\\omega ^2$",
        "$\\left\\langle \\dot u \\right \\rangle$",
        "$\\left\\langle \\dot \\phi \\right \\rangle$",
    ]

    right_metric = "epsilon_phi"
    right_label = "$\\varepsilon_\\phi$"

    # Pre-calculate global limits for the left metrics
    left_limits = []
    for metric in left_metrics:
        vmax = np.nanmax(df[metric])
        if metric == "time_avg_var_re":
            # Ignore the 1e-20 padding when calculating global min for better scaling
            valid_min_mask = df[metric] > 1e-19
            vmin = (
                np.nanmin(df.loc[valid_min_mask, metric])
                if valid_min_mask.any()
                else 1e-3
            )
            # Multiplicative padding for log scale
            left_limits.append((max(vmin / 2, 1e-21), vmax * 2))
        else:
            vmin = np.nanmin(df[metric])
            padding = (vmax - vmin) * 0.05 if vmax > vmin else 0.1
            left_limits.append((vmin - padding, vmax + padding))

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.3, top=0.8, wspace=0.57)

    twin_axes = [ax.twinx() for ax in axes]

    # Initialize line objects for dynamic updating
    lines_left = []
    lines_right = []

    for i in range(3):
        # Primary axis (Left) - Blue
        (line_l,) = axes[i].plot(
            [], [], marker="o", linestyle="-", color="tab:blue", label=left_labels[i]
        )
        lines_left.append(line_l)
        axes[i].set_xlabel("h", fontsize=FONT_LABEL_UI)
        axes[i].set_ylabel(
            left_labels[i], color="tab:blue", fontsize=FONT_LABEL_UI, labelpad=15
        )
        axes[i].tick_params(axis="y", labelcolor="tab:blue")
        axes[i].tick_params(axis="both", labelsize=FONT_TICK_UI)

        if left_metrics[i] == "time_avg_var_re":
            axes[i].set_yscale("log")

        # Apply the global limits to the primary axis initially
        axes[i].set_ylim(left_limits[i])

        # Secondary axis (Right) - Red
        (line_r,) = twin_axes[i].plot(
            [],
            [],
            marker="s",
            linestyle="--",
            color="tab:red",
            alpha=0.7,
            label=right_label,
        )
        lines_right.append(line_r)
        twin_axes[i].set_ylabel(
            right_label, color="tab:red", fontsize=FONT_LABEL_UI, labelpad=15
        )
        twin_axes[i].tick_params(axis="y", labelcolor="tab:red")
        twin_axes[i].tick_params(axis="both", labelsize=FONT_TICK_UI)

        # Apply standard scale for sin^2 initially
        twin_axes[i].set_ylim(-0.02, 0.52)

    # Set up parameter text box at the figure level to avoid data overlap
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", lw=1, alpha=0.9)
    param_text = fig.text(
        0.5,
        0.95,
        "Loading...",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=FONT_TEXT_UI,
        bbox=bbox_props,
    )

    # Set up UI elements (Sliders and Checkbox)
    ax_h0 = plt.axes([0.15, 0.15, 0.55, 0.03])
    ax_sys = plt.axes([0.15, 0.05, 0.55, 0.03])
    ax_check = plt.axes([0.8, 0.08, 0.12, 0.1])

    slider_h0 = Slider(
        ax=ax_h0,
        label="h0 value",
        valmin=h0_vals[0],
        valmax=h0_vals[-1],
        valinit=h0_vals[0],
        valstep=h0_vals,
    )

    slider_sys = Slider(
        ax=ax_sys,
        label="$L/L_0$ (Sys Size)",
        valmin=sys_sizes[0],
        valmax=sys_sizes[-1],
        valinit=sys_sizes[0],
        valstep=sys_sizes,
    )

    check_global = CheckButtons(ax_check, ["Global Y Limits"], [True])

    def update(val):
        target_h0 = slider_h0.val
        target_sys = slider_sys.val
        use_global_limits = check_global.get_status()[0]

        # Snap to closest valid values if somehow between steps
        closest_h0 = h0_vals[np.argmin(np.abs(h0_vals - target_h0))]
        closest_sys = sys_sizes[np.argmin(np.abs(sys_sizes - target_sys))]

        # Filter the DataFrame
        mask = np.isclose(df["h0"], closest_h0, atol=1e-3) & np.isclose(
            df["system_size"], closest_sys, atol=1e-3
        )
        subset = df[mask].sort_values(by="h")

        if subset.empty:
            for i in range(3):
                lines_left[i].set_data([], [])
                lines_right[i].set_data([], [])
            param_text.set_text(
                f"No data\n$h_0 = {closest_h0:.2f}$\n$L/L_0 = {closest_sys:.1f}$"
            )
        else:
            h_data = subset["h"].values
            right_data = subset[right_metric].values

            for i, metric in enumerate(left_metrics):
                left_data = subset[metric].values

                # Update data
                lines_left[i].set_data(h_data, left_data)
                lines_right[i].set_data(h_data, right_data)

                # Dynamically adjust X axis to prevent clipping
                x_min, x_max = np.min(h_data), np.max(h_data)
                x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
                axes[i].set_xlim(x_min - x_margin, x_max + x_margin)

                # Handle Y axis based on checkbox state
                if use_global_limits:
                    axes[i].set_ylim(left_limits[i])
                    twin_axes[i].set_ylim(-0.02, 0.52)
                else:
                    # Manually calculate dynamic limits for the primary (left) axis
                    l_min, l_max = np.nanmin(left_data), np.nanmax(left_data)

                    if left_metrics[i] == "time_avg_var_re":
                        # Multiplicative padding for the log scale
                        l_min = max(l_min, 1e-3)  # Prevent zero in log scale
                        axes[i].set_ylim(l_min / 2, l_max * 2)
                    else:
                        # Standard 5% padding for linear scales
                        l_margin = 0.05 * (l_max - l_min) if l_max > l_min else 0.1
                        axes[i].set_ylim(l_min - l_margin, l_max + l_margin)

                    # Manually calculate dynamic limits for the secondary (right) axis
                    y_min, y_max = np.nanmin(right_data), np.nanmax(right_data)
                    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
                    twin_axes[i].set_ylim(y_min - y_margin, y_max + y_margin)

            param_text.set_text(
                f"$h_0 = {closest_h0:.2f}$\n$L/L_0 = {closest_sys:.1f}$"
            )

        fig.canvas.draw_idle()

    # Register callbacks
    slider_h0.on_changed(update)
    slider_sys.on_changed(update)
    check_global.on_clicked(update)

    # Initialize the plot with the starting slider values
    update(None)

    # Set up Save Button UI
    ax_save = plt.axes([0.8, 0.20, 0.12, 0.05])
    btn_save = Button(ax_save, "Save PDF")

    def save_plot(event):
        # Hide UI elements for a clean export
        for ax in (ax_h0, ax_sys, ax_check, ax_save):
            ax.set_visible(False)

        # Temporarily adjust figure size and fonts for saving
        original_size = fig.get_size_inches()
        fig.set_size_inches(24, 8)  # Increase image size

        for i in range(3):
            axes[i].xaxis.label.set_size(FONT_LABEL_SAVE)
            axes[i].yaxis.label.set_size(FONT_LABEL_SAVE)
            axes[i].tick_params(axis="both", labelsize=FONT_TICK_SAVE)

            twin_axes[i].yaxis.label.set_size(FONT_LABEL_SAVE)
            twin_axes[i].tick_params(axis="both", labelsize=FONT_TICK_SAVE)

        param_text.set_fontsize(FONT_TEXT_SAVE)

        # Get current parameters for the filename
        target_h0 = slider_h0.val
        target_sys = slider_sys.val
        closest_h0 = h0_vals[np.argmin(np.abs(h0_vals - target_h0))]
        closest_sys = sys_sizes[np.argmin(np.abs(sys_sizes - target_sys))]

        filename = f"plot_h0_{closest_h0:.2f}_L_{closest_sys:.1f}.pdf"

        # Save publication quality image
        fig.savefig(filename, dpi=300, bbox_inches="tight", format="pdf")

        # Restore original figure size and fonts
        fig.set_size_inches(original_size)

        for i in range(3):
            axes[i].xaxis.label.set_size(FONT_LABEL_UI)
            axes[i].yaxis.label.set_size(FONT_LABEL_UI)
            axes[i].tick_params(axis="both", labelsize=FONT_TICK_UI)

            twin_axes[i].yaxis.label.set_size(FONT_LABEL_UI)
            twin_axes[i].tick_params(axis="both", labelsize=FONT_TICK_UI)

        param_text.set_fontsize(FONT_TEXT_UI)

        # Restore UI elements
        for ax in (ax_h0, ax_sys, ax_check, ax_save):
            ax.set_visible(True)

        fig.canvas.draw_idle()
        print(f"Saved: {filename}")

    btn_save.on_clicked(save_plot)

    plt.show()


if __name__ == "__main__":
    main()
