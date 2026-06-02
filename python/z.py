import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import pandas as pd
import glob
import sys
import os
import matplotlib.colors as colors
import matplotlib.ticker as ticker


def load_simulation_data():
    all_data = []
    file_paths = glob.glob("build/output_data_heatmap3_h0_*/*.bin")

    if not file_paths:
        return pd.DataFrame()

    for filepath in file_paths:
        if os.path.getsize(filepath) == 0:
            continue

        raw_array = np.fromfile(filepath, dtype=np.float64)

        # Calculate how many complete 8-column rows exist
        complete_elements = (len(raw_array) // 8) * 8
        if complete_elements == 0:
            continue

        # Slice the array to drop any incomplete trailing data before reshaping
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

    # Round floating point columns once to group identical loop steps perfectly
    combined_df["h"] = combined_df["h"].round(3)
    combined_df["system_size"] = combined_df["system_size"].round(1)
    combined_df["h0"] = combined_df["h0"].round(2)

    combined_df["time_avg_var_re_log"] = np.log10(combined_df["time_avg_var_re"])

    return combined_df


def main():
    metrics = ["time_avg_var_re", "speed_mean_re", "speed_mean_im"]
    metrics_names = [
        "$\\omega ^2$",
        "$\\left\\langle \\dot u \\right \\rangle$",
        "$\\left\\langle \\dot \\phi \\right \\rangle$",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    title_text = fig.suptitle("Simulation Metrics Loading...", fontsize=14)

    # Dictionary to maintain state across animation frames
    state = {
        "df": pd.DataFrame(),
        "hs": np.array([]),
        "system_sizes": np.array([]),
        "h0_vals": np.array([]),
        "meshes": [None, None, None],
        "cbars": [None, None, None],
    }

    # Setup Slider with placeholders; boundaries update dynamically when data arrives
    ax_slider = plt.axes((0.2, 0.1, 0.5, 0.03))
    h0_slider = Slider(
        ax=ax_slider,
        label="h0 value",
        valmin=0.0,
        valmax=1.0,
        valinit=0.0,
    )

    def get_grid_data(current_df, target_h0, metric, sys_sizes, h_vals):
        subset = current_df[np.isclose(current_df["h0"], target_h0, atol=1e-3)]
        if subset.empty:
            return np.full((len(sys_sizes), len(h_vals)), np.nan)

        # pivot_table with aggfunc='mean' prevents crashes if the simulation
        # temporarily writes duplicate coordinate rows while actively running
        pivot_df = subset.pivot_table(
            index="system_size", columns="h", values=metric, aggfunc="mean"
        )
        pivot_df = pivot_df.reindex(index=sys_sizes, columns=h_vals)
        return pivot_df.values

    my_cmap = plt.colormaps["magma"].copy()
    my_cmap.set_bad(color="lightgray")

    def update_plot(frame=None):
        new_df = load_simulation_data()
        if new_df.empty:
            return

        state["df"] = new_df

        # Detect new dimensions dynamically
        new_h0_vals = np.sort(new_df["h0"].unique())
        new_hs = np.sort(new_df["h"].unique())
        new_system_sizes = np.sort(new_df["system_size"].unique())

        # Update slider limits if new h0 values are detected
        if not np.array_equal(new_h0_vals, state["h0_vals"]):
            state["h0_vals"] = new_h0_vals
            h0_slider.valmin = new_h0_vals[0]
            h0_slider.valmax = new_h0_vals[-1]
            h0_slider.valstep = new_h0_vals
            h0_slider.ax.set_xlim(new_h0_vals[0], new_h0_vals[-1])

            # Snap to the closest valid h0 if the current value is missing
            if h0_slider.val not in new_h0_vals:
                closest = new_h0_vals[np.argmin(np.abs(new_h0_vals - h0_slider.val))]
                h0_slider.set_val(closest)

        target_h0 = h0_slider.val

        # Check if grid dimensions expanded
        shape_changed = not (
            np.array_equal(new_hs, state["hs"])
            and np.array_equal(new_system_sizes, state["system_sizes"])
        )

        if shape_changed:
            state["hs"] = new_hs
            state["system_sizes"] = new_system_sizes

        # Check if we only have a single system size row
        is_1d = len(state["system_sizes"]) == 1
        plot_mode = "1d" if is_1d else "2d"

        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            Z = get_grid_data(
                state["df"], target_h0, metric, state["system_sizes"], state["hs"]
            ).copy()

            if metric == "time_avg_var_re":
                Z = Z + 1e-30

            prev_mode = state.get(f"mode_{i}", None)

            # Rebuild axes if shape changed, on first render, or when switching between 1D and 2D
            if shape_changed or state["meshes"][i] is None or plot_mode != prev_mode:
                ax.clear()

                # Clean up existing colorbars if switching to 1D
                if state["cbars"][i] is not None:
                    state["cbars"][i].remove()
                    state["cbars"][i] = None

                state[f"mode_{i}"] = plot_mode

                if is_1d:
                    # 1D Plot vs h
                    (line,) = ax.plot(state["hs"], Z[0], marker="o", linestyle="-")
                    state["meshes"][i] = line

                    ax.set_xlabel("h")
                    ax.set_ylabel(metrics_names[i])
                    ax.set_title(f"{metrics_names[i]}")

                    if metric == "time_avg_var_re":
                        ax.set_yscale("log")

                    # Add textbox indicating system size
                    sys_size = state["system_sizes"][0]
                    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
                    ax.text(
                        0.05,
                        0.95,
                        f"$L/L_0$: {sys_size}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=props,
                    )
                else:
                    # 2D Heatmap
                    X, Y = np.meshgrid(state["hs"], state["system_sizes"])
                    vmin, vmax = 0, 1
                    if new_df[metric].notna().any():
                        vmin, vmax = (
                            np.nanmin(new_df[metric]),
                            np.nanmax(new_df[metric]),
                        )

                    mesh = ax.pcolormesh(
                        X,
                        Y,
                        Z,
                        cmap=my_cmap,
                        shading="nearest",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    state["meshes"][i] = mesh
                    ax.set_ylabel("$L/L_0$")
                    ax.set_yscale("log")
                    ax.set_xlabel("h")
                    ax.set_title(f"{metrics_names[i]}")
                    # ax.set_ylim(90, 325)

                    state["cbars"][i] = fig.colorbar(mesh, ax=ax)
                    state["cbars"][i].set_label(metrics_names[i])

            else:
                # Fast update path
                if is_1d:
                    state["meshes"][i].set_ydata(Z[0])
                    state["meshes"][i].set_xdata(state["hs"])
                    ax.relim()
                    ax.autoscale_view()
                else:
                    vmin, vmax = 0, 1
                    if new_df[metric].notna().any():
                        vmin, vmax = (
                            np.nanmin(new_df[metric]),
                            np.nanmax(new_df[metric]),
                        )
                    state["meshes"][i].set_array(Z.ravel())
                    state["meshes"][i].set_clim(vmin=vmin, vmax=vmax)

        title_text.set_text(f"Simulation Metrics (h0 = {target_h0:.1f})")
        fig.canvas.draw_idle()

    h0_slider.on_changed(lambda val: update_plot())

    # FuncAnimation handles automated periodic background polling
    # interval=2000 means it checks the disk every 2000 milliseconds (2 seconds)
    ani = FuncAnimation(fig, update_plot, interval=2000, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    main()
