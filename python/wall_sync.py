import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.animation as manimation
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_regression

# Configuration
DATA_DIR = "build/output_data_grande"
ROW_BYTES = 7 * 8  # 7 doubles * 8 bytes per double

# Find the newest binary file
list_of_files = glob.glob(os.path.join(DATA_DIR, "*.bin"))
if not list_of_files:
    print(
        f"No binary files found in {DATA_DIR}. Please start the C++ simulation first."
    )
    exit(1)
latest_file = max(list_of_files, key=os.path.getctime)

# Global data storage
all_data = np.empty((0, 7), dtype=np.float64)
file_handle = open(latest_file, "rb")

# Setup Matplotlib Figure (1x2 layout for scatter and hexbin)
fig, (ax_scatter, ax_hexbin) = plt.subplots(1, 2, figsize=(14, 7))
plt.subplots_adjust(bottom=0.35, wspace=0.3)

# Initialize empty scatter plot line
(line_scatter,) = ax_scatter.plot([], [], "b.", markersize=8)

ax_scatter.set_ylabel(r"$\omega ^2$")
ax_scatter.grid(True, linestyle="--", alpha=0.7)

# Text box for live statistical metrics
metrics_text = fig.text(
    0.6,
    0.15,
    "",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)

# Global reference for the hexbin colorbar
cb = None

ax_h0 = plt.axes((0.15, 0.2, 0.7, 0.03))
ax_omega = plt.axes((0.15, 0.15, 0.7, 0.03))
ax_radio = plt.axes((0.15, 0.02, 0.4, 0.1))  # Left, Bottom, Width, Height

slider_h0 = Slider(ax_h0, "h0", 0.0, 9.8, valinit=0.0, valstep=0.2)
slider_omega = Slider(ax_omega, "$\\Omega$", 0.2, 1.8, valinit=0.2, valstep=0.2)

radio_metric = RadioButtons(ax_radio, ("Sin^2", "Distancia", "Derivada"))

# ==========================================
# Figure 3: Metric and Rugosity vs h
# ==========================================
fig3, ax_lines = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.15)

ax_lines_twin = ax_lines.twinx()
(line_metric,) = ax_lines.plot([], [], "b-", linewidth=1.5, label="Métrica")
(line_rugosity,) = ax_lines_twin.plot([], [], "r-", linewidth=1.5, label="$\\omega ^2$")

ax_lines.set_xlabel(r"$h$")
ax_lines.set_ylabel("Métrica", color="blue")
ax_lines_twin.set_ylabel(r"$\omega^2$", color="red")
ax_lines.grid(True, linestyle=":", alpha=0.6)

# Add a combined legend
lines_1, labels_1 = ax_lines.get_legend_handles_labels()
lines_2, labels_2 = ax_lines_twin.get_legend_handles_labels()
ax_lines.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
)


def update_plot(val=None):
    """Filters data, calculates the selected metric, and updates plots/stats."""
    global all_data, cb
    if all_data.shape[0] == 0:
        return

    target_h0 = slider_h0.val
    target_omega = slider_omega.val
    selected_metric = radio_metric.value_selected

    mask = np.isclose(all_data[:, 2], target_h0, atol=1e-3) & np.isclose(
        all_data[:, 3], target_omega, atol=1e-3
    )

    filtered_data = all_data[mask]

    if filtered_data.shape[0] > 0:
        sort_idx = np.argsort(filtered_data[:, 1])
        filtered_data = filtered_data[sort_idx]

        h_vals = filtered_data[:, 1]
        omega_sq = filtered_data[:, 4]
        phi_dot = -1 * filtered_data[:, 6]

        ratio = 2 * phi_dot / target_omega

        # Calculate x-axis based on selected metric
        if selected_metric == "Sin^2":
            x_vals = np.sin(ratio * np.pi) ** 2
            xlabel = r"$\sin^2(\pi 2 \langle \dot{\phi} \rangle / \Omega)$"

        elif selected_metric == "Distancia":
            x_vals = np.abs(ratio - np.round(ratio))
            xlabel = r"Distancia al entero más cercano de $2 \langle \dot{\phi} \rangle / \Omega$"

        elif selected_metric == "Derivada":
            if len(h_vals) > 1:
                raw_deriv = np.gradient(phi_dot, h_vals)
                window_size = min(9, len(raw_deriv))
                kernel = np.ones(window_size) / window_size
                x_vals = np.convolve(raw_deriv, kernel, mode="same")
            else:
                x_vals = np.zeros_like(phi_dot)
            xlabel = r"$\partial \langle \dot{\phi} \rangle / \partial h$"

        # --- Plot Updates ---
        # Update scatter plot
        line_scatter.set_data(x_vals, omega_sq)
        ax_scatter.set_xlabel(xlabel)
        ax_scatter.relim()
        ax_scatter.autoscale_view()

        ax_hexbin.clear()
        hb = ax_hexbin.hexbin(x_vals, omega_sq, gridsize=30, cmap="inferno", bins="log")
        ax_hexbin.set_xlabel(xlabel)
        ax_hexbin.set_ylabel(r"$\omega ^2$")

        # Initialize or update the colorbar
        if cb is None:
            cb = fig.colorbar(hb, ax=ax_hexbin, label=r"$\log_{10}(\text{cuentas})$")
        else:
            cb.update_normal(hb)

        # --- Update Figure 3 (Metric & Rugosity vs h) ---
        line_metric.set_data(h_vals, x_vals)
        line_rugosity.set_data(h_vals, omega_sq)

        # Dynamically update the blue y-axis label based on the radio button
        ax_lines.set_ylabel(xlabel, color="blue")

        # Rescale both axes
        ax_lines.relim()
        ax_lines.autoscale_view()
        ax_lines_twin.relim()
        ax_lines_twin.autoscale_view()

        # Force redraw of the third figure
        fig3.canvas.draw_idle()

        # --- Statistical Metrics ---
        # Define states for conditional prob & biserial
        tol_lock = 1e-2  # Distance to int threshold
        tol_smooth = 1e-8  # Rugosity floor

        dist_to_int = np.abs(ratio - np.round(ratio))
        is_locked = dist_to_int < tol_lock
        is_smooth = omega_sq < tol_smooth

        # Conditional Probability P(Smooth | Locked)
        locked_count = np.sum(is_locked)
        if locked_count > 0:
            p_s_given_l = np.sum(is_locked & is_smooth) / locked_count
        else:
            p_s_given_l = np.nan

        smooth_count = np.sum(is_smooth)
        if smooth_count > 0:
            p_l_given_s = np.sum(is_smooth & is_locked) / smooth_count
        else:
            p_l_given_s = np.nan

        # Point-Biserial Correlation
        if 0 < locked_count < len(is_locked):
            pb_corr, _ = pointbiserialr(is_locked, omega_sq)
        else:
            pb_corr = np.nan

        # Mutual Information
        mi = mutual_info_regression(x_vals.reshape(-1, 1), omega_sq)[0]

        # Update text box
        metrics_text.set_text(
            f"Información mutua: {mi:.3f}\n"
            f"P(Plano | Locked): {p_s_given_l:.3f}\n"
            f"P(Locked | Plano): {p_l_given_s:.3f}\n"
            f"Correlación punto-biserial: {pb_corr:.3f}"
        )


def read_and_animate(frame):
    """Reads new binary data incrementally and triggers a plot update."""
    global all_data

    # Read all currently available unread bytes
    raw_bytes = file_handle.read()
    num_full_rows = len(raw_bytes) // ROW_BYTES

    if num_full_rows > 0:
        valid_bytes = num_full_rows * ROW_BYTES
        new_array = np.frombuffer(raw_bytes[:valid_bytes], dtype=np.float64).reshape(
            -1, 7
        )
        all_data = np.vstack((all_data, new_array))

        # If there were partial bytes at the end, seek back to read them next time
        remainder = len(raw_bytes) - valid_bytes
        if remainder > 0:
            file_handle.seek(-remainder, os.SEEK_CUR)

        update_plot()
        update_metrics_plot()  # Also update the second figure

    return (line_scatter,)


# Attach update function to slider changes so it updates instantly when dragged
slider_h0.on_changed(update_plot)
slider_omega.on_changed(update_plot)
radio_metric.on_clicked(update_plot)

# Run the live animation loop (checks file every 500 ms)
ani = FuncAnimation(fig, read_and_animate, interval=500, cache_frame_data=False)

# ==========================================
# Figure 2: Interactive Statistical Metrics vs h0 Sweep
# ==========================================

fig2, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
plt.subplots_adjust(bottom=0.15, hspace=0.3)

metric_titles = [
    "Información Mutua",
    "P(Plano | Locked)",
    "P(Locked | Plano)",
    "Correlación punto-biserial",
]

lines_metrics = []
for i, ax_m in enumerate(axs):
    ax_m.set_ylabel(metric_titles[i], fontsize=10)
    ax_m.grid(True, linestyle=":", alpha=0.6)
    (line_m,) = ax_m.plot([], [], "r-o", markersize=4, linewidth=1.2)
    lines_metrics.append(line_m)

axs[-1].set_xlabel(r"$h_0$", fontsize=12)

# Independent slider for Figure 2
ax_omega2 = plt.axes((0.15, 0.05, 0.7, 0.03))
slider_omega2 = Slider(ax_omega2, "$\\Omega$", 0.2, 1.8, valinit=0.2, valstep=0.2)


def update_metrics_plot(val=None):
    if all_data.shape[0] == 0:
        return

    target_omega = slider_omega2.val
    selected_metric = radio_metric.value_selected  # Uses the radio button from Fig 1
    tol_lock = 1e-2
    tol_smooth = 1e-8

    mask_omega = np.isclose(all_data[:, 3], target_omega, atol=1e-3)
    data_omega = all_data[mask_omega]

    if data_omega.shape[0] == 0:
        return

    unique_h0 = np.unique(data_omega[:, 2])
    unique_h0.sort()

    valid_h0, mi_list, p_s_given_l_list, p_l_given_s_list, pb_corr_list = (
        [],
        [],
        [],
        [],
        [],
    )

    for h0 in unique_h0:
        mask_h0 = np.isclose(data_omega[:, 2], h0, atol=1e-3)
        subset = data_omega[mask_h0]

        if subset.shape[0] < 5:
            continue

        sort_idx = np.argsort(subset[:, 1])
        subset = subset[sort_idx]

        h_vals = subset[:, 1]
        omega_sq = subset[:, 4]
        phi_dot = -1 * subset[:, 6]
        ratio = 2 * phi_dot / target_omega

        if selected_metric == "Sin^2":
            x_vals = np.sin(ratio * np.pi) ** 2
        elif selected_metric == "Distancia":
            x_vals = np.abs(ratio - np.round(ratio))
        elif selected_metric == "Derivada":
            raw_deriv = np.gradient(phi_dot, h_vals)
            window_size = min(9, len(raw_deriv))
            kernel = np.ones(window_size) / window_size
            x_vals = np.convolve(raw_deriv, kernel, mode="same")

        dist_to_int = np.abs(ratio - np.round(ratio))
        is_locked = dist_to_int < tol_lock
        is_smooth = omega_sq < tol_smooth

        locked_count = np.sum(is_locked)
        smooth_count = np.sum(is_smooth)

        p_s_given_l = (
            np.sum(is_locked & is_smooth) / locked_count if locked_count > 0 else np.nan
        )
        p_l_given_s = (
            np.sum(is_smooth & is_locked) / smooth_count if smooth_count > 0 else np.nan
        )

        if 0 < locked_count < len(is_locked):
            pb_corr, _ = pointbiserialr(is_locked, omega_sq)
        else:
            pb_corr = np.nan

        if np.var(x_vals) > 0 and np.var(omega_sq) > 0:
            mi = mutual_info_regression(x_vals.reshape(-1, 1), omega_sq)[0]
        else:
            mi = np.nan

        valid_h0.append(h0)
        mi_list.append(mi)
        p_s_given_l_list.append(p_s_given_l)
        p_l_given_s_list.append(p_l_given_s)
        pb_corr_list.append(pb_corr)

    if not valid_h0:
        return

    lists = [mi_list, p_s_given_l_list, p_l_given_s_list, pb_corr_list]
    for i, line in enumerate(lines_metrics):
        line.set_data(valid_h0, lists[i])
        axs[i].relim()
        axs[i].autoscale_view()

    fig2.canvas.draw_idle()


# Hook up the slider and radio buttons to the new figure
slider_omega2.on_changed(update_metrics_plot)
radio_metric.on_clicked(lambda val: update_metrics_plot())

plt.show()
