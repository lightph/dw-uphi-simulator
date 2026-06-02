import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

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
print(f"Monitoring file: {latest_file}")

# Global data storage
file_handle = open(latest_file, "rb")
raw_bytes = file_handle.read()
num_full_rows = len(raw_bytes) // ROW_BYTES

if num_full_rows > 0:
    valid_bytes = num_full_rows * ROW_BYTES
    all_data = np.frombuffer(raw_bytes[:valid_bytes], dtype=np.float64).reshape(-1, 7)
    remainder = len(raw_bytes) - valid_bytes
    if remainder > 0:
        file_handle.seek(-remainder, os.SEEK_CUR)
else:
    all_data = np.empty((0, 7), dtype=np.float64)

# Setup Matplotlib Figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
plt.subplots_adjust(bottom=0.25, hspace=0.1)

# Initialize empty plot lines
(line_var,) = ax1.plot([], [], "b.-", markersize=6)
(line_spre,) = ax2.plot([], [], "r.-", markersize=6)
(line_spim,) = ax3.plot([], [], "g.-", markersize=6)

# Initialize theoretical reference lines
(line_spre_theory,) = ax2.plot([], [], "k--", alpha=0.7)
(line_spim_theory,) = ax3.plot([], [], "k--", alpha=0.7)

ax1.set_ylabel("$\\omega ^2$")
ax2.set_ylabel("$\\left\\langle \\dot u \\right\\rangle$")
ax3.set_ylabel("$\\left\\langle \\dot \\phi \\right\\rangle$")
ax3.set_xlabel("h")

# Add legends for the axes with theoretical lines
ax2.legend(loc="upper left")
ax3.legend(loc="upper left")

for ax in (ax1, ax2, ax3):
    ax.grid(True, linestyle="--", alpha=0.7)

# Setup Sliders
ax_h0 = plt.axes((0.15, 0.1, 0.7, 0.03))
ax_omega = plt.axes((0.15, 0.05, 0.7, 0.03))

# Set slider bounds to match your C++ sweep ranges
# Automatically detect bounds and step sizes from initial data
if all_data.shape[0] > 0:
    # Round to handle minor floating point variations in the sweep parameters
    h0_vals = np.unique(np.round(all_data[:, 2], decimals=5))
    omega_vals = np.unique(np.round(all_data[:, 3], decimals=5))

    h0_min, h0_max = h0_vals.min(), h0_vals.max()
    omega_min, omega_max = omega_vals.min(), omega_vals.max()

    h0_step = h0_vals[1] - h0_vals[0] if len(h0_vals) > 1 else 0.1
    omega_step = omega_vals[1] - omega_vals[0] if len(omega_vals) > 1 else 0.1
else:
    # Fallback bounds if the file is completely empty at startup
    h0_min, h0_max, h0_step = 0.0, 10.0, 0.2
    omega_min, omega_max, omega_step = 0.0, 2.0, 0.2

slider_h0 = Slider(ax_h0, "h0", h0_min, h0_max, valinit=h0_min, valstep=h0_step)
slider_omega = Slider(
    ax_omega, "Omega", omega_min, omega_max, valinit=omega_min, valstep=omega_step
)


def update_plot(val=None):
    """Filters data based on sliders and updates the plot lines."""
    if all_data.shape[0] == 0:
        return

    target_h0 = slider_h0.val
    target_omega = slider_omega.val

    # Filter data using a tolerance for floating point comparisons
    mask = np.isclose(all_data[:, 2], target_h0, atol=1e-3) & np.isclose(
        all_data[:, 3], target_omega, atol=1e-3
    )

    filtered_data = all_data[mask]

    # Sort by 'h' (column 1) to ensure the line plots correctly
    if filtered_data.shape[0] > 0:
        filtered_data = filtered_data[filtered_data[:, 1].argsort()]

        h_vals = filtered_data[:, 1]

        # Extract alpha from the first row of the filtered data (column 0)
        alpha = filtered_data[0, 0]

        # Calculate theoretical curves
        # np.maximum prevents invalid sqrt calculations caused by floating point noise near h=1
        phi_vel_theory = np.where(
            h_vals > 1, alpha * 0.5 * np.sqrt(np.maximum(h_vals**2 - 1, 0)), 0.0
        )
        u_vel_theory = 0.5 * h_vals * (alpha**2 + 1) - phi_vel_theory / alpha

        # Update theoretical lines
        line_spre_theory.set_data(h_vals, u_vel_theory)
        line_spim_theory.set_data(h_vals, phi_vel_theory)

    # Update simulation line data
    line_var.set_data(filtered_data[:, 1], filtered_data[:, 4])
    line_spre.set_data(filtered_data[:, 1], filtered_data[:, 5])
    line_spim.set_data(filtered_data[:, 1], -1 * filtered_data[:, 6])

    # Dynamically rescale axes
    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()


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

    return line_var, line_spre, line_spim, line_spre_theory, line_spim_theory


# Attach update function to slider changes so it updates instantly when dragged
slider_h0.on_changed(update_plot)
slider_omega.on_changed(update_plot)

# Run the live animation loop (checks file every 500 ms)
ani = FuncAnimation(fig, read_and_animate, interval=500, cache_frame_data=False)

plt.show()
exit()

import matplotlib.animation as manimation

# 1. Reset and read the full dataset
file_handle.seek(0)
all_data = np.fromfile(file_handle, dtype=np.float64).reshape(-1, 7)

# 2. Global Aesthetic & Range Setup
# Determine absolute maximums for consistent axes in the video
max_var = np.max(all_data[:, 4])
max_u_vel = np.max(all_data[:, 5])
min_u_vel = np.min(all_data[:, 5])
max_phi_vel = np.max(-all_data[:, 6])

ax1.set_ylim(0, max_var * 1.1)
ax2.set_ylim(min_u_vel - 0.1, max_u_vel * 1.1)
ax3.set_ylim(0, max_phi_vel * 1.1)

# Style refinement: Thinner grid, cleaner line markers
for ax in (ax1, ax2, ax3):
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    for line in ax.get_lines():
        line.set_linewidth(1.2)
        line.set_markersize(4)

# 3. Horizontal Lines for Phase-Locking (Omega/2 multiples)
omega_lines = []


def refresh_omega_lines(current_omega):
    global omega_lines
    # Remove old lines
    for line in omega_lines:
        line.remove()
    omega_lines = []

    # Calculate multiples of Omega/2 that fall within the current view
    step = current_omega / 2.0
    num_lines = int(max_phi_vel / step) + 1
    for n in range(num_lines):
        line = ax3.axhline(
            n * step, color="red", linestyle=":", alpha=0.6, linewidth=1.5
        )
        omega_lines.append(line)


# 4. Video Generation Loop
# Selects the first available Omega value dynamically, or defaults to 0.2
fixed_omega = omega_vals[0] if "omega_vals" in locals() and len(omega_vals) > 0 else 0.2
h0_sweep = np.unique(all_data[np.isclose(all_data[:, 3], fixed_omega, atol=1e-3)][:, 2])
h0_sweep.sort()

# Hide UI elements for a "clean" render
ax_h0.set_visible(False)
ax_omega.set_visible(False)

# Writer Configuration
metadata = dict(title="Slonczewski Wall Sweep", artist="Matplotlib")
writer = manimation.FFMpegWriter(fps=10, metadata=metadata)

print(f"Generating video for Omega={fixed_omega}...")

with writer.saving(fig, "sweep_animation_grande.mp4", dpi=200):
    for val_h0 in h0_sweep:
        # Update parameters
        slider_h0.set_val(val_h0)
        slider_omega.set_val(fixed_omega)

        # Refresh the Omega/2 markers and the data
        refresh_omega_lines(fixed_omega)
        update_plot()

        # Dynamic Title
        fig.suptitle(
            f"Fixed $\\Omega = {fixed_omega}$ | Sweeping $h_0 = {val_h0:.2f}$",
            fontsize=12,
            fontweight="bold",
            y=0.95,
        )

        writer.grab_frame()

print("Video saved as sweep_animation.mp4")
