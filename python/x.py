import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Find all generated v3 binary files
file_pattern = "output/pseudospectral/velocity_vs_h_h0_v3_k1_*.bin"
files = glob.glob(file_pattern)

if not files:
    print(f"No files found matching: {file_pattern}")
    exit()

alpha = 0.27
data_dict = {}

for f in files:
    match = re.search(r"k1_([0-9.]+)\.bin", f)
    if match:
        k1 = float(match.group(1))
        L = 2 * np.pi / k1

        raw_data = np.fromfile(f, dtype=np.float64)

        # Safeguard: check if data can be reshaped into 3 columns
        if raw_data.size == 0 or raw_data.size % 3 != 0:
            print(f"Skipping {f}: Invalid data size {raw_data.size}.")
            continue

        data = raw_data.reshape((-1, 3))
        h_vals = data[:, 0]
        h0_vals = data[:, 1]
        v_avg = data[:, 2]

        unique_h = np.unique(h_vals)
        unique_h0 = np.unique(h0_vals)

        # Safeguard: check if data forms a complete grid
        expected_points = len(unique_h) * len(unique_h0)
        if len(h_vals) != expected_points:
            print(
                f"Skipping {f}: Grid mismatch. Expected {expected_points} points, got {len(h_vals)}."
            )
            continue

        # Normalization
        v_avg /= 0.5 * (alpha**2 + 1)

        # Transposing maps h to columns and h0 to rows
        V_grid = v_avg.reshape(len(unique_h), len(unique_h0)).T

        L_key = round(L, 4)
        data_dict[L_key] = (unique_h, unique_h0, V_grid, k1)

if not data_dict:
    print("No valid files successfully processed.")
    exit()

L_list = sorted(data_dict.keys())

# Plotting setup
fig, ax = plt.subplots(figsize=(10, 7))
# Adjust margins to fit 3 sliders and 1 radio button group
plt.subplots_adjust(bottom=0.45, left=0.25)

initial_L = L_list[0]
unique_h, unique_h0, V_grid, initial_k1 = data_dict[initial_L]
initial_h0_idx = 0
initial_h_idx = 0
initial_h0 = unique_h0[initial_h0_idx]
initial_h = unique_h[initial_h_idx]

(line_sim,) = ax.plot(
    unique_h,
    V_grid[initial_h0_idx, :],
    marker="o",
    linestyle="-",
    color="teal",
    label="Simulation",
)

ax.set_title(
    f"Average Wall Velocity vs. h\n$L$ = {initial_L:.4f} | $h_0$ = {initial_h0:.4f}"
)
ax.set_xlabel("Forcing Parameter h")
ax.set_ylabel("Normalized Average Velocity <u>")
ax.grid(True, linestyle="--", alpha=0.7)

# Radio Buttons for Mode Selection
ax_radio = plt.axes((0.02, 0.25, 0.15, 0.15))
radio = RadioButtons(ax_radio, ("vs h", "vs h0"))

# Slider setup for L
ax_L = plt.axes((0.25, 0.25, 0.6, 0.03))
slider_L = Slider(
    ax_L, "System Size $L$ Index", 0, len(L_list) - 1, valinit=0, valstep=1
)
k1_text = plt.text(
    1.05,
    0.5,
    f"$k_1$ = {initial_k1:.4f}",
    transform=ax_L.transAxes,
    verticalalignment="center",
)

# Slider setup for h0
ax_h0 = plt.axes((0.25, 0.15, 0.6, 0.03))
slider_h0 = Slider(
    ax_h0, "$h_0$ Index", 0, len(unique_h0) - 1, valinit=initial_h0_idx, valstep=1
)
h0_val_text = plt.text(
    1.05,
    0.5,
    f"$h_0$ = {initial_h0:.4f}",
    transform=ax_h0.transAxes,
    verticalalignment="center",
)

# Slider setup for h
ax_h = plt.axes((0.25, 0.05, 0.6, 0.03))
slider_h = Slider(
    ax_h, "$h$ Index", 0, len(unique_h) - 1, valinit=initial_h_idx, valstep=1
)
h_val_text = plt.text(
    1.05,
    0.5,
    f"$h$ = {initial_h:.4f}",
    transform=ax_h.transAxes,
    verticalalignment="center",
)


def update(val):
    L_idx = int(slider_L.val)
    h0_idx = int(slider_h0.val)
    h_idx = int(slider_h.val)
    mode = radio.value_selected

    current_L = L_list[L_idx]
    u_h, u_h0, V, current_k1 = data_dict[current_L]

    h0_idx = min(h0_idx, len(u_h0) - 1)
    h_idx = min(h_idx, len(u_h) - 1)

    current_h0 = u_h0[h0_idx]
    current_h = u_h[h_idx]

    if mode == "vs h":
        line_sim.set_data(u_h, V[h0_idx, :])
        ax.set_title(
            f"Average Wall Velocity vs. h\n$L$ = {current_L:.4f} | $h_0$ = {current_h0:.4f}"
        )
        ax.set_xlabel("Forcing Parameter h")
    else:
        # V_grid uses h0 as rows and h as columns
        line_sim.set_data(u_h0, V[:, h_idx])
        ax.set_title(
            f"Average Wall Velocity vs. $h_0$\n$L$ = {current_L:.4f} | $h$ = {current_h:.4f}"
        )
        ax.set_xlabel("Forcing Parameter $h_0$")

    k1_text.set_text(f"$k_1$ = {current_k1:.4f}")
    h0_val_text.set_text(f"$h_0$ = {current_h0:.4f}")
    h_val_text.set_text(f"$h$ = {current_h:.4f}")

    ax.relim()
    ax.autoscale_view(scalex=True, scaley=True)
    fig.canvas.draw_idle()


slider_L.on_changed(update)
slider_h0.on_changed(update)
slider_h.on_changed(update)
radio.on_clicked(update)

plt.show()
