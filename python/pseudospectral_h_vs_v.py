import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Find all generated binary files
file_pattern = "output/pseudospectral/velocity_vs_h_v2_k1_*.bin"
files = glob.glob(file_pattern)

if not files:
    print(f"No files found matching: {file_pattern}")
    exit()

alpha = 0.27
# Dictionary to hold data: key = k1, value = (h_vals, v_avg)
data_dict = {}
for f in files:
    match = re.search(r"k1_([0-9.]+)\.bin", f)
    if match:
        k1 = float(match.group(1))
        raw_data = np.fromfile(f, dtype=np.float64)
        if raw_data.size > 0:
            data = raw_data.reshape((-1, 2))
            h_vals = data[:, 0]
            v_avg = data[:, 1]

            # Normalization
            v_avg /= 0.5 * (alpha**2 + 1)

            data_dict[k1] = (h_vals, v_avg)

# Sort the data by k1 to ensure the slider moves logically
k1_list = sorted(data_dict.keys())

# Analytical rigid curve setup
max_h = max(np.max(h) for h, v in data_dict.values()) if data_dict else 7.5
x_rigid = np.linspace(0, max_h, 1000)

y_rigid = np.where(
    x_rigid <= 1,
    (x_rigid / 2) * (alpha**2 + 1),
    (x_rigid / 2) * alpha**2 - 0.5 * (np.sqrt(np.maximum(x_rigid**2 - 1, 0)) - x_rigid),
)

mask_rigid = x_rigid < 1.2
if np.any(mask_rigid) and np.max(y_rigid[mask_rigid]) != 0:
    y_rigid /= np.max(y_rigid[mask_rigid])

# Plotting setup
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

initial_k1 = k1_list[0]
h_init, v_init = data_dict[initial_k1]

(line_sim,) = ax.plot(
    h_init,
    v_init,
    marker="o",
    linestyle="-",
    markersize=4,
    color="teal",
    label="Simulation",
)
(line_rigid,) = ax.plot(
    x_rigid, y_rigid, linestyle="--", color="gray", label="Rigid Limit"
)

ax.set_title(f"Average Wall Velocity vs. Forcing Parameter h\n$k_1$ = {initial_k1:.4f}")
ax.set_xlabel("h")
ax.set_ylabel("Average Velocity <u>")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend()

# Slider setup
ax_k1 = plt.axes((0.2, 0.1, 0.65, 0.03))
slider_idx = Slider(
    ax=ax_k1,
    label="$k_1$ Index",
    valmin=0,
    valmax=len(k1_list) - 1,
    valinit=0,
    valstep=1,
)

# Text annotation to show the corresponding system size L next to the slider
L_initial = 2 * np.pi / initial_k1
L_text = plt.text(
    1.05,
    0.5,
    f"L = {L_initial:.1f}",
    transform=ax_k1.transAxes,
    verticalalignment="center",
)


def update(val):
    idx = int(slider_idx.val)
    k1 = k1_list[idx]
    h, v = data_dict[k1]

    line_sim.set_data(h, v)
    ax.set_title(f"Average Wall Velocity vs. Forcing Parameter h\n$k_1$ = {k1:.4f}")
    L_text.set_text(f"L = {2 * np.pi / k1:.1f}")

    # Dynamically adjust y-limits to fit the new data curve if necessary
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

    fig.canvas.draw_idle()


slider_idx.on_changed(update)
plt.show()
