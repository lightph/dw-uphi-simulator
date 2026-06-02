import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Constants matching the updated C++ simulation
delta_t = 0.1
time_measure = 10000
n_measure = int(time_measure / delta_t)  # 100000
n_save = n_measure // 2 + 1  # 50001
block_size = n_save + 1

# Fixed parameters for labeling
h_fixed = 2.5
Omega = 0.5 * np.sqrt(h_fixed**2 - 1.0)

# Find all generated binary files for the updated power spectrum
file_pattern = "output/pseudospectral/power_spectrum_vs_h0_v2_k1_*.bin"
files = glob.glob(file_pattern)

if not files:
    print(f"No files found matching: {file_pattern}")
    exit()

target_file = files[0]
match = re.search(r"k1_([0-9.]+)\.bin", target_file)
k1_val = float(match.group(1)) if match else "Unknown"

print(f"Loading data from: {target_file}")

# Read the entire file as a flat 1D array of 64-bit floats
raw_data = np.fromfile(target_file, dtype=np.float64)

# Verify alignment
if raw_data.size % block_size != 0:
    print(
        f"Size mismatch: Read {raw_data.size} doubles, but expected multiples of {block_size}."
    )
    print("Ensure n_measure and n_save match the C++ output exactly.")
    exit()

# Reshape the flat array into a 2D array: (number_of_h0_values, block_size)
data = raw_data.reshape((-1, block_size))

# Slice the 2D array to separate h0 values and power arrays
h0_vals = data[:, 0]
powers = data[:, 1:]

# Calculate the frequencies for the x-axis
freqs = np.fft.rfftfreq(n_measure, d=delta_t)

# To use a log scale on the x-axis, we must exclude the zero frequency
valid_idx = freqs > 0
freqs_log = freqs[valid_idx]
powers_log = powers[:, valid_idx]

# Mask zero/negative values for the logarithmic color scale
powers_log = np.maximum(powers_log, 1e-20)

# Plotting setup
fig, ax = plt.subplots(figsize=(10, 6))

# Create the colormap plot
# shading='auto' automatically infers the boundaries for the quadrilaterals
X, Y = np.meshgrid(freqs_log, h0_vals)
c = ax.pcolormesh(
    X,
    Y,
    powers_log,
    norm=LogNorm(vmin=1e-15, vmax=np.max(powers_log)),
    cmap="viridis",
    shading="auto",
)

ax.set_title(
    f"Time Power Spectrum Colormap\n$k_1$ = {k1_val} | $h = {h_fixed}$ | $\\Omega = {Omega:.4f}$"
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("$h_0$")
ax.set_xscale("log")
ax.grid(True, linestyle="--", alpha=0.3)

# Add a colorbar to indicate power levels
cbar = fig.colorbar(c, ax=ax)
cbar.set_label("Power")

plt.tight_layout()
plt.show()
