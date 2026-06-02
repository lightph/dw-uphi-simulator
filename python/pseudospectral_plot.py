import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Simulation parameters matching the C++ code
n_steps = 10000
vec_size = 2**14

# Use memory mapping to read directly from disk without loading the full 2.6 GB into RAM
zk_history = np.memmap(
    "output/pseudospectral/spectrum_history.bin",
    dtype=np.complex128,
    mode="r",
    shape=(n_steps, vec_size),
)

# Downsample parameters
t_skip = 10
k_skip = 8

# Extract time slices first, keeping all frequencies to maintain symmetry
zk_t_sub = zk_history[::t_skip, :]

# To find Z_{-k} in standard FFT layout, reverse the array from index 1 to the end
Z_k = zk_t_sub
Z_minus_k = np.empty_like(Z_k)
Z_minus_k[:, 0] = Z_k[:, 0]
Z_minus_k[:, 1:] = np.flip(Z_k[:, 1:], axis=1)

# Apply the updated symmetry formulas for z = u - i*phi
U_k = 0.5 * (Z_k + np.conj(Z_minus_k))
Phi_k = 0.5j * (Z_k - np.conj(Z_minus_k))

# Keep only positive frequencies and downsample
half_vec = vec_size // 2
u_magnitude = np.abs(U_k[:, :half_vec:k_skip])
phi_magnitude = np.abs(Phi_k[:, :half_vec:k_skip])

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Spectrum of u
im1 = ax1.imshow(
    u_magnitude,
    aspect="auto",
    cmap="magma",
    origin="lower",
    extent=(0, half_vec, 0, n_steps),
    norm=LogNorm(),
)
ax1.set_title("Spectrum of u")
ax1.set_xlabel("Wave Number Index (Positive)")
ax1.set_ylabel("Time Step (n)")
fig.colorbar(im1, ax=ax1, label="Magnitude $|U_k|$")

# Plot Spectrum of phi
im2 = ax2.imshow(
    phi_magnitude,
    aspect="auto",
    cmap="magma",
    origin="lower",
    extent=(0, half_vec, 0, n_steps),
    norm=LogNorm(),
)
ax2.set_title(r"Spectrum of $\phi$")
ax2.set_xlabel("Wave Number Index (Positive)")
fig.colorbar(im2, ax=ax2, label=r"Magnitude $|\Phi_k|$")

plt.tight_layout()
plt.show()


from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import pyfftw

# Create figure for the animation
fig, ax = plt.subplots(figsize=(10, 6))

# Define the spatial domain (using array indices since physical length isn't specified)
x = np.arange(vec_size)
ax.set_xlim(0, vec_size)

# Calculate total frames for the animation based on your skip rate
total_frames = n_steps // t_skip

# Pre-allocate aligned array and build the FFTW plan for optimal performance
fftw_in = pyfftw.empty_aligned(vec_size, dtype=np.complex128)
ifft_plan = pyfftw.builders.ifft(fftw_in)

# Pre-calculate the global maximum to ensure the y-scale fits every step
global_y_max = 0
for frame in range(total_frames):
    # Load data directly into the aligned array
    fftw_in[:] = zk_history[frame * t_skip]

    # Execute the FFTW plan
    z_x = ifft_plan()

    u = np.real(z_x)
    y = u - np.mean(u)
    frame_max = np.max(np.abs(y))
    if frame_max > global_y_max:
        global_y_max = frame_max

if global_y_max == 0:
    global_y_max = 1.0

# Set y-limits based on the global maximum with a 10% margin
ax.set_ylim(-global_y_max * 1.1, global_y_max * 1.1)

# Initialize LineCollection with empty segments
points = np.array([x, np.zeros_like(x)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
lc = LineCollection(segments, cmap="hsv", norm=norm)
ax.add_collection(lc)

cbar = plt.colorbar(lc, ax=ax, label="Phase phi(x)")
ax.set_xlabel("x (Spatial Index)")
ax.set_ylabel("u(x) - <u>")
ax.set_title("Evolution of the Wall in Real Space")


def update(frame):
    # Read the downsampled frame directly into the FFTW input array
    fftw_in[:] = zk_history[frame * t_skip]

    # Transform back to real space using the pre-calculated plan
    z_x = ifft_plan()

    # Extract u and phi from z = u - i*phi
    u = np.real(z_x)
    phi = -np.imag(z_x)

    # Wrap phi modulo 2*pi to fit within the colormap range
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    # Subtract spatial mean
    y = u - np.mean(u)

    # Update line segments for the current frame
    current_points = np.array([x, y]).T.reshape(-1, 1, 2)
    current_segments = np.concatenate([current_points[:-1], current_points[1:]], axis=1)

    lc.set_segments(current_segments)
    lc.set_array(phi[:-1])

    return (lc,)


# Run the animation
ani = FuncAnimation(fig, update, frames=total_frames, interval=30, blit=True)
plt.tight_layout()

# Save the animation as an MP4 file
ani.save("wall_evolution.mp4", writer="ffmpeg", fps=30, dpi=150)
plt.show()
