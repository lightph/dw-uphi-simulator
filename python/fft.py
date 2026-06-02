import numpy as np
import pyODPend
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

# Parameters
testsA = np.linspace(0.0, 3.0, 500)
block_size = 100
h = 0.001

# Choose specific values for B and C for this slice
B_fixed = 1.0
C_fixed = 1.0
B_list = np.full(block_size, B_fixed)
C_list = np.full(block_size, C_fixed)

# We need to run one batch first to determine the frequency resolution/size
sample_data = pyODPend.fourier_batch(
    testsA[:block_size], B_list, C_list, h, 200, 20, 0.0
)
freqs = 2 * np.pi * np.array(sample_data[0][1:])
num_freqs = len(freqs)

# Initialize the heatmap grid: Rows = Frequencies, Cols = A values
heatmap_grid = np.zeros((num_freqs, len(testsA)))

# Batch processing
for i in range(0, len(testsA), block_size):
    A_block = testsA[i : i + block_size]
    # Ensure B and C lists match block size (especially for the last remainder block)
    current_batch_size = len(A_block)
    data_list = pyODPend.fourier_batch(
        A_block,
        B_list[:current_batch_size],
        C_list[:current_batch_size],
        h,
        200,
        20,
        0.0,
    )

    for j in range(0, len(data_list), 2):
        power = np.array(data_list[j + 1][1:])

        # Calculate the global column index for parameter A
        a_col_idx = i + (j // 2)
        heatmap_grid[:, a_col_idx] = power

# Plotting
plt.figure(figsize=(12, 7))

# pcolormesh is often better than imshow for mapping physical axes directly
pcm = plt.pcolormesh(
    testsA, freqs, heatmap_grid, shading="auto", norm=LogNorm(), cmap="magma"
)

plt.colorbar(pcm, label="Power (log scale)")
plt.xlabel("Parameter A")
plt.ylabel("Frequency (Hz)")
plt.title(f"Spectral Heatmap (B={B_fixed}, C={C_fixed})")

# Optional: Limit frequency range if it's too high (e.g., Nyquist limit)
plt.ylim(0, 10.0)

plt.show()
