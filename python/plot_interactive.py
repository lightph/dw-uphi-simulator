import numpy as np
import subprocess
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

# --- Configuration ---
executable_path = "./solveODPend"
output_file = "output/forzado/xt/run_03.csv"
cwd_build = "build"

# Check if executable exists
if not os.path.exists(os.path.join(cwd_build, executable_path)):
    # Note: Checking inside build folder since you used cwd="build"
    print(
        f"Warning: Executable not found at {os.path.join(cwd_build, executable_path)}"
    )


def sol_analitica(f):
    if f <= 1.0:
        return 0.0
    else:
        return np.sqrt(f**2 - 1)


def run_simulation(val_A, val_B, val_C):
    """Runs the external C++ executable and returns the data."""
    try:
        result = subprocess.run(
            [executable_path, str(val_A), str(val_B), str(val_C)],
            capture_output=True,
            text=True,
            cwd=cwd_build,  # Running inside the build directory
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None, None, None

        # Check if file exists before reading
        # Note: path depends on where the script is run vs where the C++ code outputs
        full_output_path = output_file
        if not os.path.exists(full_output_path):
            print(f"Output file not found: {full_output_path}")
            return None, None, None

        t, x, v = np.genfromtxt(full_output_path, delimiter=",", unpack=True)
        return t, x, v

    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None, None


# --- Plot Setup ---
# Adjust bottom to make room for sliders
fig, ax = plt.subplots(2, 1, tight_layout=False, figsize=(10, 8))
plt.subplots_adjust(bottom=0.30)

# Initial values
init_A, init_B, init_C = 1.0, 1.0, 1.0

# Run once to get initial data
t_init, x_init, v_init = run_simulation(init_A, init_B, init_C)

# If simulation fails initially, create dummy data to prevent crash
if t_init is None:
    t_init, x_init, v_init = [0], [0], [0]

# Create plot objects (We use plot with markers to look like scatter, but it updates faster)
# Plot Position
(line_x,) = ax[0].plot(
    t_init,
    x_init,
    color="red",
    linestyle="None",
    marker=".",
    markersize=4,
    alpha=0.95,
    label="Position",
)
ax[0].set_ylabel("Position (x)")
ax[0].grid(True)
ax[0].legend()

# Plot Velocity
(line_v,) = ax[1].plot(
    t_init,
    v_init,
    color="black",
    linestyle="None",
    marker=".",
    markersize=4,
    label="Velocity",
)
ax[1].set_ylabel("Velocity (v)")
ax[1].set_xlabel("Time (t)")
ax[1].grid(True)

# --- Slider Setup ---
# Define axes positions for sliders [left, bottom, width, height]
ax_A = plt.axes((0.25, 0.15, 0.65, 0.03))
ax_B = plt.axes((0.25, 0.10, 0.65, 0.03))
ax_C = plt.axes((0.25, 0.05, 0.65, 0.03))

# Create Sliders (0 to 5 range)
slider_A = Slider(ax_A, "Param A", 0.0, 5.0, valinit=init_A)
slider_B = Slider(ax_B, "Param B", 0.0, 5.0, valinit=init_B)
slider_C = Slider(ax_C, "Param C", 0.0, 5.0, valinit=init_C)


# --- Update Function ---
def update(val):
    # 1. Get values from sliders
    A = slider_A.val
    B = slider_B.val
    C = slider_C.val

    # 2. Run simulation
    t, x, v = run_simulation(A, B, C)

    if t is not None:
        # 3. Update data in the plots
        line_x.set_data(t, x)
        line_v.set_data(t, v)

        # 4. Rescale axes automatically (optional, remove if you want fixed axes)
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()

        # Update legend text if needed (optional)
        # ax[0].legend(title=f"A:{A:.2f} B:{B:.2f} C:{C:.2f}")

    # 5. Redraw the canvas
    fig.canvas.draw_idle()


# Register the update function with the sliders
slider_A.on_changed(update)
slider_B.on_changed(update)
slider_C.on_changed(update)

print("Interactive plot ready. Adjust sliders to simulate.")
plt.show()
