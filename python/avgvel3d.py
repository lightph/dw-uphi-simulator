import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import LogNorm


def load_vel_data(filepath):
    """Loads the entire velocity dataset into RAM."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".bin":
        raw = np.fromfile(filepath, dtype=np.float64).reshape(-1, 4)
        return pd.DataFrame(raw, columns=["A", "B", "C", "AvgVelocity"])

    elif ext == ".csv":
        df = pd.read_csv(filepath)
        if len(df.columns) == 4 and df.columns[0] != "A":
            df.columns = ["A", "B", "C", "AvgVelocity"]
        return df

    else:
        raise ValueError("Velocity file must be .csv or .bin")


def load_fft_slice(filepath, b_target, c_target):
    """Lazily loads only the requested B and C slice from disk."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".bin":
        raw = np.memmap(filepath, dtype=np.float64, mode="r").reshape(-1, 5)
        mask = np.isclose(raw[:, 1], b_target, atol=1e-4) & np.isclose(
            raw[:, 2], c_target, atol=1e-4
        )
        slice_data = raw[mask]
        df = pd.DataFrame(slice_data, columns=["A", "B", "C", "Freq", "Power"])

    elif ext == ".csv":
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=500000):
            if len(chunk.columns) == 5 and chunk.columns[0] != "A":
                chunk.columns = ["A", "B", "C", "Freq", "Power"]
            mask = np.isclose(chunk["B"], b_target, atol=1e-4) & np.isclose(
                chunk["C"], c_target, atol=1e-4
            )
            chunks.append(chunk[mask])
        df = pd.concat(chunks, ignore_index=True)

    else:
        raise ValueError("FFT file must be .csv or .bin")

    return df.pivot(index="Freq", columns="A", values="Power")


def plot_interactive_sweep(df_vel, fft_file_path):
    """Creates a stacked interactive plot mapping velocity and FFTs."""
    df_vel["B_round"] = df_vel["B"].round(4)
    df_vel["C_round"] = df_vel["C"].round(4)

    b_vals = np.sort(df_vel["B_round"].unique())
    c_vals = np.sort(df_vel["C_round"].unique())

    vel_dict = {
        k: v.sort_values("A") for k, v in df_vel.groupby(["B_round", "C_round"])
    }

    fig = plt.figure(figsize=(10, 10))

    # Create axes
    ax_empty = fig.add_axes([0.12, 0.605, 0.78, 0.345])
    ax_vel = fig.add_axes([0.12, 0.220, 0.78, 0.345], sharex=ax_empty)
    ax_spec = fig.add_axes([0.12, 0.220, 0.78, 0.223], sharex=ax_empty)

    b_init = b_vals[len(b_vals) // 2]
    c_init = c_vals[len(c_vals) // 2]
    init_key = (b_init, c_init)

    # Setup Calculated Plot (Top): A - 0.5 * Velocity
    init_vel = vel_dict[init_key]
    calc_data = init_vel["A"] - (0.5 * init_vel["AvgVelocity"])
    (line_calc,) = ax_empty.plot(init_vel["A"], calc_data, lw=2, color="#d62728")
    ax_empty.set_ylabel(r"$A - \frac{1}{2}V$")
    ax_empty.grid(True, linestyle="--", alpha=0.7)

    # Setup Velocity Plot (Middle)
    (line_vel,) = ax_vel.plot(
        init_vel["A"], init_vel["AvgVelocity"], lw=2, color="#1f77b4"
    )
    ax_vel.set_ylabel("Average Velocity")
    ax_vel.set_title(f"Parameters  |  B: {b_init:.4f}  |  C: {c_init:.4f}")
    ax_vel.grid(True, linestyle="--", alpha=0.7)

    # Setup Spectrogram (Bottom)
    ax_spec.set_ylabel("Frequency")

    # Setup Sliders and Checkbox
    ax_b = plt.axes([0.15, 0.12, 0.65, 0.03])
    ax_c = plt.axes([0.15, 0.05, 0.65, 0.03])
    ax_check = plt.axes([0.85, 0.05, 0.12, 0.1])

    fig.slider_b = Slider(
        ax_b, "Parameter B", b_vals.min(), b_vals.max(), valinit=b_init, valstep=b_vals
    )
    fig.slider_c = Slider(
        ax_c, "Parameter C", c_vals.min(), c_vals.max(), valinit=c_init, valstep=c_vals
    )

    check = CheckButtons(ax_check, ["Show FFT"], [False])
    mesh = None

    def update_layout(show_fft):
        if show_fft:
            ax_empty.set_position([0.12, 0.726, 0.78, 0.223])
            ax_vel.set_position([0.12, 0.473, 0.78, 0.223])
            ax_spec.set_position([0.12, 0.220, 0.78, 0.223])
            ax_spec.set_visible(True)
            ax_empty.tick_params(labelbottom=False)
            ax_vel.tick_params(labelbottom=False)
            ax_spec.tick_params(labelbottom=True)
            ax_vel.set_xlabel("")
            ax_spec.set_xlabel("Driving Amplitude (A)")
        else:
            ax_empty.set_position([0.12, 0.605, 0.78, 0.345])
            ax_vel.set_position([0.12, 0.220, 0.78, 0.345])
            ax_spec.set_visible(False)
            ax_empty.tick_params(labelbottom=False)
            ax_vel.tick_params(labelbottom=True)
            ax_vel.set_xlabel("Driving Amplitude (A)")
            ax_spec.set_xlabel("")

    def update(val):
        nonlocal mesh
        b_curr = fig.slider_b.val
        c_curr = fig.slider_c.val
        key = (b_curr, c_curr)

        show_fft = check.get_status()[0]
        update_layout(show_fft)

        if key in vel_dict:
            curr_vel = vel_dict[key]

            # Update Velocity Plot
            line_vel.set_ydata(curr_vel["AvgVelocity"])
            line_vel.set_xdata(curr_vel["A"])
            y_min, y_max = curr_vel["AvgVelocity"].min(), curr_vel["AvgVelocity"].max()
            pad = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            ax_vel.set_ylim(y_min - pad, y_max + pad)

            # Update Calculated Plot: A - 0.5 * Velocity
            new_calc = curr_vel["A"] - (0.5 * curr_vel["AvgVelocity"])
            line_calc.set_ydata(new_calc)
            line_calc.set_xdata(curr_vel["A"])

            c_min, c_max = new_calc.min(), new_calc.max()
            c_pad = (c_max - c_min) * 0.05 if c_max != c_min else 0.1
            ax_empty.set_ylim(c_min - c_pad, c_max + c_pad)

            ax_vel.set_title(f"Parameters  |  B: {b_curr:.4f}  |  C: {c_curr:.4f}")

            if show_fft:
                curr_fft = load_fft_slice(fft_file_path, b_curr, c_curr)
                z_data = curr_fft.values
                z_min_val = z_data[z_data > 0].min() if (z_data > 0).any() else 1e-10

                if mesh is None:
                    mesh = ax_spec.pcolormesh(
                        curr_fft.columns.values,
                        curr_fft.index.values,
                        z_data,
                        cmap="viridis",
                        shading="nearest",
                        norm=LogNorm(vmin=z_min_val, vmax=z_data.max()),
                    )
                else:
                    mesh.set_array(z_data.ravel())
                    mesh.set_clim(vmin=z_min_val, vmax=z_data.max())

        fig.canvas.draw_idle()

    fig.slider_b.on_changed(update)
    fig.slider_c.on_changed(update)
    check.on_clicked(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    vel_file_path = "output/avgvel3d/20260305_114222__avgvel_sweep_3D_A1000_B10_C10.bin"
    fft_file_path = "output/fft3d/20260305_122203__fft_sweep_3D_A1000_B10_C10.bin"

    try:
        df_vel = load_vel_data(vel_file_path)
        plot_interactive_sweep(df_vel, fft_file_path)
    except Exception as e:
        print(f"Error: {e}")
