#!/usr/bin/env python3
r"""
average_nc_ew_ensemble.py

Computes the ensemble average of the structure factor S_\kappa(t) and
interface roughness w^2(t) across multiple independent random realization
seeds for Figure 6 (Edwards-Wilkinson Universal Scaling Collapse).

Usage:
    python3 python/average_nc_ew_ensemble.py <ensemble_dir> [--out-dir <out_dir>] [--plot]

Outputs:
    <ensemble_dir>/ensemble_averaged/
        time_data_ps_step_<step>.txt (ensemble-averaged power spectra)
        time_data_transient_w2.txt   (ensemble-averaged roughness)
        parameters.txt
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np


def average_ensemble(ensemble_dir, out_dir=None):
    ensemble_path = Path(ensemble_dir).resolve()
    if not ensemble_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {ensemble_dir}")

    # Discover seed directories
    seed_dirs = sorted([d for d in ensemble_path.glob("seed_*") if d.is_dir()])
    if not seed_dirs:
        # Check if current directory has seed files directly
        raise FileNotFoundError(f"No seed_* subdirectories found in {ensemble_dir}")

    n_seeds = len(seed_dirs)
    print(f"Discovered {n_seeds} ensemble seeds in {ensemble_path}:")
    for d in seed_dirs:
        print(f"  - {d.name}")

    if out_dir is None:
        out_path = ensemble_path / "ensemble_averaged"
    else:
        out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    prefix_base = out_path / "time_data"

    # 1. Process Power Spectra
    # Find all unique steps across all seeds
    step_sets = []
    for d in seed_dirs:
        ps_files = list(d.glob("*_ps_step_*.txt"))
        steps = set()
        for f in ps_files:
            if "inst" in f.name:
                continue
            try:
                s = int(f.stem.split("_step_")[-1])
                steps.add(s)
            except ValueError:
                pass
        step_sets.append(steps)

    all_steps = sorted(list(set.union(*step_sets))) if step_sets else []
    print(f"\nAveraging {len(all_steps)} spectrum checkpoints across {n_seeds} seeds...")

    saved_steps = 0
    for step in all_steps:
        k_grid = None
        powers = []
        for d in seed_dirs:
            f = d / f"time_data_ps_step_{step}.txt"
            if not f.exists():
                candidates = list(d.glob(f"*_ps_step_{step}.txt"))
                candidates = [c for c in candidates if "inst" not in c.name]
                f = candidates[0] if candidates else None

            if f and f.exists():
                try:
                    data = np.loadtxt(f, skiprows=1)
                    if data.size > 0:
                        if k_grid is None:
                            k_grid = data[:, 0]
                        powers.append(data[:, 1])
                except Exception as e:
                    print(f"  Warning reading {f}: {e}")

        if powers and k_grid is not None:
            powers_arr = np.array(powers)
            mean_power = np.mean(powers_arr, axis=0)
            std_power = np.std(powers_arr, axis=0) / np.sqrt(len(powers)) if len(powers) > 1 else np.zeros_like(mean_power)

            out_file = f"{prefix_base}_ps_step_{step}.txt"
            with open(out_file, "w") as f_out:
                f_out.write("k power power_stderr\n")
                for k_val, p_val, err_val in zip(k_grid, mean_power, std_power):
                    f_out.write(f"{k_val:.16e} {p_val:.16e} {err_val:.16e}\n")
            saved_steps += 1

    print(f"Saved {saved_steps} ensemble-averaged spectrum files to {out_path}")

    # 2. Process Domain-Wall Roughness w^2(t)
    print("\nAveraging domain-wall roughness w^2(t)...")
    w2_data_list = []
    for d in seed_dirs:
        cand = list(d.glob("*transient_w2*.txt"))
        if cand:
            w2_file = str(cand[0])
            try:
                fsize = os.path.getsize(w2_file)
                if fsize < 5 * 1024 * 1024:
                    data = np.loadtxt(w2_file, skiprows=1)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    t_raw, w_raw = data[:, 1], data[:, 3]
                else:
                    est_lines = fsize / 65.0
                    stride = max(1, int(est_lines / 6000))
                    t_list, w_list = [], []
                    with open(w2_file, "r") as f:
                        f.readline()
                        for i, line in enumerate(f):
                            if i % stride == 0:
                                parts = line.split()
                                if len(parts) >= 4:
                                    try:
                                        t_list.append(float(parts[1]))
                                        w_list.append(float(parts[3]))
                                    except ValueError:
                                        pass
                    t_raw = np.array(t_list)
                    w_raw = np.array(w_list)

                # Filter valid points: t >= 400, positive roughness below extreme outliers
                m = (t_raw >= 400.0) & (t_raw < 1.5e7) & (w_raw > 0.0) & (w_raw < 1500.0)
                t_raw, w_raw = t_raw[m], w_raw[m]

                # Clean spikes
                if len(w_raw) > 5:
                    window = 5
                    pad = window // 2
                    padded = np.pad(np.log10(w_raw), pad, mode="edge")
                    med = np.array([np.median(padded[i:i+window]) for i in range(len(w_raw))])
                    good = np.abs(np.log10(w_raw) - med) < 0.25
                    t_raw, w_raw = t_raw[good], w_raw[good]

                if len(t_raw) > 0:
                    w2_data_list.append((t_raw, w_raw))
            except Exception as e:
                print(f"  Warning reading roughness in {d.name}: {e}")

    if w2_data_list:
        # Determine global time range
        min_t = min(d[0].min() for d in w2_data_list if len(d[0]) > 0)
        max_t = max(d[0].max() for d in w2_data_list if len(d[0]) > 0)

        # Logarithmic common time grid with 5000 points
        t_common = np.logspace(np.log10(min_t), np.log10(max_t), 5000)
        
        # Interpolate each seed onto grid where valid
        w2_matrix = []
        counts = np.zeros(len(t_common))
        sum_w2 = np.zeros(len(t_common))
        sq_sum_w2 = np.zeros(len(t_common))

        for t_s, w_s in w2_data_list:
            if len(t_s) > 1:
                # Mask where t_common is within this seed's range
                valid = (t_common >= t_s.min()) & (t_common <= t_s.max())
                w_interp = np.interp(t_common[valid], t_s, w_s)
                sum_w2[valid] += w_interp
                sq_sum_w2[valid] += w_interp ** 2
                counts[valid] += 1

        valid_times = counts > 0
        t_final = t_common[valid_times]
        c_final = counts[valid_times]
        mean_w2 = sum_w2[valid_times] / c_final
        var_w2 = np.maximum(0.0, (sq_sum_w2[valid_times] / c_final) - (mean_w2 ** 2))
        stderr_w2 = np.where(c_final > 1, np.sqrt(var_w2 / c_final), 0.0)

        out_w2_file = f"{prefix_base}_transient_w2.txt"
        with open(out_w2_file, "w") as f_out:
            f_out.write("step time mean_u var_u var_u_stderr\n")
            for t_val, w_val, err_val in zip(t_final, mean_w2, stderr_w2):
                step_est = int(np.round(t_val / 0.0196349540849362))
                f_out.write(f"{step_est} {t_val:.8e} 0.0 {w_val:.8e} {err_val:.8e}\n")
        print(f"Saved ensemble-averaged roughness to {out_w2_file}")

    # Copy or create parameters file
    param_files = list(seed_dirs[0].glob("parameters.txt"))
    if param_files:
        with open(param_files[0], "r") as f_in, open(out_path / "parameters.txt", "w") as f_out:
            f_out.write(f"# Ensemble Averaged over {n_seeds} seeds\n")
            f_out.write(f"# Seed directories: {', '.join(d.name for d in seed_dirs)}\n")
            f_out.write(f_in.read())
    else:
        with open(out_path / "parameters.txt", "w") as f_out:
            f_out.write(f"# Ensemble Averaged over {n_seeds} seeds\n")
            f_out.write(f"# Seed directories: {', '.join(d.name for d in seed_dirs)}\n")
            f_out.write("Simulation: time_evo NC Edwards-Wilkinson (Ensemble Averaged)\n")
            f_out.write("DT: 0.049087385212340519\n")
            f_out.write("SIZE: 16384\n")
            f_out.write("L: 8192.0\n")
            f_out.write("ALPHA: 0.27\n")
            f_out.write("H0: 5.0\n")
            f_out.write("HA: 3.0\n")
            f_out.write("OMEGA: 1.0\n")
            f_out.write("EPS: 0.0005\n")

    print(f"\nEnsemble averaging complete! Data saved to: {out_path}\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Ensemble-average power spectra and roughness for Figure 6.")
    parser.add_argument("ensemble_dir", help="Directory containing seed_* subdirectories")
    parser.add_argument("--out-dir", default=None, help="Destination directory (default: <ensemble_dir>/ensemble_averaged)")
    parser.add_argument("--plot", action="store_true", help="Plot Figure 6 variants immediately after averaging")
    args = parser.parse_args()

    out_path = average_ensemble(args.ensemble_dir, args.out_dir)

    if args.plot:
        plot_script = Path(__file__).parent / "plot_figure6_nc_ew.py"
        if plot_script.exists():
            import subprocess
            cmd = [sys.executable, str(plot_script), str(out_path), "--layout", "all"]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

