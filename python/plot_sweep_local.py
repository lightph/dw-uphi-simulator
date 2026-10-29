#!/usr/bin/env python3
"""
plot_sweep_local.py - Local and CLI-optimized 1D/2D parameter sweep plotter.

Recreates exactly the PRB publication-ready plots of plot_sweep.py with:
- Zero hard dependencies beyond numpy and matplotlib (no pandas required).
- Auto-discovery of sweep directories and sensible default outputs.
- Metric filtering (-m/--metrics) to quickly plot only what you need.
- Threshold overriding (--thresh-lock, --thresh-rug, etc.) for fast physical tuning.
- Interactive figure preview (--show / -s).
- Fast data caching (.npz) for sub-second iteration on large sweeps.
- 1D slicing (--slice-ha / --slice-h0) from 2D parameter sweeps.
- Parameter zooming (--h0-min, --h0-max, --ha-min, --ha-max).
- Configurable Floquet overlay support without hardcoded remote paths.
"""

import os
import sys
import glob
import time
import argparse
import tempfile
from pathlib import Path

# Ensure writable matplotlib config/cache directory to prevent warnings in sandboxes or restricted environments
if 'MPLCONFIGDIR' not in os.environ:
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    cache_dir = os.path.join(tempfile.gettempdir(), f"matplotlib_{uid}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = cache_dir

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.patheffects as pe

# ==============================================================================
# PRB PUBLICATION-READY PLOT SETTINGS (Preserved identically from plot_sweep.py)
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.fontsize': 12,
    'legend.frameon': False,
    'figure.figsize': (7, 5),  # Professional aspect ratio
})

# ==============================================================================
# DATA-DRIVEN THRESHOLD DISCOVERY ALGORITHMS
# ==============================================================================
def get_first_major_gap_threshold(arr, min_gap_orders=2.0, default_val=1e-8):
    """
    Scans from the smallest valid values upwards in log space.
    Returns the midpoint of the FIRST gap that exceeds min_gap_orders.
    Separates the numerical noise floor from macroscopic physics.
    """
    arr = np.asarray(arr)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if len(valid) < 2:
        return default_val
        
    log_vals = np.log10(np.sort(valid))
    gaps = np.diff(log_vals)
    
    for i, gap in enumerate(gaps):
        if gap > min_gap_orders:
            return float(10 ** (log_vals[i] + gap / 2.0))
            
    return default_val

def get_largest_gap_threshold(arr, max_search_val=1e-1, default_val=1e-8, gap_fraction=0.5):
    """
    Finds the largest logarithmic gap in the data below a maximum physical threshold.
    gap_fraction (0.0 to 1.0): 
        0.1 = Very Strict (threshold placed just above the noise floor)
        0.5 = Balanced (geometric center of the gap)
        0.9 = Very Loose (threshold placed just below macroscopic physics)
    """
    arr = np.asarray(arr)
    valid = arr[np.isfinite(arr) & (arr > 0) & (arr < max_search_val)]
    if len(valid) < 2:
        return default_val
        
    log_vals = np.log10(np.sort(valid))
    gaps = np.diff(log_vals)
    
    if len(gaps) == 0:
        return default_val
        
    max_gap_idx = np.argmax(gaps)
    return float(10 ** (log_vals[max_gap_idx] + gaps[max_gap_idx] * gap_fraction))

# ==============================================================================
# METRICS CONFIGURATION (Preserved identically from plot_sweep.py)
# ==============================================================================
METRIC_SPECS = {
    'phase_diagram': {
        'cmap': ListedColormap(['green', 'blue', 'cyan', 'red', 'darkred']), 
        'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 4.5,
        'ticks': [0, 1, 2, 3, 4], 
        'labels': ['PF', 'SF', 'SP', 'NF', 'NR'],
        'display_name': 'Phase',
        'desc': '5-state phase classification (Pinned Flat, Locked Flat, Locked Periodic, Unlocked Flat, Unlocked Rough)'
    },
    'binarized_rugosity': {
        'cmap': ListedColormap(['#4575b4', '#d73027']), 
        'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 1.5,
        'ticks': [0, 1], 'labels': ['Flat (0)', 'Rough (1)'],
        'display_name': 'Binarized Rugosity',
        'desc': 'Binary indicator of interface roughness'
    },
    'phase_locked': {
        'cmap': ListedColormap(['#d73027', '#1a9850']), 
        'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 1.5,
        'ticks': [0, 1], 'labels': ['False (0)', 'True (1)'],
        'display_name': 'Phase Locked',
        'desc': 'Binary indicator of stroboscopic phase synchronization'
    },
    'structure': {
        'cmap': ListedColormap(['#4575b4', '#ffffbf', '#d73027']), 
        'log': False, 'cat': True, 'vmin': -0.5, 'vmax': 2.5,
        'ticks': [0, 1, 2], 'labels': ['Flat (0)', 'Periodic (1)', 'Corrugated (2)'],
        'display_name': 'Spatial Structure',
        'desc': 'Spatial structure state (Flat, Periodic, Corrugated)'
    },
    'mean_var_strobo': {
        'cmap': 'magma', 'log': True, 'cat': False, 
        'display_name': r'$\langle \sigma^2 \rangle_x$',
        'desc': 'Mean stroboscopic variance'
    },
    'spectral_entropy': {
        'cmap': 'plasma', 'log': False, 'cat': False, 
        'display_name': 'Spectral Entropy',
        'desc': 'Spatial spectral entropy of the domain wall'
    },
    'rugosity': {
        'cmap': 'inferno', 'log': True, 'cat': False, 
        'display_name': r'$\langle w^2 \rangle_t$',
        'desc': 'Interface roughness / rugosity'
    },
    'v_u': {
        'cmap': 'coolwarm', 'log': False, 'cat': False, 
        'display_name': r'$v_u$',
        'desc': 'Mean domain wall translation velocity'
    },
    'v_phi': {
        'cmap': 'coolwarm', 'log': False, 'cat': False, 
        'display_name': r'$v_\phi$',
        'desc': 'Mean internal angle phase velocity'
    },
    'max_q': {
        'cmap': 'viridis', 'log': False, 'cat': False, 
        'display_name': r'Max $q$',
        'desc': 'Dominant spatial wavenumber'
    },
    'wall_q': {
        'cmap': 'viridis', 'log': False, 'cat': False, 
        'display_name': r'Wall $q$',
        'desc': 'Wall spatial wavenumber (for periodic structures)'
    }
}

# ==============================================================================
# FLOQUET DEFAULTS
# ==============================================================================
DEFAULT_FLOQUET_NUM_H = 1000
DEFAULT_FLOQUET_NUM_HAC = 1000
DEFAULT_FLOQUET_H_MAX = 10.0
DEFAULT_FLOQUET_HAC_MAX = 20.0
SYNC_COLOR = 'white'
SYNC_STYLE = 'solid'
SYNC_WIDTH = 1.0
STAB_COLOR = 'black'
STAB_STYLE = 'dashed'
STAB_WIDTH = 1.0

# ==============================================================================
# DATA LOADING & CACHING (Fast pure-numpy loader)
# ==============================================================================
def find_latest_sweep_dir():
    """Locate the most recently modified sweep directory in output/ or current dir."""
    search_dirs = [
        os.path.join(os.getcwd(), 'output'),
        os.path.join(os.getcwd(), '..', 'output'),
        os.getcwd()
    ]
    candidates = []
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for entry in os.scandir(sdir):
                if entry.is_dir() and 'sweep' in entry.name:
                    candidates.append((entry.stat().st_mtime, entry.path))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None

def load_sweep_data(input_path, pattern="results_h0_*.txt", use_cache=True, quiet=False):
    """
    Loads sweep data from directory, file, or glob pattern using pure numpy.
    Supports caching to .sweep_cache.npz for sub-second subsequent reloads.
    """
    path_obj = Path(input_path)
    
    # Check if input is a single file
    if path_obj.is_file():
        if not quiet:
            print(f"Loading single results file: {input_path}")
        with open(input_path, 'r') as f:
            header_line = f.readline().strip()
            cols = header_line.split()
        arr = np.loadtxt(input_path, skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        data = {col: arr[:, i] for i, col in enumerate(cols)}
        data['h0'] = np.round(data['h0'], 5)
        data['ha'] = np.round(data['ha'], 5)
        return data, [str(path_obj)]

    # Input is a directory
    if not path_obj.is_dir():
        # Maybe input_path is a glob pattern directly
        files = sorted(glob.glob(input_path))
        if not files:
            raise FileNotFoundError(f"Input path '{input_path}' is neither a file, directory, nor valid glob.")
        directory = os.path.dirname(files[0]) or '.'
    else:
        directory = str(path_obj)
        # Search for pattern
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        if not files:
            # Fall back to sweep_data*.txt or any results_*.txt
            files = sorted(glob.glob(os.path.join(directory, "sweep_data*.txt")))
        if not files:
            # Search for any .txt file that has h0 and ha in the header
            candidate_txts = sorted(glob.glob(os.path.join(directory, "*.txt")))
            files = []
            for txt in candidate_txts:
                if os.path.basename(txt) in ("run_parameters.txt", "parameters.txt"):
                    continue
                try:
                    with open(txt, 'r') as f:
                        line = f.readline().lower()
                        if 'h0' in line and 'ha' in line:
                            files.append(txt)
                except Exception:
                    pass

    if not files:
        raise FileNotFoundError(f"No sweep data files found in '{directory}' with pattern '{pattern}'.")

    cache_file = os.path.join(directory, ".sweep_cache.npz")
    
    # Check cache validity
    if use_cache and os.path.exists(cache_file):
        try:
            cache_mtime = os.path.getmtime(cache_file)
            newest_data_mtime = max(os.path.getmtime(f) for f in files)
            if cache_mtime >= newest_data_mtime:
                if not quiet:
                    print(f"Loading from cache: {cache_file}")
                with np.load(cache_file) as npz:
                    data = {k: npz[k] for k in npz.files}
                return data, files
        except Exception:
            pass  # Fall back to raw reading if cache corrupted

    if not quiet:
        print(f"Reading {len(files)} result files from {directory} ...")

    # Read header from the first non-empty file
    cols = None
    for f in files:
        if os.path.getsize(f) > 0:
            with open(f, 'r') as fp:
                cols = fp.readline().strip().split()
            break

    if cols is None:
        raise ValueError("All found sweep files are empty.")

    arrays = []
    for f in files:
        if os.path.getsize(f) > 0:
            try:
                arr = np.loadtxt(f, skiprows=1)
                if arr.ndim == 1 and arr.size > 0:
                    arr = arr.reshape(1, -1)
                if arr.size > 0:
                    arrays.append(arr)
            except Exception as e:
                if not quiet:
                    print(f"Warning: Skipping unreadable file {f}: {e}")

    if not arrays:
        raise ValueError("Could not extract any valid data rows from files.")

    all_data = np.vstack(arrays)
    data = {col: all_data[:, i] for i, col in enumerate(cols)}
    data['h0'] = np.round(data['h0'], 5)
    data['ha'] = np.round(data['ha'], 5)

    # Save cache if possible and requested
    if use_cache:
        try:
            np.savez_compressed(cache_file, **data)
            if not quiet:
                print(f"Cached dataset to {cache_file} for fast re-runs.")
        except Exception:
            pass

    return data, files

# ==============================================================================
# FLOQUET BINARY LOADER
# ==============================================================================
def load_floquet_binary(bin_path, num_h=DEFAULT_FLOQUET_NUM_H, num_hac=DEFAULT_FLOQUET_NUM_HAC,
                        h_max=DEFAULT_FLOQUET_H_MAX, hac_max=DEFAULT_FLOQUET_HAC_MAX, verbose=False):
    """Loads Floquet stability binary file if it exists."""
    if not bin_path or not os.path.exists(bin_path):
        return False, None, None, None, None

    dtype = np.dtype([
        ('h', '<f8'),
        ('h_ac', '<f8'),
        ('omega', '<f8'),
        ('x', '<f8'),
        ('max_k_mu', '<f8'),
        ('max_mu', '<f8'),
        ('max_k_sigma', '<f8'),
        ('max_sigma', '<f8'),
        ('converged', '<?')
    ], align=True)

    try:
        raw_data = np.fromfile(bin_path, dtype=dtype)
        expected_len = num_h * num_hac
        if len(raw_data) == expected_len:
            converged_grid = raw_data['converged'].reshape((num_h, num_hac))
            mu_grid = raw_data['max_mu'].reshape((num_h, num_hac))
            mu_grid = np.where(converged_grid, mu_grid, np.nan)

            h_vals = np.linspace(0, h_max, num_h)
            hac_vals = np.linspace(0, hac_max, num_hac)
            H_GRID, HAC_GRID = np.meshgrid(h_vals, hac_vals, indexing='ij')

            if verbose:
                print(f"Loaded Floquet stability data from {bin_path} ({num_h}x{num_hac}).")
            return True, H_GRID, HAC_GRID, converged_grid, mu_grid
        else:
            if verbose:
                print(f"Warning: Floquet file size mismatch ({len(raw_data)} vs {expected_len}). Skipping overlay.")
            return False, None, None, None, None
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read Floquet file: {e}")
        return False, None, None, None, None

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Local & CLI-friendly Parameter Sweep Plotter (PRB Style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Run on latest sweep found in output/ (saves to <sweep_dir>/figures):
  python python/plot_sweep_local.py

  # 2. Run on a specific sweep directory:
  python python/plot_sweep_local.py output/sweep_full_973971

  # 3. Plot only the 5-phase diagram and rugosity:
  python python/plot_sweep_local.py output/sweep_full_973971 -m phase_diagram rugosity

  # 4. Preview phase diagram interactively on-screen without saving:
  python python/plot_sweep_local.py output/sweep_full_973971 -m phase_diagram --show --no-save

  # 5. Extract and plot 1D slice at ha = 3.0:
  python python/plot_sweep_local.py output/sweep_full_973971 --slice-ha 3.0

  # 6. Override classification thresholds:
  python python/plot_sweep_local.py output/sweep_full_973971 --thresh-lock 1e-25 --thresh-rug 1e-15

  # 7. Zoom into parameter range:
  python python/plot_sweep_local.py output/sweep_full_973971 --h0-min 0 --h0-max 5 --ha-min 0 --ha-max 10
        """
    )

    # Input and Output options
    io_group = parser.add_argument_group("Input & Output")
    io_group.add_argument("input_path", nargs="?", default=None,
                          help="Path to results directory, single .txt file, or glob pattern. (Default: auto-detect latest sweep in output/)")
    io_group.add_argument("-o", "--output-dir", default=None,
                          help="Directory to save plots. (Default: <input_dir>/figures)")
    io_group.add_argument("--pattern", default="results_h0_*.txt",
                          help="File glob pattern inside directory (default: 'results_h0_*.txt')")
    io_group.add_argument("--no-save", action="store_true",
                          help="Do not save plots to disk (ideal with --show)")
    io_group.add_argument("-s", "--show", action="store_true",
                          help="Display plots interactively in a GUI window")
    io_group.add_argument("--dpi", type=int, default=300,
                          help="DPI resolution for saved figures (default: 300)")
    io_group.add_argument("--ext", "--format", default="png", choices=["png", "pdf", "svg"],
                          help="Output image format (default: png)")
    io_group.add_argument("--prefix", default="",
                          help="Prefix prepended to output plot filenames")

    # Metrics selection
    metric_group = parser.add_argument_group("Metrics Selection")
    metric_group.add_argument("-m", "--metrics", nargs="+", default=None,
                              help="Specific metrics to plot (e.g. -m phase_diagram rugosity). Default: plot all available.")
    metric_group.add_argument("--list-metrics", action="store_true",
                              help="List all supported metrics and descriptions, then exit.")

    # Data filtering and slicing
    slice_group = parser.add_argument_group("Slicing & Filtering")
    slice_group.add_argument("--slice-ha", "--ha", type=float, default=None,
                             help="Extract a 1D slice at this ha value and plot vs h0")
    slice_group.add_argument("--slice-h0", "--h0", type=float, default=None,
                             help="Extract a 1D slice at this h0 value and plot vs ha")
    slice_group.add_argument("--h0-min", type=float, default=None, help="Filter min h0")
    slice_group.add_argument("--h0-max", type=float, default=None, help="Filter max h0")
    slice_group.add_argument("--ha-min", type=float, default=None, help="Filter min ha")
    slice_group.add_argument("--ha-max", type=float, default=None, help="Filter max ha")
    slice_group.add_argument("--stride", type=int, default=1, help="Downsample stride for quick preview")

    # Threshold tuning
    thresh_group = parser.add_argument_group("Threshold Tuning")
    thresh_group.add_argument("--thresh-lock", type=float, default=None,
                              help="Manual threshold for mean_var_strobo (phase locking)")
    thresh_group.add_argument("--thresh-rug", type=float, default=None,
                              help="Manual threshold for rugosity (interface flatness)")
    thresh_group.add_argument("--thresh-vphi", type=float, default=None,
                              help="Manual threshold for |v_phi| (domain wall pinning)")
    thresh_group.add_argument("--thresh-method", choices=["largest_gap", "first_gap"], default="largest_gap",
                              help="Gap detection algorithm (default: largest_gap)")
    thresh_group.add_argument("--lock-frac", type=float, default=0.1,
                              help="Gap fraction for THRESH_LOCK (default: 0.1)")
    thresh_group.add_argument("--rug-frac", type=float, default=0.2,
                              help="Gap fraction for THRESH_RUG (default: 0.2)")
    thresh_group.add_argument("--vphi-frac", type=float, default=0.2,
                              help="Gap fraction for THRESH_VPHI (default: 0.2)")

    # Floquet overlays
    floquet_group = parser.add_argument_group("Floquet Overlays")
    floquet_group.add_argument("--floquet-bin", default=None,
                               help="Path to Floquet stability binary file (e.g. floquet_transient_omega_1.000_L_1024.bin)")
    floquet_group.add_argument("--no-floquet", action="store_true",
                               help="Disable Floquet boundary contours")
    floquet_group.add_argument("--floquet-num-h", type=int, default=DEFAULT_FLOQUET_NUM_H)
    floquet_group.add_argument("--floquet-num-hac", type=int, default=DEFAULT_FLOQUET_NUM_HAC)
    floquet_group.add_argument("--floquet-h-max", type=float, default=DEFAULT_FLOQUET_H_MAX)
    floquet_group.add_argument("--floquet-hac-max", type=float, default=DEFAULT_FLOQUET_HAC_MAX)

    # Cache & Verbosity
    misc_group = parser.add_argument_group("Performance & Verbosity")
    misc_group.add_argument("--no-cache", action="store_true",
                            help="Disable reading/writing .sweep_cache.npz")
    misc_group.add_argument("-q", "--quiet", action="store_true",
                            help="Quiet mode: minimize stdout messages")
    misc_group.add_argument("-v", "--verbose", action="store_true",
                            help="Verbose debug logging")

    args = parser.parse_args()

    # List metrics helper
    if args.list_metrics:
        print("\nSupported Metrics:")
        print("----------------------------------------------------------------------")
        for m, spec in METRIC_SPECS.items():
            print(f"  {m:20s} : {spec['desc']}")
        print("----------------------------------------------------------------------\n")
        return

    # Auto-detect input path if omitted
    if args.input_path is None:
        latest = find_latest_sweep_dir()
        if latest is None:
            print("Error: No input path specified and no sweep directory found in output/.")
            print("Usage: python python/plot_sweep_local.py [input_dir_or_file]")
            sys.exit(1)
        args.input_path = latest
        if not args.quiet:
            print(f"Auto-selected latest sweep directory: {args.input_path}")

    # Determine output directory
    if args.output_dir is None:
        if os.path.isdir(args.input_path):
            args.output_dir = os.path.join(args.input_path, "figures")
        else:
            args.output_dir = os.path.join(os.path.dirname(args.input_path) or ".", "figures")

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # Load data
    use_cache = not args.no_cache
    data, matched_files = load_sweep_data(
        args.input_path, pattern=args.pattern, use_cache=use_cache, quiet=args.quiet
    )

    n_raw = len(data['h0'])

    # Apply data filtering / cropping
    mask = np.ones(n_raw, dtype=bool)
    if args.h0_min is not None:
        mask &= (data['h0'] >= args.h0_min)
    if args.h0_max is not None:
        mask &= (data['h0'] <= args.h0_max)
    if args.ha_min is not None:
        mask &= (data['ha'] >= args.ha_min)
    if args.ha_max is not None:
        mask &= (data['ha'] <= args.ha_max)

    # 1D slice filtering
    slice_label = None
    if args.slice_ha is not None:
        unique_has = np.unique(data['ha'])
        closest_ha = unique_has[np.argmin(np.abs(unique_has - args.slice_ha))]
        mask &= np.isclose(data['ha'], closest_ha, atol=1e-4)
        slice_label = f"ha_{closest_ha:.4f}"
        if not args.quiet:
            print(f"Slicing 2D data at ha = {closest_ha:.4f} (target: {args.slice_ha})")

    if args.slice_h0 is not None:
        unique_h0s = np.unique(data['h0'])
        closest_h0 = unique_h0s[np.argmin(np.abs(unique_h0s - args.slice_h0))]
        mask &= np.isclose(data['h0'], closest_h0, atol=1e-4)
        slice_label = f"h0_{closest_h0:.4f}"
        if not args.quiet:
            print(f"Slicing 2D data at h0 = {closest_h0:.4f} (target: {args.slice_h0})")

    if args.stride > 1:
        indices = np.where(mask)[0]
        sub_indices = indices[::args.stride]
        mask = np.zeros(n_raw, dtype=bool)
        mask[sub_indices] = True

    # Filter data dictionary
    filtered_data = {k: v[mask] for k, v in data.items()}
    n_points = len(filtered_data['h0'])
    if n_points == 0:
        print("Error: No data points left after applying filters/slices.")
        sys.exit(1)

    # ==============================================================================
    # COMPUTE DATA-DRIVEN THRESHOLDS & DERIVED COLUMNS
    # ==============================================================================
    thresh_fn = get_largest_gap_threshold if args.thresh_method == "largest_gap" else get_first_major_gap_threshold

    if args.thresh_lock is not None:
        THRESH_LOCK = args.thresh_lock
        lock_src = "override"
    else:
        THRESH_LOCK = thresh_fn(filtered_data['mean_var_strobo'], max_search_val=1e-1, default_val=1e-10, gap_fraction=args.lock_frac)
        lock_src = f"gap fraction={args.lock_frac}"

    if args.thresh_rug is not None:
        THRESH_RUG = args.thresh_rug
        rug_src = "override"
    else:
        THRESH_RUG = thresh_fn(filtered_data['rugosity'], max_search_val=1e-1, default_val=1e-7, gap_fraction=args.rug_frac)
        rug_src = f"gap fraction={args.rug_frac}"

    if args.thresh_vphi is not None:
        THRESH_VPHI = args.thresh_vphi
        vphi_src = "override"
    else:
        THRESH_VPHI = thresh_fn(np.abs(filtered_data['v_phi']), max_search_val=1e-1, default_val=1e-6, gap_fraction=args.vphi_frac)
        vphi_src = f"gap fraction={args.vphi_frac}"

    is_locked = filtered_data['mean_var_strobo'] < THRESH_LOCK
    is_flat = filtered_data['rugosity'] <= THRESH_RUG
    is_pinned = is_locked & (np.abs(filtered_data['v_phi']) < THRESH_VPHI)

    filtered_data['phase_locked'] = is_locked.astype(int)
    filtered_data['structure'] = np.where(is_flat, 0, np.where(is_locked, 1, 2))
    filtered_data['binarized_rugosity'] = (~is_flat).astype(int)

    conditions = [
        is_pinned,                             # 0: Pinned (P)
        is_locked & is_flat & ~is_pinned,      # 1: Phase Locked Flat (SF)
        is_locked & ~is_flat,                  # 2: Phase Locked Periodic (SP)
        ~is_locked & is_flat,                  # 3: Unlocked Flat (NF)
        ~is_locked & ~is_flat                  # 4: Unlocked Rough (NR)
    ]
    filtered_data['phase_diagram'] = np.select(conditions, [0, 1, 2, 3, 4], default=-1)
    if 'max_q' in filtered_data:
        filtered_data['wall_q'] = np.where(filtered_data['structure'] == 1, filtered_data['max_q'], 0.0)

    # Parameter grid statistics
    unique_ha = np.unique(filtered_data['ha'])
    unique_h0 = np.unique(filtered_data['h0'])
    is_1d = (len(unique_ha) == 1) or (len(unique_h0) == 1)

    if not args.quiet:
        print("\n" + "="*70)
        print("  SWEEP PLOTTER (Local CLI Edition)")
        print("="*70)
        print(f"Data points: {n_points:,} (from {len(matched_files)} files)")
        print(f"Grid range : h0 in [{unique_h0.min():.4f}, {unique_h0.max():.4f}] ({len(unique_h0)} pts), "
              f"ha in [{unique_ha.min():.4f}, {unique_ha.max():.4f}] ({len(unique_ha)} pts)")
        print(f"Plot mode  : {'1D Line Curve' if is_1d else '2D Parameter Heatmap'}")
        print("\nThresholds:")
        print(f"  - THRESH_LOCK (mean_var_strobo) : {THRESH_LOCK:.3e} [{lock_src}]")
        print(f"  - THRESH_RUG  (rugosity)        : {THRESH_RUG:.3e} [{rug_src}]")
        print(f"  - THRESH_VPHI (|v_phi|)         : {THRESH_VPHI:.3e} [{vphi_src}]")

        # Phase counts
        phase_names = {0: 'P  (Pinned)', 1: 'SF (Locked Flat)', 2: 'SP (Locked Periodic)',
                       3: 'NF (Unlocked Flat)', 4: 'NR (Unlocked Rough)'}
        u_p, counts = np.unique(filtered_data['phase_diagram'], return_counts=True)
        print("\nPhase Distribution:")
        for p_val, cnt in zip(u_p, counts):
            name = phase_names.get(p_val, f'Unknown ({p_val})')
            print(f"  - {name:24s}: {cnt:7,d} ({cnt / n_points * 100:5.2f}%)")
        print("="*70 + "\n")

    # ==============================================================================
    # FLOQUET STABILITY OVERLAYS
    # ==============================================================================
    floquet_data_loaded = False
    H_GRID = HAC_GRID = converged_grid = mu_grid = None

    if not args.no_floquet:
        floquet_paths = []
        if args.floquet_bin:
            floquet_paths.append(args.floquet_bin)
        else:
            # Common local and relative project paths
            floquet_paths.extend([
                os.path.join(os.getcwd(), "output", "floquet_transient_omega_1.000_L_1024.bin"),
                os.path.join(os.getcwd(), "..", "output", "floquet_transient_omega_1.000_L_1024.bin"),
                os.path.join(os.getcwd(), "stability", "floquet_transient_omega_1.000_L_1024.bin"),
                os.path.join(os.getcwd(), "..", "stability", "floquet_transient_omega_1.000_L_1024.bin"),
                "/home/luz.perez/inv1/stability/floquet_transient_omega_1.000_L_1024.bin"
            ])

        for fpath in floquet_paths:
            if os.path.exists(fpath):
                floquet_data_loaded, H_GRID, HAC_GRID, converged_grid, mu_grid = load_floquet_binary(
                    fpath,
                    num_h=args.floquet_num_h,
                    num_hac=args.floquet_num_hac,
                    h_max=args.floquet_h_max,
                    hac_max=args.floquet_hac_max,
                    verbose=(not args.quiet)
                )
                if floquet_data_loaded:
                    break

    # ==============================================================================
    # SELECT METRICS TO PLOT
    # ==============================================================================
    if args.metrics:
        # User specified specific metrics
        selected_metrics = []
        for m in args.metrics:
            if m in METRIC_SPECS or m in filtered_data:
                selected_metrics.append(m)
            else:
                print(f"Warning: Metric '{m}' not recognized. Skipping.")
    else:
        # Default: all standard metrics present in filtered_data
        selected_metrics = [m for m in METRIC_SPECS.keys() if m in filtered_data]

    if not selected_metrics:
        print("Error: No valid metrics selected for plotting.")
        sys.exit(1)

    label_h0 = r'$h_0$'
    label_ha = r'$h_\mathrm{a}$'

    # Precompute 2D meshgrid indices if in 2D mode
    if not is_1d:
        u_h0, h0_idx = np.unique(filtered_data['h0'], return_inverse=True)
        u_ha, ha_idx = np.unique(filtered_data['ha'], return_inverse=True)
        X, Y = np.meshgrid(u_h0, u_ha)
        n_ha, n_h0 = len(u_ha), len(u_h0)
        flat_idx = ha_idx * n_h0 + h0_idx

    total_metrics = len(selected_metrics)
    saved_files = []

    for idx, metric in enumerate(selected_metrics, 1):
        props = METRIC_SPECS.get(metric, {
            'cmap': 'viridis', 'log': False, 'cat': False, 'display_name': metric
        })

        plt.figure()  # Uses PRB size configured in rcParams

        if is_1d:
            # 1D line plot
            x_col = 'h0' if len(unique_ha) == 1 else 'ha'
            x_label = label_h0 if x_col == 'h0' else label_ha
            
            x_vals = filtered_data[x_col]
            y_vals = filtered_data[metric]
            valid = np.isfinite(y_vals)
            
            if not np.any(valid):
                plt.close()
                continue
                
            x_valid = x_vals[valid]
            y_valid = y_vals[valid]
            
            # Sort by x
            sort_order = np.argsort(x_valid)
            x_sorted = x_valid[sort_order]
            y_sorted = y_valid[sort_order]
            
            # Handle potential duplicate x values cleanly by averaging
            u_x, x_inv = np.unique(x_sorted, return_inverse=True)
            if len(u_x) < len(x_sorted):
                sums = np.bincount(x_inv, weights=y_sorted)
                counts = np.bincount(x_inv)
                x_plot = u_x
                y_plot = sums / counts
            else:
                x_plot = x_sorted
                y_plot = y_sorted

            if props['log']:
                plt.semilogy(x_plot, y_plot + 1e-15, marker='o', linestyle='-', color='black', markersize=4)
            else:
                plt.plot(x_plot, y_plot, marker='o', linestyle='-', color='black', markersize=4)

            plt.xlabel(x_label)
            plt.ylabel(props.get('display_name', metric))

            if props['cat']:
                plt.yticks(props['ticks'], props['labels'])

        else:
            # 2D heatmap
            val = filtered_data[metric]
            val_valid = np.nan_to_num(val, nan=0.0)
            valid_mask = np.isfinite(val)

            # Robust pivot table using vectorized bincount (supports duplicate coordinates and missing cells)
            sums = np.bincount(flat_idx[valid_mask], weights=val_valid[valid_mask], minlength=n_ha * n_h0)
            counts = np.bincount(flat_idx[valid_mask], minlength=n_ha * n_h0)
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_vals = np.where(counts > 0, sums / counts, np.nan)
            Z = grid_vals.reshape((n_ha, n_h0))
            Z_ma = np.ma.masked_invalid(Z)

            if Z_ma.count() == 0:
                plt.close()
                continue

            kwargs = {'shading': 'nearest', 'cmap': props['cmap']}

            if props['cat']:
                kwargs.update({'vmin': props['vmin'], 'vmax': props['vmax']})
                mesh = plt.pcolormesh(X, Y, Z_ma, **kwargs)

                if metric == 'phase_diagram':
                    colors = props['cmap'].colors
                    patches = [mpatches.Patch(color=colors[i], label=props['labels'][i]) for i in range(len(colors))]
                    plt.legend(handles=patches, loc='best', framealpha=0.9, edgecolor='black')
                else:
                    cbar = plt.colorbar(mesh, ticks=props['ticks'])
                    cbar.set_ticklabels(props['labels'])
                    cbar.set_label(props.get('display_name', metric))
            else:
                if props['log']:
                    Z_plot = np.clip(Z_ma, a_min=1e-15, a_max=None)
                    kwargs['norm'] = LogNorm(vmin=Z_plot.min(), vmax=Z_plot.max())
                    mesh = plt.pcolormesh(X, Y, Z_plot, **kwargs)
                else:
                    mesh = plt.pcolormesh(X, Y, Z_ma, **kwargs)

                cbar = plt.colorbar(mesh)
                cbar.set_label(props.get('display_name', metric))

            plt.xlabel(label_h0)
            plt.ylabel(label_ha)

            # Draw Floquet overlays if loaded
            if floquet_data_loaded:
                xlims = plt.xlim()
                ylims = plt.ylim()

                cs_sync = plt.contour(H_GRID, HAC_GRID, converged_grid, levels=[0.5],
                                      colors=[SYNC_COLOR], linestyles=[SYNC_STYLE], 
                                      linewidths=[SYNC_WIDTH], alpha=0.95)
                if hasattr(cs_sync, 'set_path_effects'):
                    cs_sync.set_path_effects([pe.Stroke(linewidth=SYNC_WIDTH + 0.6, foreground='#000000'), pe.Normal()])
                
                if np.nanmin(mu_grid) <= 1.0 <= np.nanmax(mu_grid):
                    cs_stab = plt.contour(H_GRID, HAC_GRID, mu_grid, levels=[1.0001],
                                          colors=[STAB_COLOR], linestyles=[STAB_STYLE], 
                                          linewidths=[STAB_WIDTH], alpha=0.95)
                    if hasattr(cs_stab, 'set_path_effects'):
                        cs_stab.set_path_effects([pe.Stroke(linewidth=STAB_WIDTH + 0.6, foreground='#ffffff'), pe.Normal()])

                plt.xlim(xlims)
                plt.ylim(ylims)

        plt.tight_layout()

        # Save plot to disk
        out_filename = None
        if not args.no_save:
            name_parts = []
            if args.prefix:
                name_parts.append(args.prefix)
            if slice_label:
                name_parts.append(slice_label)
            name_parts.append(metric)
            base_name = "_".join(name_parts) + f".{args.ext}"
            out_filename = os.path.join(args.output_dir, base_name)
            plt.savefig(out_filename, dpi=args.dpi, bbox_inches='tight')
            saved_files.append(out_filename)

        if not args.quiet:
            save_msg = f"-> {out_filename}" if out_filename else "(screen only)"
            print(f"  [{idx}/{total_metrics}] {metric} {save_msg}")

        # Interactive display
        if args.show:
            plt.show()

        plt.close()

    elapsed = time.time() - t0
    if not args.quiet:
        print(f"\nDone! Rendered {len(selected_metrics)} plot(s) in {elapsed:.2f}s.")
        if saved_files:
            print(f"Output directory: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()

