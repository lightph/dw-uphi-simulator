#!/usr/bin/env python3
r"""
plot_figure8_sc_scaling.py - Reproduces PRB Figure 8:
Finite-size scaling analysis of the Synchronized Periodic (SP) phase.

Shows:
  (a) Domain wall steady-state roughness \langle w^2 \rangle_t vs system length L/L0.
      Saturates to a constant in the large-L thermodynamic limit; flat for L/L0 <= 64.
  (b) Dominant wavenumber \kappa_\mathrm{peak} vs system length L/L0.
      Follows discrete fundamental mode 2\pi/(L/L0) for small systems, and converges
      to the maximum Floquet growth mode \kappa^* \approx 0.833 for large systems.
"""

import sys
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 8 (SP Phase Finite-Size Scaling).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "output" / "scaling_data"),
                        help="Data directory containing multi-batch scaling folders")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"), help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "python" / "plot_figure8_sc_scaling.py"),
        "--data-dir", str(args.data_dir),
        "--out-dir", str(args.out_dir)
    ]
    if args.export_paper:
        cmd.append("--export-paper")

    print(f"Running Figure 8 generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

