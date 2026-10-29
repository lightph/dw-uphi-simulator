#!/usr/bin/env python3
r"""
plot_figure6_nc_ew.py - Reproduces PRB Figure 6:
Universal 1D Edwards-Wilkinson scaling collapse of the Non-synchronized Rough (NR) phase.

Shows:
  - Structure factor power-law decay S_\kappa \propto \kappa^{-2}
  - Surface roughness growth w^2 \propto t^{1/2}
  - Universal master curve collapse: S_\kappa t^{-1} vs \kappa t^{1/2} (\zeta = 1/2, z = 2)
"""

import sys
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 6 (Edwards-Wilkinson Scaling Collapse).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "output" / "reference_nc_ew_data"),
                        help="Data directory containing EW time evolution files")
    parser.add_argument("--layout", default="inset", choices=["inset", "stacked", "side_by_side", "compensated", "all"],
                        help="Panel layout (default: inset)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"), help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "python" / "plot_figure6_nc_ew.py"),
        str(args.data_dir),
        "--layout", args.layout,
        "--outdir", str(args.out_dir)
    ]

    print(f"Running Figure 6 generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if args.export_paper:
        import shutil
        src_pdf = Path(args.out_dir) / "NC_EW_inset.pdf"
        dst_pdf = REPO_ROOT / "paper" / "figuras" / "NC_EW_inset.pdf"
        if src_pdf.exists():
            shutil.copyfile(src_pdf, dst_pdf)
            print(f"Exported to paper figures: {dst_pdf}")


if __name__ == '__main__':
    main()

