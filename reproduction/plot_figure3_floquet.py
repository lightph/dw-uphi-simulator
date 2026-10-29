#!/usr/bin/env python3
r"""
plot_figure3_floquet.py - Reproduces PRB Figure 3:
Linear Floquet stability diagrams for ha = 2 and ha = 3.
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "floquet" / "python"))

from plot_figure3 import plot_figure3


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 3 (Floquet Stability Diagrams).")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation without cache")
    parser.add_argument("--h0-pts", type=int, default=300, help="Number of points in h0 grid")
    parser.add_argument("--kappa-pts", type=int, default=200, help="Number of points in kappa grid")
    args = parser.parse_args()

    dirs = [Path(args.out_dir)]
    if args.export_paper:
        dirs.append(REPO_ROOT / "paper" / "figuras")

    plot_figure3(out_dirs=dirs, recompute=args.recompute, n_h0=args.h0_pts, n_kappa=args.kappa_pts)


if __name__ == '__main__':
    main()

