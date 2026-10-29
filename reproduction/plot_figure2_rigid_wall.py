#!/usr/bin/env python3
r"""
plot_figure2_rigid_wall.py - Reproduces PRB Figure 2:
Rigid domain wall synchronization, Shapiro steps, and average velocities.
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rigid_wall" / "python"))

from plot_figure2 import plot_figure2


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 2 (Rigid Wall Shapiro Steps).")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    parser.add_argument("--h0-pts", type=int, default=600, help="Number of points in h0 grid")
    args = parser.parse_args()

    dirs = [Path(args.out_dir)]
    if args.export_paper:
        dirs.append(REPO_ROOT / "paper" / "figuras")

    plot_figure2(out_dirs=dirs, num_h0=args.h0_pts)


if __name__ == '__main__':
    main()

