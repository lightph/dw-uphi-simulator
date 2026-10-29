#!/usr/bin/env python3
r"""
plot_figure4_wall_metrics.py - Reproduces PRB Figure 4:
Flexible extended domain wall velocities and roughness vs h0 at ha = 3 (L/L0 = 8 and 1024).
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from plot_figure4_wall_metrics import plot_figure_4


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 4 (Wall Metrics vs h0 at ha=3).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "output" / "sweep_ha3_L8_L1024_979329"),
                        help="Data directory containing sweep results")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"), help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate figure variants
    plot_figure_4(args.data_dir, str(out_dir / "sweep_ha_3.pdf"), include_legend=False)
    plot_figure_4(args.data_dir, str(out_dir / "sweep_ha_3.003_caption_only.pdf"), include_legend=False)
    plot_figure_4(args.data_dir, str(out_dir / "sweep_ha_3.003_with_legend.pdf"), include_legend=True)

    if args.export_paper:
        paper_dest = REPO_ROOT / "paper" / "figuras" / "sweep_ha_3.pdf"
        paper_dest.parent.mkdir(parents=True, exist_ok=True)
        plot_figure_4(args.data_dir, str(paper_dest), include_legend=False)
        print(f"Exported: {paper_dest}")


if __name__ == '__main__':
    main()

