#!/usr/bin/env python3
r"""
plot_figure1_schematic.py - Verification and asset manager for PRB Figure 1:
Schematic diagram of the 1D domain wall coordinate system (u(x,t), \phi(x,t)).

Figure 1 is the physical model schematic illustrating:
  - Domain wall interface displacement u(x,t) along \hat{x}
  - In-plane azimuthal precession angle \phi(x,t)
  - Intrinsic domain wall width \Delta and in-plane characteristic scale L_0
  - Up and down magnetic domain magnetizations (\pm M_s \hat{z})
"""

import os
import shutil
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_figure1(out_dirs=None):
    if out_dirs is None:
        out_dirs = [REPO_ROOT / "figures", REPO_ROOT / "paper" / "figuras"]

    source_candidates = [
        REPO_ROOT / "paper" / "figuras" / "snap_system_345.pdf",
        REPO_ROOT / "paper_recs" / "figuras" / "snap_system_345.pdf",
    ]

    source_fig = None
    for c in source_candidates:
        if c.exists():
            source_fig = c
            break

    if source_fig is None:
        raise FileNotFoundError("Figure 1 vector source 'snap_system_345.pdf' not found.")

    saved_paths = []
    for out_d in out_dirs:
        out_p = Path(out_d)
        out_p.mkdir(parents=True, exist_ok=True)
        dest = out_p / "snap_system_345.pdf"
        if dest != source_fig:
            shutil.copy2(source_fig, dest)
        saved_paths.append(dest)
        print(f"Verified Figure 1 asset: {dest}")

        # Render PNG preview if pdftoppm is available
        png_preview = out_p / "snap_system_345.png"
        try:
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", str(dest),
                            str(out_p / "snap_system_345")],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    return saved_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify and prepare Figure 1 asset.")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    args = parser.parse_args()

    dirs = [Path(args.out_dir)]
    if args.export_paper:
        dirs.append(REPO_ROOT / "paper" / "figuras")

    setup_figure1(out_dirs=dirs)

