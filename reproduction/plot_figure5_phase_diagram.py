#!/usr/bin/env python3
r"""
plot_figure5_phase_diagram.py - Reproduces PRB Figure 5:
Full 5-phase macroscopic dynamical phase diagram in the (h0, ha) plane for L/L0 = 256
(PF, SF, SP, NF, NR), with solid contiguous regions, analytical Shapiro boundaries,
Floquet instability contours, and bottom DC baseline strip.
"""

import sys
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 5 (Phase Diagram L/L0 = 256).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "output" / "sweep_full_973970"),
                        help="Data directory containing 2D sweep results (L=256)")
    parser.add_argument("--floquet-bin", default=str(REPO_ROOT / "output" / "floquet_transient_omega_1.000_L_256.bin"),
                        help="Path to Floquet binary stability dataset for L=256")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"), help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    args = parser.parse_args()

    # Fallback floquet file if L_256.bin not found
    floquet_file = Path(args.floquet_bin)
    if not floquet_file.exists():
        floquet_file = REPO_ROOT / "output" / "floquet_transient_omega_1.000_L_1024.bin"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "python" / "plot_prb_publication.py"),
        "--data-dir", str(args.data_dir),
        "--floquet-bin", str(floquet_file),
        "-o", str(args.out_dir)
    ]
    if args.export_paper:
        cmd.append("--export-paper")

    print(f"Running Figure 5 generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Ensure phasediagram_L256.pdf exists in out_dir
    import shutil
    src_pdf = Path(args.out_dir) / "prb_phase_diagram.pdf"
    dst_pdf = Path(args.out_dir) / "phasediagram_L256.pdf"
    if src_pdf.exists():
        shutil.copyfile(src_pdf, dst_pdf)
        print(f"Saved: {dst_pdf}")
    if args.export_paper:
        paper_dest = REPO_ROOT / "paper" / "figuras" / "phasediagram_L256.pdf"
        if src_pdf.exists() and not paper_dest.exists():
            shutil.copyfile(src_pdf, paper_dest)
            print(f"Exported to paper figures: {paper_dest}")


if __name__ == '__main__':
    main()
