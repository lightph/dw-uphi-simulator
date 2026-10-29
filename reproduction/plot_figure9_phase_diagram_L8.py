#!/usr/bin/env python3
r"""
plot_figure9_phase_diagram_L8.py - Reproduces PRB Figure 9:
Finite-size phase diagram for small systems (L/L0 = 8), showing spatial mode quantization,
suppression of the NR phase, and downshift of the critical field hc to ~3.8.
"""

import sys
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Recreate PRB Figure 9 (Phase Diagram L/L0 = 8).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "output" / "sweep_full_973971"),
                        help="Data directory containing full 2D L=8 sweep results")
    parser.add_argument("--floquet-bin", default=str(REPO_ROOT / "output" / "floquet_transient_omega_1.000_L_8.bin"),
                        help="Path to Floquet binary stability dataset for L=8")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"), help="Output directory")
    parser.add_argument("--export-paper", action="store_true", help="Also export to paper/figuras/")
    args = parser.parse_args()

    floquet_file = Path(args.floquet_bin)
    if not floquet_file.exists():
        floquet_file = REPO_ROOT / "to_integrate" / "stability" / "floquet_transient_omega_1.000_L_8.bin"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "python" / "plot_prb_publication.py"),
        "--data-dir", str(args.data_dir),
        "-o", str(args.out_dir)
    ]
    if floquet_file.exists():
        cmd.extend(["--floquet-bin", str(floquet_file)])
    else:
        cmd.append("--no-floquet")

    if args.export_paper:
        cmd.append("--export-paper")

    print(f"Running Figure 9 generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Ensure phasediagram_L8.pdf exists in out_dir
    import shutil
    src_pdf = Path(args.out_dir) / "prb_phase_diagram.pdf"
    dst_pdf = Path(args.out_dir) / "phasediagram_L8.pdf"
    if src_pdf.exists():
        shutil.copyfile(src_pdf, dst_pdf)
        print(f"Saved: {dst_pdf}")
    if args.export_paper:
        paper_dest = REPO_ROOT / "paper" / "figuras" / "phasediagram_L8.pdf"
        if src_pdf.exists() and not paper_dest.exists():
            shutil.copyfile(src_pdf, paper_dest)
            print(f"Exported to paper figures: {paper_dest}")


if __name__ == '__main__':
    main()
