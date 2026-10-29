#!/usr/bin/env python3
r"""
reproduce_all_figures.py - Master Reproduction Suite for Physical Review B Publication Figures.

Executes and recreates every figure from the paper:
  Figure 1: Schematic of the 1D domain wall coordinate system u(x,t) and \phi(x,t)
  Figure 2: Rigid domain wall synchronization, Shapiro steps, and average velocities
  Figure 3: Floquet linear stability diagrams for ha = 2 and ha = 3
  Figure 4: Flexible extended domain wall dynamics vs h0 at ha = 3 (L/L0 = 8 and 1024)
  Figure 5: Full 5-phase macroscopic dynamical phase diagram for L/L0 = 256
  Figure 6: Universal Edwards-Wilkinson scaling collapse of the NR phase
  Figure 7: Spatiotemporal cascade snapshots of the SP phase
  Figure 8: Finite-size scaling of roughness and dominant mode in the SP phase
  Figure 9: Small-system dynamical phase diagram for L/L0 = 8 (mode quantization)
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FIGURE_REGISTRY = {
    1: {
        'title': 'Fig. 1: DW Coordinates & Schematic',
        'script': 'plot_figure1_schematic.py',
        'output_pdf': 'snap_system_345.pdf',
        'paper_label': 'fig:snap'
    },
    2: {
        'title': 'Fig. 2: Rigid DW Shapiro Steps & Velocities',
        'script': 'plot_figure2_rigid_wall.py',
        'output_pdf': 'figure2.pdf',
        'paper_label': 'fig:ej_walker'
    },
    3: {
        'title': 'Fig. 3: Floquet Linear Stability Diagrams',
        'script': 'plot_figure3_floquet.py',
        'output_pdf': 'figure3.pdf',
        'paper_label': 'fig:sync_stability'
    },
    4: {
        'title': 'Fig. 4: Flexible Extended DW Dynamics vs h0',
        'script': 'plot_figure4_wall_metrics.py',
        'output_pdf': 'sweep_ha_3.pdf',
        'paper_label': 'fig:wall_metrics'
    },
    5: {
        'title': 'Fig. 5: Full Dynamical Phase Diagram (L=256)',
        'script': 'plot_figure5_phase_diagram.py',
        'output_pdf': 'phasediagram_L256.pdf',
        'paper_label': 'fig:fases'
    },
    6: {
        'title': 'Fig. 6: Edwards-Wilkinson Scaling Collapse (NR)',
        'script': 'plot_figure6_nc_ew.py',
        'output_pdf': 'NC_EW_inset.pdf',
        'paper_label': 'fig:estructura'
    },
    7: {
        'title': 'Fig. 7: SP Phase Spatiotemporal Cascade',
        'script': 'plot_figure7_sp_cascade.py',
        'output_pdf': 'sc_cascade.pdf',
        'paper_label': 'fig:sc_cascade'
    },
    8: {
        'title': 'Fig. 8: SP Phase Finite-Size Scaling vs L',
        'script': 'plot_figure8_sc_scaling.py',
        'output_pdf': 'SC_scaling.pdf',
        'paper_label': 'fig:estructuraSCvsL'
    },
    9: {
        'title': 'Fig. 9: Small-System Phase Diagram (L=8)',
        'script': 'plot_figure9_phase_diagram_L8.py',
        'output_pdf': 'phasediagram_L8.pdf',
        'paper_label': 'fig:phasediagram_smallL'
    }
}


def run_figure(fig_num, out_dir, export_paper=False):
    info = FIGURE_REGISTRY[fig_num]
    script_path = REPO_ROOT / "reproduction" / info['script']

    cmd = [sys.executable, str(script_path), "--out-dir", str(out_dir)]
    if export_paper:
        cmd.append("--export-paper")

    print(f"\n[{fig_num}/9] ========================================")
    print(f"Generating {info['title']} ({info['output_pdf']})...")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================")

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        target_path = Path(out_dir) / info['output_pdf']
        status = "SUCCESS" if target_path.exists() else "WARNING: Output file missing"
        return {'fig': fig_num, 'status': status, 'time': elapsed, 'file': str(target_path)}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"Error generating Figure {fig_num}: {e}")
        return {'fig': fig_num, 'status': f"FAILED: {e}", 'time': elapsed, 'file': None}


def main():
    parser = argparse.ArgumentParser(description="Reproduce all Physical Review B publication figures.")
    parser.add_argument("--figures", nargs="+", type=int, default=list(range(1, 10)),
                        help="List of figure numbers to recreate (default: 1 2 3 4 5 6 7 8 9)")
    parser.add_argument("--all", action="store_true", help="Recreate all 9 publication figures")
    parser.add_argument("--out-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--export-paper", action="store_true", default=True,
                        help="Export directly to paper/figuras/ for pdflatex build (default: True)")
    args = parser.parse_args()

    targets = list(range(1, 10)) if args.all else sorted(set(args.figures))
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("PHYSICAL REVIEW B PUBLICATION FIGURE REPRODUCTION PIPELINE")
    print("==================================================================")
    print(f"Output directory:      {out_dir}")
    print(f"Export to paper/figuras/: {args.export_paper}")
    print(f"Selected figures:      {targets}")
    print("==================================================================")

    results = []
    total_t0 = time.time()
    for f in targets:
        if f not in FIGURE_REGISTRY:
            print(f"Warning: Figure {f} not in registry. Skipping.")
            continue
        res = run_figure(f, out_dir, export_paper=args.export_paper)
        results.append(res)

    total_time = time.time() - total_t0

    # Summary table
    print("\n==================================================================")
    print("REPRODUCTION SUMMARY")
    print("==================================================================")
    print(f"{'Fig #':<7} | {'Paper Label':<22} | {'Status':<10} | {'Time (s)':<9} | {'Output PDF'}")
    print("-" * 75)
    all_success = True
    for r in results:
        info = FIGURE_REGISTRY[r['fig']]
        status = r['status']
        if "FAILED" in status:
            all_success = False
        print(f"Fig {r['fig']:<3} | {info['paper_label']:<22} | {status:<10} | {r['time']:<9.2f} | {info['output_pdf']}")

    print("==================================================================")
    print(f"Total time elapsed: {total_time:.2f} s")
    if all_success:
        print("ALL PUBLICATION FIGURES READY FOR LaTeX COMPILATION.")
    else:
        print("SOME FIGURES HAD ISSUES. Please review logs above.")
    print("==================================================================")


if __name__ == '__main__':
    main()

