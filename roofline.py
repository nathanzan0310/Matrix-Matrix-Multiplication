"""
Expected CSV format:
size,algo,threads,seconds,GF/s,TF/s,AI,RoofGF/s,RoofTF/s
2048,ijk,1,2.345678,7.123456,0.007123,12.312312,1600.0,1.6

Example command
    python roofline.py results.csv --peak-bw 160 --peak-tf 1.0 --out roofline.png

This script will groups points by algorithm label when available. We default to peak bandwidth = 160 GB/s, 
peak perf = 1.0 TFLOP/s.
"""
import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# CSV helpers
def read_results(csv_path: str) -> Dict[str, Tuple[List[float], List[float]]]:
    """Return mapping label -> (AIs, GFLOPs) from the CSV.

    The benchmark emits a fixed header (AI, GF/s, TF/s, algo, ...).  We assume the
    CSV respects that format.
    """
    ais_by_label: Dict[str, List[float]] = defaultdict(list)
    gfs_by_label: Dict[str, List[float]] = defaultdict(list)
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}

        for row in reader:
            if row.get('AI', '') == '':
                continue
            try:
                ai = float(row['AI'])
            except Exception:
                continue

            gf = None
            if row.get('GF/s', '') not in ('', 'nan', None):
                try:
                    gf = float(row['GF/s'])
                except Exception:
                    gf = None
            if gf is None and row.get('TF/s', '') not in ('', 'nan', None):
                try:
                    gf = float(row['TF/s']) * 1e3
                except Exception:
                    gf = None

            if gf is not None and ai > 0 and gf > 0:
                label = row.get('algo', '').strip() or 'run'
                ais_by_label[label].append(ai)
                gfs_by_label[label].append(gf)

    # Convert defaultdicts to regular dicts for downstream plotting.
    return {label: (ais_by_label[label], gfs_by_label[label]) for label in ais_by_label}


def roofline_curves(peak_bw_gbs: float, peak_tf: float, ai_min: float, ai_max: float, n: int = 200):
    """Return arrays for the bandwidth roof and compute roof (both in GF/s).

    Points are spaced logarithmically so that the characteristic straight lines of
    the roofline model appear smooth on a log-log plot.
    """
    a_vals: List[float] = []
    bw_roof: List[float] = []
    comp_roof: List[float] = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        ai = math.exp(math.log(ai_min) * (1 - t) + math.log(ai_max) * t)
        a_vals.append(ai)
        bw_roof.append(peak_bw_gbs * ai)      # GB/s * flop/byte = Gflop/s
        comp_roof.append(peak_tf * 1e3)       # TF -> GF
    return a_vals, bw_roof, comp_roof


def plot_roofline(csv_path: str, peak_bw_gbs: float = 160.0, peak_tf: float = 1.0, out_png: str = None):
    """Create a roofline plot from benchmark output."""
    grouped = read_results(csv_path)
    if not grouped:
        raise SystemExit("No usable rows found in CSV. Ensure it contains columns AI and GF/s (or TF/s).")

    all_ais = [ai for ais, _ in grouped.values() for ai in ais]
    ai_min = min(all_ais) * 0.7
    ai_max = max(all_ais) * 1.4
    ai_min = max(ai_min, 1e-3)  # avoid zero
    a_vals, bw_roof, comp_roof = roofline_curves(peak_bw_gbs, peak_tf, ai_min, ai_max)

    # Plot
    fig = plt.figure(figsize=(7.5, 5.0))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Roofs
    ax.plot(a_vals, bw_roof, label=f"Bandwidth roof ({peak_bw_gbs} GB/s)")
    ax.plot(a_vals, comp_roof, label=f"Compute roof ({peak_tf} TF/s)")

    # Measured points
    for label, (ais, gfs) in sorted(grouped.items()):
        ax.scatter(ais, gfs, label=label, marker='o')

    ax.set_xlabel("Arithmetic Intensity (flop/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline: Matrix–Matrix Multiplication")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    if out_png:
        plt.savefig(out_png, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {out_png}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot a roofline graph from matmat.cpp benchmark CSV output.")
    p.add_argument("csv", help="CSV file produced by matmat.cpp (stdout).")
    p.add_argument("--peak-bw", type=float, default=160.0, help="Peak memory bandwidth in GB/s (default: 160).")
    p.add_argument("--peak-tf", type=float, default=1.0, help="Peak FP rate in TFLOP/s (default: 1.0).")
    p.add_argument("--out", type=str, default=None, help="If set, save PNG to this path instead of showing.")
    args = p.parse_args()

    plot_roofline(args.csv, peak_bw_gbs=args.peak_bw, peak_tf=args.peak_tf, out_png=args.out)


if __name__ == "__main__":
    main()
