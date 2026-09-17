#!/usr/bin/env python3
"""
Generate OSU micro-benchmark plots (native vs container) from ORFEO data.
Reads the raw benchmark outputs produced by osu.sbatch, computes medians over
repetitions, and produces latency and bandwidth comparison plots with ratio subplots,
following the exact design developed by Luca.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_osu_file(filepath):
    """Parse an official OSU output file into a dict of {size_bytes: value}."""
    data = {}
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    size = int(parts[0])
                    val = float(parts[1])
                    data[size] = val
                except ValueError:
                    continue
    return data

def load_median_series(file_pattern):
    """Load multiple repetition files and compute median value for each message size."""
    files = sorted(glob.glob(file_pattern))
    if not files:
        return None, None
    
    all_runs = {}
    for f in files:
        run_data = parse_osu_file(f)
        for size, val in run_data.items():
            all_runs.setdefault(size, []).append(val)
            
    sizes = sorted(all_runs.keys())
    medians = [np.median(all_runs[s]) for s in sizes]
    return np.array(sizes), np.array(medians)

def format_size_label(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1048576:
        return f"{b // 1024} KB"
    else:
        return f"{b // 1048576} MB"

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    osu_dir = os.path.join(base_dir, "profilings", "container", "osu")
    out_dir = os.path.join(base_dir, "reports", "container")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Latency Data
    lat_sizes_nat, lat_nat = load_median_series(os.path.join(osu_dir, "*native*latency*.txt"))
    lat_sizes_cont, lat_cont = load_median_series(os.path.join(osu_dir, "*container*latency*.txt"))

    if lat_sizes_nat is not None and lat_sizes_cont is not None:
        common_sizes = np.intersect1d(lat_sizes_nat, lat_sizes_cont)
        # Filter to common sizes
        lat_nat_vals = np.array([lat_nat[np.where(lat_sizes_nat == s)[0][0]] for s in common_sizes])
        lat_cont_vals = np.array([lat_cont[np.where(lat_sizes_cont == s)[0][0]] for s in common_sizes])
        lat_ratio = lat_cont_vals / lat_nat_vals
        lat_labels = [format_size_label(s) for s in common_sizes]

        fig1, (ax1, ax1r) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 1],
                                          sharex=True, gridspec_kw={'hspace': 0.08})

        ax1.loglog(common_sizes, lat_nat_vals, 'bo-', markersize=7, linewidth=2, label='Native (CMA)')
        ax1.loglog(common_sizes, lat_cont_vals, 'rs--', markersize=7, linewidth=2, label='Container (vader)')

        ax1.set_ylabel('Latency (μs)', fontsize=12)
        ax1.set_title('OSU Latency — Native vs Container\n'
                      'ORFEO Genoa / Zen 4, 2 ranks, same node', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, which="both", alpha=0.3)

        # Ratio subplot
        ax1r.semilogx(common_sizes, lat_ratio, 'k^-', markersize=6, linewidth=1.5)
        ax1r.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='2x threshold')
        ax1r.set_ylabel('Ratio (Cont/Nat)', fontsize=11)
        ax1r.set_xlabel('Message size', fontsize=12)
        ax1r.set_ylim(0.5, max(3.5, np.max(lat_ratio) * 1.15))
        ax1r.set_xticks(common_sizes[::2] if len(common_sizes) > 12 else common_sizes)
        ax1r.set_xticklabels([lat_labels[i] for i in (range(0, len(common_sizes), 2) if len(common_sizes) > 12 else range(len(common_sizes)))], 
                             fontsize=9, rotation=30)
        ax1r.grid(True, alpha=0.3)

        fig1.tight_layout(h_pad=0.5)
        lat_out = os.path.join(out_dir, 'osu_latency_orfeo.png')
        fig1.savefig(lat_out, dpi=200)
        plt.close(fig1)
        print(f"Saved: {lat_out}")
    else:
        print("Latency data not found yet in profilings/container/osu/")

    # 2. Bandwidth Data
    bw_sizes_nat, bw_nat = load_median_series(os.path.join(osu_dir, "*native*bw*.txt"))
    bw_sizes_cont, bw_cont = load_median_series(os.path.join(osu_dir, "*container*bw*.txt"))

    if bw_sizes_nat is not None and bw_sizes_cont is not None:
        common_bw = np.intersect1d(bw_sizes_nat, bw_sizes_cont)
        bw_nat_vals = np.array([bw_nat[np.where(bw_sizes_nat == s)[0][0]] for s in common_bw])
        bw_cont_vals = np.array([bw_cont[np.where(bw_sizes_cont == s)[0][0]] for s in common_bw])
        bw_ratio = bw_nat_vals / bw_cont_vals
        bw_labels = [format_size_label(s) for s in common_bw]

        fig2, (ax2, ax2r) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[3, 1],
                                          sharex=True, gridspec_kw={'hspace': 0.08})

        ax2.semilogx(common_bw, bw_nat_vals / 1000.0, 'bo-', markersize=7, linewidth=2, label='Native (CMA)')
        ax2.semilogx(common_bw, bw_cont_vals / 1000.0, 'rs--', markersize=7, linewidth=2, label='Container (vader)')

        ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax2.set_title('OSU Bandwidth — Native vs Container\n'
                      'ORFEO Genoa / Zen 4, 2 ranks, same node', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, which="both", alpha=0.3)

        # Ratio subplot
        ax2r.semilogx(common_bw, bw_ratio, 'k^-', markersize=6, linewidth=1.5)
        ax2r.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2r.set_ylabel('Ratio (Nat/Cont)', fontsize=11)
        ax2r.set_xlabel('Message size', fontsize=12)
        ax2r.set_xticks(common_bw[::2] if len(common_bw) > 12 else common_bw)
        ax2r.set_xticklabels([bw_labels[i] for i in (range(0, len(common_bw), 2) if len(common_bw) > 12 else range(len(common_bw)))], 
                             fontsize=9, rotation=30)
        ax2r.grid(True, alpha=0.3)

        fig2.tight_layout(h_pad=0.5)
        bw_out = os.path.join(out_dir, 'osu_bandwidth_orfeo.png')
        fig2.savefig(bw_out, dpi=200)
        plt.close(fig2)
        print(f"Saved: {bw_out}")
    else:
        print("Bandwidth data not found yet in profilings/container/osu/")

if __name__ == "__main__":
    main()
