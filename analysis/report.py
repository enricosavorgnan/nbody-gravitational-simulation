"""
report.py - Terminal Reporting and Data Export for Direct N-body Simulations

Generates formatted ASCII summary tables and exports results to CSV and Markdown.
"""

import os
import csv
from typing import List
from analysis.parser import Experiment


def format_time(t: float) -> str:
    """Intelligently formats time into seconds, milliseconds, or microseconds."""
    if t >= 1.0:
        return f"{t:.4f} s"
    elif t >= 1e-3:
        return f"{t * 1e3:.3f} ms"
    elif t >= 1e-6:
        return f"{t * 1e6:.2f} us"
    else:
        return f"{t:.3e} s"


def print_summary_table(experiments: List[Experiment], baseline_idx: int = 0) -> None:
    """
    Prints an aesthetic ASCII table summarizing execution times, speedups,
    energy conservation, and PAPI hardware metrics.
    """
    has_papi = any(e.has_valid_papi for e in experiments)

    print("\n" + "=" * 105)
    print("                     DIRECT N-BODY SIMULATION - KERNEL BENCHMARK SUMMARY")
    print("=" * 105)

    base_name = experiments[baseline_idx].name
    print(f"Reference Baseline: '{base_name}' (Speedup = 1.00x)\n")

    # Header
    if has_papi:
        header = (
            f"{'Method / Kernel':<22} | {'Mean Force':<12} | {'Std':<10} | {'Median':<10} | "
            f"{'Speedup':<14} | {'IPC':<6} | {'L1 Misses':<11} | {'Max Rel Err':<12}"
        )
    else:
        header = (
            f"{'Method / Kernel':<22} | {'Mean Force':<13} | {'Std':<11} | {'Median':<11} | "
            f"{'Trimmed Mean':<13} | {'Speedup':<15} | {'Max Rel Err':<12}"
        )

    print(header)
    print("-" * len(header))

    for exp in experiments:
        sp_str = f"{exp.speedup:5.2f}x +/- {exp.speedup_err:4.2f}"
        err_str = f"{exp.max_relative_error:.2e}" if exp.max_relative_error > 0 else "N/A"

        if has_papi:
            l1_str = f"{exp.papi_means.get('l1_misses', 0.0):.1f}"
            ipc_str = f"{exp.ipc:.2f}" if exp.ipc > 0 else "N/A"
            row = (
                f"{exp.name:<22} | {format_time(exp.mean_force_time):<12} | "
                f"{format_time(exp.std_force_time):<10} | {format_time(exp.median_force_time):<10} | "
                f"{sp_str:<14} | {ipc_str:<6} | {l1_str:<11} | {err_str:<12}"
            )
        else:
            row = (
                f"{exp.name:<22} | {format_time(exp.mean_force_time):<13} | "
                f"{format_time(exp.std_force_time):<11} | {format_time(exp.median_force_time):<11} | "
                f"{format_time(exp.trimmed_mean_force_time):<13} | {sp_str:<15} | {err_str:<12}"
            )
        print(row)

    print("=" * 105 + "\n")


def export_summary_csv(experiments: List[Experiment], save_path: str) -> str:
    """Exports comprehensive numerical metrics for all experiments to a CSV file."""
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, "summary_statistics.csv")

    has_papi = any(e.has_valid_papi for e in experiments)

    fields = [
        "method",
        "file_path",
        "runs",
        "steps",
        "mean_force_time_s",
        "std_force_time_s",
        "median_force_time_s",
        "trimmed_mean_force_time_s",
        "min_force_time_s",
        "max_force_time_s",
        "mean_step_time_s",
        "std_step_time_s",
        "mean_drift_time_s",
        "mean_kick_time_s",
        "speedup_vs_baseline",
        "speedup_err",
        "max_relative_energy_error",
    ]

    if has_papi:
        fields += [
            "mean_cycles",
            "mean_instructions",
            "ipc",
            "mean_l1_misses",
            "mean_l2_misses",
            "mean_vec_dp",
        ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for exp in experiments:
            row = {
                "method": exp.name,
                "file_path": exp.file_path,
                "runs": exp.n_runs,
                "steps": exp.n_steps,
                "mean_force_time_s": exp.mean_force_time,
                "std_force_time_s": exp.std_force_time,
                "median_force_time_s": exp.median_force_time,
                "trimmed_mean_force_time_s": exp.trimmed_mean_force_time,
                "min_force_time_s": exp.min_force_time,
                "max_force_time_s": exp.max_force_time,
                "mean_step_time_s": exp.mean_step_time,
                "std_step_time_s": exp.std_step_time,
                "mean_drift_time_s": exp.mean_drift_time,
                "mean_kick_time_s": exp.mean_kick_time,
                "speedup_vs_baseline": exp.speedup,
                "speedup_err": exp.speedup_err,
                "max_relative_energy_error": exp.max_relative_error,
            }
            if has_papi:
                row["mean_cycles"] = exp.papi_means.get("cycles", 0.0)
                row["mean_instructions"] = exp.papi_means.get("instructions", 0.0)
                row["ipc"] = exp.ipc
                row["mean_l1_misses"] = exp.papi_means.get("l1_misses", 0.0)
                row["mean_l2_misses"] = exp.papi_means.get("l2_misses", 0.0)
                row["mean_vec_dp"] = exp.papi_means.get("vec_dp", 0.0)

            writer.writerow(row)

    return csv_file


def export_summary_markdown(experiments: List[Experiment], save_path: str, baseline_idx: int = 0) -> str:
    """Exports a ready-to-use Markdown table for project documentation/reports."""
    os.makedirs(save_path, exist_ok=True)
    md_file = os.path.join(save_path, "summary_statistics.md")

    base_name = experiments[baseline_idx].name
    has_papi = any(e.has_valid_papi for e in experiments)

    lines = [
        "# Direct N-Body Simulation — Kernel Benchmark Summary\n",
        f"**Baseline reference**: `{base_name}` (Speedup = 1.00x)\n",
    ]

    if has_papi:
        lines.append("| Kernel / Method | Mean Force Time | Std | Speedup | IPC | L1 Misses | Max Rel Energy Drift |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for exp in experiments:
            l1_str = f"{exp.papi_means.get('l1_misses', 0.0):.1f}"
            ipc_str = f"{exp.ipc:.2f}" if exp.ipc > 0 else "N/A"
            err_str = f"{exp.max_relative_error:.2e}" if exp.max_relative_error > 0 else "N/A"
            lines.append(
                f"| **{exp.name}** | {format_time(exp.mean_force_time)} | {format_time(exp.std_force_time)} | "
                f"**{exp.speedup:.2f}x** ± {exp.speedup_err:.2f} | {ipc_str} | {l1_str} | `{err_str}` |"
            )
    else:
        lines.append("| Kernel / Method | Mean Force Time | Std | Median | Trimmed Mean (IQR) | Speedup | Max Rel Energy Drift |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for exp in experiments:
            err_str = f"{exp.max_relative_error:.2e}" if exp.max_relative_error > 0 else "N/A"
            lines.append(
                f"| **{exp.name}** | {format_time(exp.mean_force_time)} | {format_time(exp.std_force_time)} | "
                f"{format_time(exp.median_force_time)} | {format_time(exp.trimmed_mean_force_time)} | "
                f"**{exp.speedup:.2f}x** ± {exp.speedup_err:.2f} | `{err_str}` |"
            )

    lines.append("\n")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_file
