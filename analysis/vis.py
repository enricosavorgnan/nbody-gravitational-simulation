"""
vis.py - High-Quality Visualizations for Direct N-Body Simulation Profiling

Provides:
  - plot_execution_times_bar: Bar chart of mean execution times with error bars.
  - plot_speedup_bar: Bar chart of speedup relative to baseline with analytical error bars.
  - plot_step_trajectories: Per-step trajectory curves showing stability across simulation steps.
  - plot_papi_comparison: Hardware counter comparison (L1/L2 misses, cycles, IPC) for memory layout studies.
  - plot: Legacy function for backwards compatibility with raw DataFrame dicts.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from analysis.parser import Experiment


# Modern aesthetic palette
PALETTE = [
    "#2563EB",  # Blue
    "#059669",  # Emerald
    "#D97706",  # Amber
    "#DC2626",  # Red
    "#7C3AED",  # Violet
    "#0891B2",  # Cyan
    "#DB2777",  # Pink
    "#4B5563",  # Cool Gray
    "#4F46E5",  # Indigo
    "#10B981",  # Teal
    "#F59E0B",  # Yellow-orange
    "#8B5CF6",  # Purple
    "#EC4899",  # Magenta
    "#14B8A6",  # Teal-cyan
]


def _format_time_label(t: float) -> str:
    """Helper to format time values nicely for plot labels."""
    if t >= 1.0:
        return f"{t:.3f} s"
    elif t >= 1e-3:
        return f"{t * 1e3:.1f} ms"
    elif t >= 1e-6:
        return f"{t * 1e6:.1f} µs"
    else:
        return f"{t:.2e} s"


def plot_execution_times_bar(
    experiments: List[Experiment],
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plots mean execution time (with std error bars) across all methods as a bar chart.
    """
    names = [e.name for e in experiments]
    means = [e.mean_force_time for e in experiments]
    stds = [e.std_force_time for e in experiments]

    n = len(names)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5.5), dpi=300)
    bars = ax.bar(
        range(n),
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        alpha=0.88,
        edgecolor="#1F2937",
        linewidth=1.0,
    )

    # Label on top of each bar
    max_val = max(means) if means else 1.0
    for bar, m, s in zip(bars, means, stds):
        y_pos = bar.get_height() + (s if s > 0 else 0)
        ax.annotate(
            _format_time_label(m),
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=30 if n > 5 else 0, ha="right" if n > 5 else "center")
    ax.set_ylabel("Compute Force Time (s)", fontsize=11, fontweight="bold")
    ax.set_title(title or "Mean Force Computation Time per DKD Step", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, max_val * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_speedup_bar(
    experiments: List[Experiment],
    baseline_idx: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plots speedup relative to the baseline experiment with error bars.
    """
    names = [e.name for e in experiments]
    speedups = [e.speedup for e in experiments]
    errs = [e.speedup_err for e in experiments]

    n = len(names)
    # Highlight baseline in neutral gray; speedups >= 1 in emerald/blue; slowdowns in light red
    colors = []
    for i, sp in enumerate(speedups):
        if i == baseline_idx:
            colors.append("#6B7280")  # Baseline gray
        elif sp >= 1.0:
            colors.append("#059669")  # Green / Emerald
        else:
            colors.append("#DC2626")  # Red (slowdown)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5.5), dpi=300)
    bars = ax.bar(
        range(n),
        speedups,
        yerr=errs,
        capsize=4,
        color=colors,
        alpha=0.88,
        edgecolor="#1F2937",
        linewidth=1.0,
    )

    # Reference line at y = 1.0
    base_name = experiments[baseline_idx].name
    ax.axhline(1.0, color="#374151", linestyle="--", linewidth=1.2, label=f"Baseline: {base_name} (1.0x)")

    max_sp = max(speedups) if speedups else 1.0
    for bar, sp, err in zip(bars, speedups, errs):
        y_pos = bar.get_height() + (err if err > 0 else 0)
        ax.annotate(
            f"{sp:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=30 if n > 5 else 0, ha="right" if n > 5 else "center")
    ax.set_ylabel("Speedup (relative to baseline)", fontsize=11, fontweight="bold")
    ax.set_title(title or f"Speedup vs. Baseline ({base_name})", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, max(max_sp * 1.2, 1.25))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_step_trajectories(
    experiments: List[Experiment],
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plots per-step execution time trajectories across the 25 simulation steps.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    n_steps = experiments[0].n_steps
    steps = np.arange(1, n_steps + 1)

    for i, exp in enumerate(experiments):
        color = PALETTE[i % len(PALETTE)]
        # Mean across runs for each step
        step_means = np.mean(exp.force_time_matrix, axis=0)
        step_stds = np.std(exp.force_time_matrix, axis=0)

        ax.plot(
            steps,
            step_means,
            label=f"{exp.name} ({_format_time_label(exp.mean_force_time)})",
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
        ax.fill_between(
            steps,
            step_means - step_stds,
            step_means + step_stds,
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Simulation Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Compute Force Time (s)", fontsize=11, fontweight="bold")
    ax.set_title(title or "Compute Force Execution Time across Steps (Mean ± 1σ)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(1, n_steps)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_papi_comparison(
    experiments: List[Experiment],
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plots hardware performance counters (L1 misses, L2 misses, cycles, IPC).
    Returns None if no valid PAPI data is found.
    """
    valid = [e for e in experiments if e.has_valid_papi]
    if not valid:
        return None

    names = [e.name for e in valid]
    n = len(names)

    l1_misses = [e.papi_means.get("l1_misses", 0.0) for e in valid]
    l2_misses = [e.papi_means.get("l2_misses", 0.0) for e in valid]
    cycles = [e.papi_means.get("cycles", 0.0) for e in valid]
    ipcs = [e.ipc for e in valid]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    # Subplot 1: L1 Misses
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(n), l1_misses, color=colors, alpha=0.88, edgecolor="#1F2937")
    ax1.set_title("L1 Data Cache Misses (per step)", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(names, rotation=25 if n > 4 else 0, ha="right" if n > 4 else "center", fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars1, l1_misses):
        ax1.annotate(f"{v:.2e}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Subplot 2: L2 Misses
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(n), l2_misses, color=colors, alpha=0.88, edgecolor="#1F2937")
    ax2.set_title("L2 Cache Misses (per step)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(names, rotation=25 if n > 4 else 0, ha="right" if n > 4 else "center", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars2, l2_misses):
        ax2.annotate(f"{v:.2e}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Subplot 3: CPU Cycles
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(n), cycles, color=colors, alpha=0.88, edgecolor="#1F2937")
    ax3.set_title("CPU Cycles (per step)", fontsize=10.5, fontweight="bold")
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(names, rotation=25 if n > 4 else 0, ha="right" if n > 4 else "center", fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars3, cycles):
        ax3.annotate(f"{v:.2e}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Subplot 4: IPC
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(n), ipcs, color=colors, alpha=0.88, edgecolor="#1F2937")
    ax4.set_title("Instructions Per Cycle (IPC)", fontsize=10.5, fontweight="bold")
    ax4.set_xticks(range(n))
    ax4.set_xticklabels(names, rotation=25 if n > 4 else 0, ha="right" if n > 4 else "center", fontsize=8.5)
    ax4.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars4, ipcs):
        ax4.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title or "PAPI Hardware Performance Counters Comparison", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout()
    return fig


def save_all_plots(figs: Dict[str, plt.Figure], save_path: str) -> List[str]:
    """
    Saves a dictionary of named figures to disk in save_path as .png files.
    """
    os.makedirs(save_path, exist_ok=True)
    saved_paths = []
    for name, fig in figs.items():
        if fig is not None:
            out_file = os.path.join(save_path, f"{name}.png")
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            saved_paths.append(out_file)
            plt.close(fig)
    return saved_paths


def plot(data: dict, names_info=None, names_method=None):
    """
    Legacy plotting function preserved for backwards compatibility.
    """
    figs = []
    for i, (info, methods) in enumerate(data.items()):
        fig, ax = plt.subplots(dpi=300)
        for j, (method, stats) in enumerate(methods.items()):
            label = names_method[j] if (names_method and j < len(names_method)) else f"Method {j}"
            ax.plot(stats["trimmed_mean"], label=label, linewidth=1.2, linestyle="--")
        info_title = names_info[i] if (names_info and i < len(names_info)) else f"Info {info}"
        ax.set_title(f"{info_title} Trimmed Means")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Time (s)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        figs.append(fig)
    return figs


def plot_strong_scaling(
    experiments: List[Experiment],
    title: Optional[str] = None,
) -> plt.Figure:
    """Plots strong scaling efficiency (Speedup) based on MPI wall-clock time."""
    # Assume experiments are ordered by ranks: e.g. 1, 2, 4, 8, 16...
    # The user provides the names, we can extract ranks from names or assume they are given sequentially
    names = [e.name for e in experiments]
    speedups = [e.scaling_speedup for e in experiments]
    
    # Try to parse ranks from the names (e.g. "1 Rank", "2 Ranks")
    # If not parsable, just use index 1, 2, 4, etc.
    try:
        ranks = [int(''.join(filter(str.isdigit, name))) for name in names]
        if not all(ranks): ranks = [2**i for i in range(len(names))]
    except:
        ranks = [2**i for i in range(len(names))]
        
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot measured speedup
    ax.plot(ranks, speedups, marker='o', color="#2563EB", linewidth=2, label="Measured Speedup (MPI Wall-clock)")
    
    # Plot ideal speedup (Linear)
    ideal_speedup = [r / ranks[0] for r in ranks]
    ax.plot(ranks, ideal_speedup, linestyle='--', color="#4B5563", linewidth=2, label="Ideal Speedup")
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_yticks(ranks)
    ax.set_yticklabels([str(r) for r in ranks])
    
    ax.set_xlabel("MPI Ranks", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup", fontsize=11, fontweight="bold")
    ax.set_title(title or "Strong Scaling Speedup", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    return fig


def plot_weak_scaling(
    experiments: List[Experiment],
    title: Optional[str] = None,
) -> plt.Figure:
    """Plots weak scaling efficiency based on MPI wall-clock time."""
    names = [e.name for e in experiments]
    # Weak scaling efficiency = t_1 / t_n
    # Which is exactly the scaling_speedup we computed (since t_base_mpi / t_exp_mpi)
    efficiency = [e.scaling_speedup for e in experiments]
    
    try:
        ranks = [int(''.join(filter(str.isdigit, name))) for name in names]
        if not all(ranks): ranks = [2**i for i in range(len(names))]
    except:
        ranks = [2**i for i in range(len(names))]
        
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.plot(ranks, efficiency, marker='o', color="#059669", linewidth=2, label="Measured Efficiency")
    ax.axhline(1.0, linestyle='--', color="#4B5563", linewidth=2, label="Ideal Efficiency")
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    
    ax.set_xlabel("MPI Ranks", fontsize=11, fontweight="bold")
    ax.set_ylabel("Efficiency (t_1 / t_n)", fontsize=11, fontweight="bold")
    ax.set_title(title or "Weak Scaling Efficiency", fontsize=12, fontweight="bold", pad=12)
    
    # Set y-axis to focus around 0 to 1.2
    ax.set_ylim(0, max(1.2, max(efficiency) * 1.1))
    
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    
    fig.tight_layout()
    return fig
