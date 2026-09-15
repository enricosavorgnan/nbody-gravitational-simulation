import argparse
import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.parser import Experiment, load_all_experiments

# Custom colors requested
COLOR_NATIVE = "#2563EB"  # Blue
COLOR_CONTAINER = "#F59E0B"  # Gold

def parse_ranks(name: str) -> int:
    """Extract integer rank count from strings like 'Native 16 Ranks'."""
    for p in name.split():
        if p.isdigit():
            return int(p)
    return 1

def separate_experiments(experiments: List[Experiment]) -> Tuple[List[Experiment], List[Experiment], List[int]]:
    """Separates experiments into Native and Container lists and returns ranks."""
    natives = []
    containers = []
    ranks = []
    
    for exp in experiments:
        rank = parse_ranks(exp.name)
        if "Native" in exp.name:
            natives.append(exp)
            ranks.append(rank)
        elif "Container" in exp.name:
            containers.append(exp)
            
    return natives, containers, ranks

def plot_grouped_time_bar(natives: List[Experiment], containers: List[Experiment], ranks: List[int], save_path: str):
    """Plots grouped bar chart for Time."""
    n_groups = len(ranks)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    t_native = [e.trimmed_mean_step_time for e in natives]
    t_native_err = [e.trimmed_std_step_time for e in natives]
    
    t_container = [e.trimmed_mean_step_time for e in containers]
    t_container_err = [e.trimmed_std_step_time for e in containers]
    
    ax.bar(index, t_native, bar_width, yerr=t_native_err, capsize=4, 
           color=COLOR_NATIVE, label='Native', edgecolor='black', zorder=3)
    ax.bar(index + bar_width, t_container, bar_width, yerr=t_container_err, capsize=4, 
           color=COLOR_CONTAINER, label='Container', edgecolor='black', zorder=3)
           
    ax.set_xlabel('MPI Ranks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trimmed Mean Step Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Native vs Container: Step Time per Rank', fontsize=14, fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Log scale if times vary wildly (e.g. strong scaling)
    if max(max(t_native), max(t_container)) / min(min(t_native), min(t_container)) > 20:
        ax.set_yscale('log')
        
    fig.tight_layout()
    out_file = os.path.join(save_path, "container_time_bar.png")
    fig.savefig(out_file)
    print(f"Saved {out_file}")
    plt.close(fig)

def plot_grouped_speedup_bar(natives: List[Experiment], containers: List[Experiment], ranks: List[int], save_path: str):
    """Plots grouped bar chart for Speedup."""
    n_groups = len(ranks)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    sp_native = [e.scaling_speedup for e in natives]
    sp_container = [e.scaling_speedup for e in containers]
    
    bars_n = ax.bar(index, sp_native, bar_width, color=COLOR_NATIVE, label='Native', edgecolor='black', zorder=3)
    bars_c = ax.bar(index + bar_width, sp_container, bar_width, color=COLOR_CONTAINER, label='Container', edgecolor='black', zorder=3)
           
    # Annotate values on top of bars
    for bar in bars_n:
        y_pos = bar.get_height()
        ax.annotate(f"{y_pos:.1f}x", xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", rotation=45)
                    
    for bar in bars_c:
        y_pos = bar.get_height()
        ax.annotate(f"{y_pos:.1f}x", xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", rotation=45)

    ax.set_xlabel('MPI Ranks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup relative to Baseline', fontsize=12, fontweight='bold')
    ax.set_title('Native vs Container: Scaling Speedup', fontsize=14, fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([str(r) for r in ranks])
    
    # Increase y-limit slightly to fit annotations
    max_sp = max(max(sp_native), max(sp_container))
    ax.set_ylim(0, max_sp * 1.15)
    
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    fig.tight_layout()
    out_file = os.path.join(save_path, "container_speedup_bar.png")
    fig.savefig(out_file)
    print(f"Saved {out_file}")
    plt.close(fig)

def plot_scaling_lines(natives: List[Experiment], containers: List[Experiment], ranks: List[int], is_strong: bool, save_path: str):
    """Plots scaling lines (Native vs Container)."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    sp_native = [e.scaling_speedup for e in natives]
    sp_container = [e.scaling_speedup for e in containers]
    
    ax.plot(ranks, sp_native, marker='o', linewidth=2.5, markersize=8, color=COLOR_NATIVE, label='Native')
    ax.plot(ranks, sp_container, marker='s', linewidth=2.5, markersize=8, color=COLOR_CONTAINER, label='Container')
    
    if is_strong:
        # Ideal strong scaling: Speedup = P / P_baseline
        # Assuming baseline is the first rank in the list
        base_rank = ranks[0]
        ideal = [r / base_rank for r in ranks]
        ax.plot(ranks, ideal, linestyle='--', color='black', alpha=0.7, label='Ideal Strong Scaling', zorder=1)
        ax.set_title('Strong Scaling: Native vs Container', fontsize=14, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    else:
        # Ideal weak scaling: Speedup = 1.0 (or time is constant)
        ax.axhline(1.0, linestyle='--', color='black', alpha=0.7, label='Ideal Weak Scaling (Efficiency=1)', zorder=1)
        ax.set_title('Weak Scaling: Native vs Container', fontsize=14, fontweight='bold')
        ax.set_ylabel('Scaling Efficiency', fontsize=12, fontweight='bold')
        
    ax.set_xlabel('MPI Ranks', fontsize=12, fontweight='bold')
    
    # Use log-log scale for strong scaling curves since it looks much better for exponential ranks
    ax.set_xscale('log', base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    
    if is_strong:
        ax.set_yscale('log', base=2)
        ax.set_yticks(ranks)
        ax.set_yticklabels([str(r) for r in ranks])
        
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    fig.tight_layout()
    prefix = "strong" if is_strong else "weak"
    out_file = os.path.join(save_path, f"container_{prefix}_lines.png")
    fig.savefig(out_file)
    print(f"Saved {out_file}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate Native vs Container comparison plots.")
    parser.add_argument("--config", type=str, required=True, help="Path to scaling YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    files = cfg.get("files", [])
    names = cfg.get("names_method", [])
    runs = cfg.get("runs", 5)
    baseline_idx = cfg.get("baseline_idx", 0)
    save_path = cfg.get("save-path", "./reports/container/comparison/")
    is_strong = cfg.get("is_strong", False)

    os.makedirs(save_path, exist_ok=True)
    
    experiments = load_all_experiments(
        files=files,
        names=names,
        max_runs=runs,
        baseline_idx=baseline_idx,
    )
    
    natives, containers, ranks = separate_experiments(experiments)
    
    if not natives or not containers:
        print("[ERROR] Could not separate Native and Container experiments. Check your config names.")
        sys.exit(1)
        
    plot_grouped_time_bar(natives, containers, ranks, save_path)
    plot_grouped_speedup_bar(natives, containers, ranks, save_path)
    plot_scaling_lines(natives, containers, ranks, is_strong, save_path)
    
    print("\nAll container comparison plots generated successfully!")

if __name__ == "__main__":
    main()
