"""
clean.py - Main Analysis and Benchmarking Driver for Direct N-Body Simulation

Loads profiling reports, calculates statistical aggregates and analytical speedups,
prints formatted summary tables to the terminal, exports CSV and Markdown reports,
and generates publication-quality visualization figures.
"""

import argparse
import datetime
import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path so imports work regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.parser import Experiment, load_all_experiments
from analysis.report import export_summary_csv, export_summary_markdown, print_summary_table
from analysis.vis import (
    plot,
    plot_execution_times_bar,
    plot_papi_comparison,
    plot_speedup_bar,
    plot_step_trajectories,
    plot_strong_scaling,
    plot_weak_scaling,
    save_all_plots,
)


def info(file_path: str, name_info: str, runs: int) -> pd.DataFrame:
    """
    Extracts the metric name_info from file_path across runs.
    Preserved for backwards compatibility with earlier scripts.
    """
    data = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for _ in range(runs):
            run_data = []
            found = False
            for line in file:
                if line.strip() == name_info.strip():
                    found = True
                    break
            if not found:
                break

            for line in file:
                stripped = line.strip()
                if not stripped:
                    break
                try:
                    run_data.append(float(stripped))
                except ValueError:
                    continue

            data.append(run_data)

    return pd.DataFrame(data)


def process(data_raw: List[List[pd.DataFrame]], verbose: bool = True) -> Dict[Any, Dict[Any, Dict[str, Any]]]:
    """
    Converts raw metric dataframes into statistical summaries.
    Preserved for backwards compatibility.
    """
    n_methods = len(data_raw)
    n_infos = len(data_raw[0])

    data_processed: Dict[Any, Dict[Any, Dict[str, Any]]] = {}
    for inf_idx in range(n_infos):
        for method_idx in range(n_methods):
            df = data_raw[method_idx][inf_idx]
            if df.empty:
                continue
            mean = df.mean(axis=0)
            trimmed_mean = df.apply(
                lambda x: x[x.between(x.quantile(0.25), x.quantile(0.75))].mean(), axis=0
            )
            std = df.std(axis=0)

            if inf_idx not in data_processed:
                data_processed[inf_idx] = {}
            data_processed[inf_idx][method_idx] = {
                "mean": mean,
                "trimmed_mean": trimmed_mean,
                "std": std,
            }

    return data_processed


def save(figs: List[Any], config: dict, save_path: str) -> str:
    """
    Saves figures to disk in a timestamped folder.
    Preserved for backwards compatibility.
    """
    time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    path = os.path.join(save_path, time)
    os.makedirs(path, exist_ok=True)

    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(path, f"figure_{i}.png"), dpi=300)

    with open(os.path.join(path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return path


def run_analysis(config: dict) -> None:
    """
    Main analysis pipeline orchestrator.
    Parses configuration, extracts metrics, prints terminal summary tables,
    exports CSV/MD reports, and renders high-DPI visualization plots.
    """
    files = config.get("files", [])
    if not files:
        raise ValueError("Configuration must contain a non-empty 'files' list.")

    names_method = config.get("names_method", [os.path.splitext(os.path.basename(f))[0] for f in files])
    runs = config.get("runs", 5)
    baseline_idx = config.get("baseline_idx", 0)
    save_path = config.get("save-path", "./plots/default/")

    print(f"\n[INFO] Loading {len(files)} experiment profiling files (evaluating last {runs} runs)...")

    # Load and process experiments
    experiments: List[Experiment] = load_all_experiments(
        files=files,
        names=names_method,
        max_runs=runs,
        baseline_idx=baseline_idx,
    )

    # 1. Print formatted summary table to console
    print_summary_table(experiments, baseline_idx=baseline_idx)

    # 2. Export structured reports
    os.makedirs(save_path, exist_ok=True)
    csv_file = export_summary_csv(experiments, save_path)
    md_file = export_summary_markdown(experiments, save_path, baseline_idx=baseline_idx)
    print(f"[EXPORT] Summary CSV saved: {csv_file}")
    print(f"[EXPORT] Summary Markdown saved: {md_file}")

    # 3. Generate Visualizations
    figs: Dict[str, Any] = {}
    figs["execution_time_bar"] = plot_execution_times_bar(experiments)
    figs["speedup_bar"] = plot_speedup_bar(experiments, baseline_idx=baseline_idx)
    figs["step_trajectories"] = plot_step_trajectories(experiments)

    if any(e.has_valid_papi for e in experiments):
        papi_fig = plot_papi_comparison(experiments)
        if papi_fig is not None:
            figs["papi_metrics"] = papi_fig

    if config.get("is_strong", False):
        figs["strong_scaling"] = plot_strong_scaling(experiments)
    if config.get("is_weak", False):
        figs["weak_scaling"] = plot_weak_scaling(experiments)

    # 4. Save Plots and Config Copy
    saved_plots = save_all_plots(figs, save_path)
    config_copy_path = os.path.join(save_path, "config.yaml")
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[EXPORT] Saved {len(saved_plots)} plots to {save_path}")
    for p in saved_plots:
        print(f"         - {os.path.basename(p)}")
    print(f"[EXPORT] Saved config copy: {config_copy_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Direct N-body profiling analysis and visualization pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--runs", type=int, default=None, help="Override number of runs to extract (defaults to config)")
    parser.add_argument("--baseline", type=int, default=None, help="Override baseline index (defaults to config)")
    parser.add_argument("--save-path", type=str, default=None, help="Override plot save path (defaults to config)")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.runs is not None:
        cfg["runs"] = args.runs
    if args.baseline is not None:
        cfg["baseline_idx"] = args.baseline
    if args.save_path is not None:
        cfg["save-path"] = args.save_path

    run_analysis(cfg)


if __name__ == "__main__":
    main()
