"""
parser.py - Profiling Log Parser for Direct N-body Simulation

Parses single-run and multi-run .txt reports produced by the C solver profiler.
Extracts:
  - Step timings: Compute Force, Total Step, First Drift, Kick, Second Drift
  - One-time timings: File Read, File Write, Total Run
  - Configuration metadata: nsteps, dt, eps, G, mass, kernel_choice, max_relative_error
  - Hardware PAPI counters: Cycles, Instructions, L1 Misses, L2 Misses, Vector DP
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RunData:
    """Stores data for a single execution run of an experiment."""
    run_idx: int
    force_time: np.ndarray          # shape (n_steps,)
    step_time: np.ndarray           # shape (n_steps,)
    first_drift_time: np.ndarray    # shape (n_steps,)
    kick_time: np.ndarray           # shape (n_steps,)
    second_drift_time: np.ndarray   # shape (n_steps,)
    file_read: float = 0.0
    file_write: float = 0.0
    total_run: float = 0.0
    mpi_real_time: float = 0.0
    mpi_user_time: float = 0.0
    mpi_sys_time: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    papi: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Experiment:
    """Stores aggregated data across all runs for a single method/kernel."""
    name: str
    file_path: str
    runs: List[RunData]
    n_runs: int
    n_steps: int

    # 2D Matrices: shape (n_runs, n_steps)
    force_time_matrix: np.ndarray
    step_time_matrix: np.ndarray
    first_drift_matrix: np.ndarray
    kick_matrix: np.ndarray
    second_drift_matrix: np.ndarray

    # Aggregate timing statistics for Compute Force (per step)
    mean_force_time: float
    std_force_time: float
    median_force_time: float
    trimmed_mean_force_time: float
    min_force_time: float
    max_force_time: float

    # Total step time
    mean_step_time: float
    std_step_time: float

    # Drift & Kick times
    mean_drift_time: float
    mean_kick_time: float
    
    # OS / MPI Times
    mean_mpi_real_time: float
    mean_mpi_user_time: float
    mean_mpi_sys_time: float

    # PAPI metrics
    papi_matrices: Dict[str, np.ndarray]
    papi_means: Dict[str, float]
    papi_stds: Dict[str, float]
    ipc: float
    has_valid_papi: bool

    # Metadata
    config: Dict[str, Any]
    max_relative_error: float

    # Speedup relative to baseline (set externally)
    speedup: float = 1.0
    speedup_err: float = 0.0
    
    # Scaling Speedup (based on MPI real time)
    scaling_speedup: float = 1.0


KNOWN_HEADERS = {
    "File Read",
    "File Write",
    "Total Run",
    "Total Step",
    "First Drift",
    "Compute Force",
    "Kick",
    "Second Drift",
    "--- Configuration ---",
    "--- OS / MPI Launch Time ---",
    "PAPI Cycles",
    "PAPI Instructions",
    "PAPI L1 Misses",
    "PAPI L2 Misses",
    "PAPI Vectorial DP",
}


def _parse_raw_runs(file_path: str) -> List[Dict[str, List[str]]]:
    """
    Splits a profiling text file into independent run dictionaries.
    A new run starts whenever a known header repeats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Profiling file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\r\n") for line in f]

    raw_runs: List[Dict[str, List[str]]] = []
    current_run: Dict[str, List[str]] = {}
    current_header: Optional[str] = None
    current_lines: List[str] = []

    def flush_section():
        nonlocal current_header, current_lines
        if current_header is not None:
            current_run[current_header] = current_lines
            current_header = None
            current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped in KNOWN_HEADERS:
            if stripped in current_run:
                # Encountered a header already in current_run -> start of new run
                flush_section()
                raw_runs.append(current_run)
                current_run = {}
            else:
                flush_section()
            current_header = stripped
            current_lines = []
        elif current_header is not None and stripped:
            current_lines.append(stripped)

    flush_section()
    if current_run:
        raw_runs.append(current_run)

    return raw_runs


def _parse_config(lines: List[str]) -> Dict[str, Any]:
    """Parses key-value pairs from the '--- Configuration ---' block."""
    cfg: Dict[str, Any] = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                if "." in val or "e" in val.lower():
                    cfg[key] = float(val)
                else:
                    cfg[key] = int(val)
            except ValueError:
                cfg[key] = val
    return cfg


def _to_float_array(lines: List[str]) -> np.ndarray:
    """Converts a list of numerical string lines to a float numpy array."""
    out = []
    for s in lines:
        try:
            out.append(float(s))
        except ValueError:
            continue
    return np.array(out, dtype=np.float64)


def parse_experiment(
    file_path: str,
    name: Optional[str] = None,
    max_runs: Optional[int] = None,
) -> Experiment:
    """
    Parses a profiling file and constructs an aggregated Experiment object.

    Args:
        file_path: Path to the profiling .txt file.
        name: Custom display name for this experiment. Defaults to filename stem.
        max_runs: Optional limit on the number of runs to extract.

    Returns:
        An Experiment dataclass with full timing and PAPI statistics.
    """
    raw_runs = _parse_raw_runs(file_path)
    if not raw_runs:
        raise ValueError(f"No valid profiling data found in {file_path}")

    if max_runs is not None and max_runs > 0:
        raw_runs = raw_runs[-max_runs:]

    parsed_runs: List[RunData] = []
    for idx, r in enumerate(raw_runs):
        force_time = _to_float_array(r.get("Compute Force", []))
        step_time = _to_float_array(r.get("Total Step", []))
        first_drift = _to_float_array(r.get("First Drift", []))
        kick = _to_float_array(r.get("Kick", []))
        second_drift = _to_float_array(r.get("Second Drift", []))

        # Scalar values
        read_arr = _to_float_array(r.get("File Read", ["0.0"]))
        write_arr = _to_float_array(r.get("File Write", ["0.0"]))
        run_arr = _to_float_array(r.get("Total Run", ["0.0"]))

        cfg = _parse_config(r.get("--- Configuration ---", []))
        
        # Parse MPI Launch Time
        launch_lines = r.get("--- OS / MPI Launch Time ---", [])
        real_time = 0.0
        user_time = 0.0
        sys_time = 0.0
        for line in launch_lines:
            if line.startswith("Real:"):
                real_time = float(line.split()[1])
            elif line.startswith("User:"):
                user_time = float(line.split()[1])
            elif line.startswith("Sys:"):
                sys_time = float(line.split()[1])

        # PAPI counters
        papi_data = {
            "cycles": _to_float_array(r.get("PAPI Cycles", [])),
            "instructions": _to_float_array(r.get("PAPI Instructions", [])),
            "l1_misses": _to_float_array(r.get("PAPI L1 Misses", [])),
            "l2_misses": _to_float_array(r.get("PAPI L2 Misses", [])),
            "vec_dp": _to_float_array(r.get("PAPI Vectorial DP", [])),
        }

        parsed_runs.append(
            RunData(
                run_idx=idx,
                force_time=force_time,
                step_time=step_time,
                first_drift_time=first_drift,
                kick_time=kick,
                second_drift_time=second_drift,
                file_read=float(read_arr[0]) if len(read_arr) else 0.0,
                file_write=float(write_arr[0]) if len(write_arr) else 0.0,
                total_run=float(run_arr[0]) if len(run_arr) else 0.0,
                mpi_real_time=real_time,
                mpi_user_time=user_time,
                mpi_sys_time=sys_time,
                config=cfg,
                papi=papi_data,
            )
        )

    n_runs = len(parsed_runs)
    n_steps = len(parsed_runs[0].force_time)

    # Build 2D matrices: (n_runs, n_steps)
    force_mat = np.zeros((n_runs, n_steps), dtype=np.float64)
    step_mat = np.zeros((n_runs, n_steps), dtype=np.float64)
    drift1_mat = np.zeros((n_runs, n_steps), dtype=np.float64)
    drift2_mat = np.zeros((n_runs, n_steps), dtype=np.float64)
    kick_mat = np.zeros((n_runs, n_steps), dtype=np.float64)

    papi_matrices: Dict[str, np.ndarray] = {
        k: np.zeros((n_runs, n_steps), dtype=np.float64)
        for k in ["cycles", "instructions", "l1_misses", "l2_misses", "vec_dp"]
    }

    for i, rd in enumerate(parsed_runs):
        cur_len = min(n_steps, len(rd.force_time))
        force_mat[i, :cur_len] = rd.force_time[:cur_len]
        step_mat[i, :cur_len] = rd.step_time[:cur_len]
        drift1_mat[i, :cur_len] = rd.first_drift_time[:cur_len]
        drift2_mat[i, :cur_len] = rd.second_drift_time[:cur_len]
        kick_mat[i, :cur_len] = rd.kick_time[:cur_len]

        for pk in papi_matrices:
            arr = rd.papi.get(pk, np.zeros(n_steps))
            plen = min(n_steps, len(arr))
            papi_matrices[pk][i, :plen] = arr[:plen]

    # Compute per-step aggregate statistics across runs and steps
    run_mean_forces = np.mean(force_mat, axis=1) if n_steps > 0 else np.zeros(n_runs)
    overall_mean_force = float(np.mean(run_mean_forces))

    if n_runs > 1:
        overall_std_force = float(np.std(run_mean_forces, ddof=1))
    else:
        overall_std_force = float(np.std(force_mat[0])) if n_steps > 1 else 0.0

    overall_median_force = float(np.median(force_mat))
    # 25%-75% Interquartile trimmed mean
    q25 = np.percentile(force_mat, 25)
    q75 = np.percentile(force_mat, 75)
    in_iqr = force_mat[(force_mat >= q25) & (force_mat <= q75)]
    overall_trimmed_mean = float(np.mean(in_iqr)) if len(in_iqr) else overall_mean_force

    min_force = float(np.min(force_mat))
    max_force = float(np.max(force_mat))

    # Step time statistics
    run_mean_steps = np.mean(step_mat, axis=1) if n_steps > 0 else np.zeros(n_runs)
    overall_mean_step = float(np.mean(run_mean_steps))
    overall_std_step = float(np.std(run_mean_steps, ddof=1)) if n_runs > 1 else float(np.std(step_mat[0]))

    mean_drift = float(np.mean(drift1_mat + drift2_mat))
    mean_kick = float(np.mean(kick_mat))

    # MPI Time statistics
    mean_mpi_real = float(np.mean([r.mpi_real_time for r in parsed_runs]))
    mean_mpi_user = float(np.mean([r.mpi_user_time for r in parsed_runs]))
    mean_mpi_sys = float(np.mean([r.mpi_sys_time for r in parsed_runs]))

    # PAPI statistics
    papi_means: Dict[str, float] = {}
    papi_stds: Dict[str, float] = {}
    has_valid_papi = False
    for pk, pmat in papi_matrices.items():
        m = float(np.mean(pmat))
        s = float(np.std(pmat))
        papi_means[pk] = m
        papi_stds[pk] = s
        if m > 0.0:
            has_valid_papi = True

    cycles = papi_means.get("cycles", 0.0)
    instructions = papi_means.get("instructions", 0.0)
    ipc = (instructions / cycles) if cycles > 0 else 0.0

    cfg0 = parsed_runs[0].config
    max_err = float(cfg0.get("max_relative_error", 0.0))

    disp_name = name if name else os.path.splitext(os.path.basename(file_path))[0]

    return Experiment(
        name=disp_name,
        file_path=file_path,
        runs=parsed_runs,
        n_runs=n_runs,
        n_steps=n_steps,
        force_time_matrix=force_mat,
        step_time_matrix=step_mat,
        first_drift_matrix=drift1_mat,
        kick_matrix=kick_mat,
        second_drift_matrix=drift2_mat,
        mean_force_time=overall_mean_force,
        std_force_time=overall_std_force,
        median_force_time=overall_median_force,
        trimmed_mean_force_time=overall_trimmed_mean,
        min_force_time=min_force,
        max_force_time=max_force,
        mean_step_time=overall_mean_step,
        std_step_time=overall_std_step,
        mean_drift_time=mean_drift,
        mean_kick_time=mean_kick,
        mean_mpi_real_time=mean_mpi_real,
        mean_mpi_user_time=mean_mpi_user,
        mean_mpi_sys_time=mean_mpi_sys,
        papi_matrices=papi_matrices,
        papi_means=papi_means,
        papi_stds=papi_stds,
        ipc=ipc,
        has_valid_papi=has_valid_papi,
        config=cfg0,
        max_relative_error=max_err,
    )


def load_all_experiments(
    files: List[str],
    names: Optional[List[str]] = None,
    max_runs: Optional[int] = None,
    baseline_idx: int = 0,
) -> List[Experiment]:
    """
    Loads all experiment files and computes speedups relative to baseline_idx.
    """
    experiments: List[Experiment] = []
    for i, path in enumerate(files):
        exp_name = names[i] if (names and i < len(names)) else None
        exp = parse_experiment(path, name=exp_name, max_runs=max_runs)
        experiments.append(exp)

    # Compute speedup relative to baseline
    base = experiments[baseline_idx]
    t_base = base.mean_force_time
    s_base = base.std_force_time
    t_base_mpi = base.mean_mpi_real_time

    for exp in experiments:
        t_exp = exp.mean_force_time
        s_exp = exp.std_force_time
        if t_exp > 0:
            sp = t_base / t_exp
            # Error propagation: delta S = S * sqrt((s_exp/t_exp)^2 + (s_base/t_base)^2)
            rel_err_sq = (s_exp / t_exp) ** 2 + ((s_base / t_base) ** 2 if t_base > 0 else 0)
            sp_err = sp * np.sqrt(rel_err_sq)
            exp.speedup = sp
            exp.speedup_err = sp_err
        else:
            exp.speedup = 0.0
            exp.speedup_err = 0.0

        if exp.mean_mpi_real_time > 0:
            exp.scaling_speedup = t_base_mpi / exp.mean_mpi_real_time
        else:
            exp.scaling_speedup = 0.0

    return experiments
