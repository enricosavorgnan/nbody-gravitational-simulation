# High-Performance Direct N-Body Simulation

An optimized $O(N^2)$ Direct N-Body Gravitational simulation starting from a Plummer sphere initial condition.

This repository contains the codebase and analysis for the High Performance Computing coursework at the University of Trieste.

## Overview

The N-body problem fundamentally simulates the evolution of a system of particles under the influence of physical forces (e.g., gravity). **Direct N-Body** integration, the algorithm adopted, rigorously computes every pairwise interaction in exactly $O(N^2)$ time. 

This project implements a numerical solver using the **DKD (Drift-Kick-Drift) Leapfrog integration scheme** and systematically applies HPC optimizations at three distinct tiers of the computing stack:

### 1. Hardware & Instruction Level (Serial)
*   **Instruction-Level Parallelism (ILP):** Loop unrolling and Register Blocking to keep the CPU pipelines saturated.
*   **Vectorization:** AVX-512 intrinsics and Fused Multiply-Add (FMA) implementations with multiple independent accumulators to hide hardware latency.
*   **Math Approximations:** Fast inverse square root algorithms (`rsqrt`) to replace expensive division operations.
*   **Cache Optimization:** Memory layout tuning to minimize L1 and L2 cache capacity misses during the inner $j$-loop iterations.

### 2. Shared-Memory Parallelism (OpenMP)
*   Thread-level parallelism using optimized OpenMP scheduling.
*   Algorithmic halving of the computational domain by exploiting **Newton's Third Law** ($F_{ij} = -F_{ji}$), correctly managed via thread-safe reductions to avoid race conditions.
*   Demonstration of **Super-Linear Strong Scaling** (e.g., 97x speedup on 64 threads) achieved by partitioning the dataset until the thread-local working set fits entirely within the L2/L3 cache hierarchy.

### 3. Distributed-Memory Parallelism (MPI)
*   Cluster-level scaling across multiple compute nodes via MPI.
*   Implementation of a **Double-Buffered Asynchronous Ring Shift** using non-blocking communications (`MPI_Isend` / `MPI_Irecv`).
*   Complete **Computation-Communication Overlap**, ensuring the high-latency InfiniBand network transfers are perfectly hidden behind the CPU's mathematical execution.

## Repository Structure
The repo is organized as follows:

```
├── configs/                # Configuration files for the simulation
├── docs/                   # Documentation files
├── src/                    # Source code files
├── experiments/            # Experiments and test cases
└── plots/                  # Plots and images
```



