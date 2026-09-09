# Direct N-Body Simulation — Kernel Benchmark Summary

**Baseline reference**: `Naive AoS` (Speedup = 1.00x)

| Kernel / Method | Mean Force Time | Std | Speedup | IPC | L1 Misses | Max Rel Energy Drift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive AoS** | 33.9305 s | 122.459 ms | **1.00x** ± 0.01 | 1.72 | 12502581808.0 | `N/A` |
| **Naive SoA** | 7.8501 s | 6.874 ms | **4.32x** ± 0.02 | 0.99 | 3753435319.5 | `N/A` |
| **ThirdLaw AoS** | 17.6559 s | 19.680 ms | **1.92x** ± 0.01 | 1.83 | 6251142059.1 | `N/A` |
| **ThirdLaw SoA** | 4.5766 s | 18.767 ms | **7.41x** ± 0.04 | 0.92 | 4000501084.1 | `N/A` |

