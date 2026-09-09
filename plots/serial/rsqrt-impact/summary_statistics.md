# Direct N-Body Simulation — Kernel Benchmark Summary

**Baseline reference**: `Naive (1/sqrt)` (Speedup = 1.00x)

| Kernel / Method | Mean Force Time | Std | Speedup | IPC | L1 Misses | Max Rel Energy Drift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (1/sqrt)** | 7.5892 s | 330.80 us | **1.00x** ± 0.00 | 0.85 | 469379576.2 | `4.35e-08` |
| **Rsqrt (1 iter)** | 4.0318 s | 398.69 us | **1.88x** ± 0.00 | 1.88 | 469257096.3 | `4.74e-06` |
| **Rsqrt (2 iter)** | 4.1326 s | 4.021 ms | **1.84x** ± 0.00 | 1.88 | 469240492.7 | `4.74e-06` |
| **Rsqrt (3 iter)** | 4.1319 s | 3.118 ms | **1.84x** ± 0.00 | 1.88 | 469244249.5 | `4.74e-06` |
| **Rsqrt+Third (1 iter)** | 3.3171 s | 338.37 us | **2.29x** ± 0.00 | 1.41 | 4033018990.8 | `4.74e-06` |
| **Rsqrt+Third (2 iter)** | 3.4300 s | 2.817 ms | **2.21x** ± 0.00 | 1.40 | 4001049738.6 | `4.74e-06` |
| **Rsqrt+Third (3 iter)** | 3.1798 s | 1.312 ms | **2.39x** ± 0.00 | 1.50 | 3862508407.9 | `4.74e-06` |

