# Direct N-Body Simulation — Kernel Benchmark Summary

**Baseline reference**: `Naive` (Speedup = 1.00x)

| Kernel / Method | Mean Force Time | Std | Speedup | IPC | L1 Misses | Max Rel Energy Drift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive** | 7.5819 s | 454.54 us | **1.00x** ± 0.00 | 0.85 | 469342700.0 | `4.35e-08` |
| **ThirdLaw** | 4.4033 s | 333.30 us | **1.72x** ± 0.00 | 0.93 | 3964560292.8 | `4.35e-08` |
| **Rsqrt** | 4.0351 s | 614.97 us | **1.88x** ± 0.00 | 1.88 | 469324843.8 | `4.74e-06` |
| **Blocks** | 7.6407 s | 70.853 ms | **0.99x** ± 0.01 | 0.98 | 34583735.5 | `4.35e-08` |
| **Rsqrt+Third** | 3.1802 s | 3.102 ms | **2.38x** ± 0.00 | 1.51 | 3862065698.8 | `4.74e-06` |
| **Blocks+Third** | 4.2935 s | 3.468 ms | **1.77x** ± 0.00 | 1.12 | 59469346.0 | `4.35e-08` |
| **Blocks+Rsqrt** | 4.4083 s | 1.900 ms | **1.72x** ± 0.00 | 2.01 | 33604840.6 | `4.74e-06` |
| **Blocks+Rsqrt+Third** | 2.9433 s | 4.231 ms | **2.58x** ± 0.00 | 1.87 | 70750693.1 | `4.74e-06` |

