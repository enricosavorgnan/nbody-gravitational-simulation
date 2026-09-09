# Direct N-Body Simulation — Kernel Benchmark Summary

**Baseline reference**: `Naive (No Unroll)` (Speedup = 1.00x)

| Kernel / Method | Mean Force Time | Std | Speedup | IPC | L1 Misses | Max Rel Energy Drift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (No Unroll)** | 7.7735 s | 18.726 ms | **1.00x** ± 0.00 | 0.85 | 469325668.3 | `N/A` |
| **Naive (Auto Unroll)** | 7.7854 s | 11.306 ms | **1.00x** ± 0.00 | 0.85 | 469413254.7 | `N/A` |
| **Naive (N Unroll)** | 7.5943 s | 354.83 us | **1.02x** ± 0.00 | 0.75 | 469291482.2 | `N/A` |
| **Rsqrt (No Unroll)** | 4.1292 s | 4.327 ms | **1.88x** ± 0.00 | 1.88 | 469258375.0 | `N/A` |
| **Rsqrt (Auto Unroll)** | 4.1377 s | 2.134 ms | **1.88x** ± 0.00 | 1.88 | 469324344.4 | `N/A` |
| **Rsqrt (N Unroll)** | 4.0113 s | 783.95 us | **1.94x** ± 0.00 | 1.72 | 469246199.4 | `N/A` |
| **ThirdLaw (No Unroll)** | 4.4257 s | 12.111 ms | **1.76x** ± 0.01 | 0.95 | 3819935715.9 | `N/A` |
| **ThirdLaw (Auto Unroll)** | 4.3251 s | 4.604 ms | **1.80x** ± 0.00 | 0.97 | 3824524364.2 | `N/A` |
| **ThirdLaw (N Unroll)** | 4.3940 s | 3.087 ms | **1.77x** ± 0.00 | 0.85 | 3878148462.8 | `N/A` |
| **Rsqrt+Third (No Unroll)** | 3.1540 s | 17.584 ms | **2.46x** ± 0.01 | 1.52 | 3862133090.5 | `N/A` |
| **Rsqrt+Third (Auto Unroll)** | 3.3397 s | 1.611 ms | **2.33x** ± 0.01 | 1.44 | 3900075466.8 | `N/A` |
| **Rsqrt+Third (N Unroll)** | 3.0916 s | 2.223 ms | **2.51x** ± 0.01 | 1.51 | 3861477375.5 | `N/A` |

