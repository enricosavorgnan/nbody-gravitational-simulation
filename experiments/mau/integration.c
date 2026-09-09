// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif


/* =========================================================================
 * 1. Naive Baseline (Branchless, Auto-Vectorized)
 * ========================================================================= */
void compute_accelerations_naive (const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  const dtype * restrict x,
                                  const dtype * restrict y,
                                  const dtype * restrict z,
                                  dtype * restrict ax,
                                  dtype * restrict ay,
                                  dtype * restrict az
                                 )
{
  const dtype eps2 = eps * eps;

  for (size_t i = 0u; i < n; ++i)
  {
    const dtype xi  = x[i];
    const dtype yi  = y[i];
    const dtype zi  = z[i];
    dtype       axi = 0.0;
    dtype       ayi = 0.0;
    dtype       azi = 0.0;

    for (size_t j = 0u; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axi += dx * s;
      ayi += dy * s;
      azi += dz * s;
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
}


/* =========================================================================
 * 2. MAU (Multiple Accumulator Units) using Discrete Registers
 * ========================================================================= */
void compute_accelerations_mau2(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * restrict x,
                                    const dtype * restrict y,
                                    const dtype * restrict z,
                                    dtype       * restrict ax,
                                    dtype       * restrict ay,
                                    dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[2] = {0.0};
        dtype ayc[2] = {0.0};
        dtype azc[2] = {0.0};

        size_t j = 0u;
        for (; j + 1u < n; j += 2u)
        {
          #pragma GCC unroll 2
          for (size_t u = 0u; u < 2u; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 2
        for (size_t u = 0u; u < 2u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


void compute_accelerations_mau4(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * restrict x,
                                    const dtype * restrict y,
                                    const dtype * restrict z,
                                    dtype       * restrict ax,
                                    dtype       * restrict ay,
                                    dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[4] = {0.0};
        dtype ayc[4] = {0.0};
        dtype azc[4] = {0.0};

        size_t j = 0u;
        for (; j + 3u < n; j += 4u)
        {
          #pragma GCC unroll 4
          for (size_t u = 0u; u < 4u; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 4
        for (size_t u = 0u; u < 4u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


void compute_accelerations_mau8(const size_t  n,
                                const dtype   g,
                                const dtype   mass,
                                const dtype   eps,
                                const dtype * restrict x,
                                const dtype * restrict y,
                                const dtype * restrict z,
                                dtype       * restrict ax,
                                dtype       * restrict ay,
                                dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[8] = {0.0};
        dtype ayc[8] = {0.0};
        dtype azc[8] = {0.0};

        size_t j = 0u;
        for (; j + 7u < n; j += 8u)
        {
          #pragma GCC unroll 8
          for (size_t u = 0u; u < 8u; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 8
        for (size_t u = 0u; u < 8u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


void compute_accelerations_mau16(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * restrict x,
                                    const dtype * restrict y,
                                    const dtype * restrict z,
                                    dtype       * restrict ax,
                                    dtype       * restrict ay,
                                    dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[16] = {0.0};
        dtype ayc[16] = {0.0};
        dtype azc[16] = {0.0};

        size_t j = 0u;
        for (; j + 15u < n; j += 16u)
        {
          #pragma GCC unroll 16
          for (size_t u = 0u; u < 16; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 16
        for (size_t u = 0u; u < 16u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


/* =========================================================================
 * 3. Fast RSQRT + MAU using Discrete Registers
 * ========================================================================= */
void compute_accelerations_rsqrt_mau2(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * restrict x,
                                    const dtype * restrict y,
                                    const dtype * restrict z,
                                    dtype       * restrict ax,
                                    dtype       * restrict ay,
                                    dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[2] = {0.0};
        dtype ayc[2] = {0.0};
        dtype azc[2] = {0.0};

        size_t j = 0u;
        for (; j + 1u < n; j += 2u)
        {
          #pragma GCC unroll 2
          for (size_t u = 0u; u < 2u; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = dtype_rsqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 2
        for (size_t u = 0u; u < 2u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


void compute_accelerations_rsqrt_mau4(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * restrict x,
                                    const dtype * restrict y,
                                    const dtype * restrict z,
                                    dtype       * restrict ax,
                                    dtype       * restrict ay,
                                    dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[4] = {0.0};
        dtype ayc[4] = {0.0};
        dtype azc[4] = {0.0};

        size_t j = 0u;
        for (; j + 3u < n; j += 4u)
        {
          #pragma GCC unroll 4
          for (size_t u = 0u; u < 4u; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = dtype_rsqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 4
        for (size_t u = 0u; u < 4u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


void compute_accelerations_rsqrt_mau8(const size_t  n,
                                const dtype   g,
                                const dtype   mass,
                                const dtype   eps,
                                const dtype * restrict x,
                                const dtype * restrict y,
                                const dtype * restrict z,
                                dtype       * restrict ax,
                                dtype       * restrict ay,
                                dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[8] = {0.0};
        dtype ayc[8] = {0.0};
        dtype azc[8] = {0.0};

        size_t j = 0u;
        for (; j + 7u < n; j += 8u)
        {
          #pragma GCC unroll 8
          for (size_t u = 0u; u < 8u; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = dtype_rsqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 8
        for (size_t u = 0u; u < 8u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


void compute_accelerations_rsqrt_mau16(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * restrict x,
                                    const dtype * restrict y,
                                    const dtype * restrict z,
                                    dtype       * restrict ax,
                                    dtype       * restrict ay,
                                    dtype       * restrict az)
    {
      const dtype eps2 = eps * eps;

      for (size_t i = 0u; i < n; ++i)
      {
        const dtype xi = x[i];
        const dtype yi = y[i];
        const dtype zi = z[i];

        dtype axc[16] = {0.0};
        dtype ayc[16] = {0.0};
        dtype azc[16] = {0.0};

        size_t j = 0u;
        for (; j + 15u < n; j += 16u)
        {
          #pragma GCC unroll 16
          for (size_t u = 0u; u < 16; ++u)
          {
            const dtype dx   = x[j + u] - xi;
            const dtype dy   = y[j + u] - yi;
            const dtype dz   = z[j + u] - zi;
            const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = dtype_rsqrt(r2);
            const dtype s    = g * mass * invr * invr * invr;

            axc[u] += dx * s;
            ayc[u] += dy * s;
            azc[u] += dz * s;
          }
        }

        dtype axi = 0.0;
        dtype ayi = 0.0;
        dtype azi = 0.0;

        for (; j < n; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt(r2);
          const dtype s    = g * mass * invr * invr * invr;

          axi += dx * s;
          ayi += dy * s;
          azi += dz * s;
        }

        #pragma GCC unroll 16
        for (size_t u = 0u; u < 16u; ++u)
        {
          axi += axc[u];
          ayi += ayc[u];
          azi += azc[u];
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
      }
    }


/* =========================================================================
 * 4. Blocked + RSQRT + MAU (Fixed Loop Ordering, Single Memory Store)
 * ========================================================================= */
void compute_accelerations_blocks_rsqrt_mau2(const size_t  n,
                                                 const dtype   g,
                                                 const dtype   mass,
                                                 const dtype   eps,
                                                 const dtype * restrict x,
                                                 const dtype * restrict y,
                                                 const dtype * restrict z,
                                                 dtype       * restrict ax,
                                                 dtype       * restrict ay,
                                                 dtype       * restrict az)
    {
      const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const dtype  eps2   = eps * eps;

      // 1. Iterate over i-blocks
      for (size_t b_i = 0; b_i < blocks; b_i++)
      {
        const size_t i_start = b_i * BLOCK_SIZE;
        const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
        const size_t i_count = i_end - i_start;

        // Local accumulators for the current i-block to prevent main memory thrashing
        dtype block_ax[BLOCK_SIZE] = {0.0};
        dtype block_ay[BLOCK_SIZE] = {0.0};
        dtype block_az[BLOCK_SIZE] = {0.0};

        // 2. Iterate over j-blocks
        for (size_t b_j = 0; b_j < blocks; b_j++)
        {
          const size_t j_start = b_j * BLOCK_SIZE;
          const size_t j_end   = (j_start + BLOCK_SIZE <= n) ? (j_start + BLOCK_SIZE) : n;

          for (size_t i = i_start; i < i_end; i++)
          {
            const size_t i_rel = i - i_start;
            const dtype xi = x[i];
            const dtype yi = y[i];
            const dtype zi = z[i];

            // 3. Vector-friendly MAU Arrays (Zero Register Spilling)
            dtype axc[2] = {0.0};
            dtype ayc[2] = {0.0};
            dtype azc[2] = {0.0};

            size_t j = j_start;
            for (; j + 1u < j_end; j += 2u)
            {
              // Instructs GCC to vectorize the inner 2 computations
              #pragma GCC unroll 2
              for (size_t u = 0; u < 2u; ++u)
              {
                const dtype dx   = x[j + u] - xi;
                const dtype dy   = y[j + u] - yi;
                const dtype dz   = z[j + u] - zi;
                const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
                const dtype invr = dtype_rsqrt(r2); // This will call your new AVX-512 intrinsic
                const dtype s    = g * mass * invr * invr * invr;

                axc[u] += dx * s;
                ayc[u] += dy * s;
                azc[u] += dz * s;
              }
            }

            // 4. Remainder loop for j-block boundaries that don't divide by 8
            dtype axi = 0.0;
            dtype ayi = 0.0;
            dtype azi = 0.0;

            for (; j < j_end; ++j)
            {
              const dtype dx   = x[j] - xi;
              const dtype dy   = y[j] - yi;
              const dtype dz   = z[j] - zi;
              const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = dtype_rsqrt(r2);
              const dtype s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }

            // 5. Horizontal reduction: sum the 2 parallel accumulators
            #pragma GCC unroll 2
            for (size_t u = 0; u < 2u; ++u)
            {
              axi += axc[u];
              ayi += ayc[u];
              azi += azc[u];
            }

            // Add the result of this j-block into the local i-block buffer
            block_ax[i_rel] += axi;
            block_ay[i_rel] += ayi;
            block_az[i_rel] += azi;
          }
        }

        // 6. Write the completed i-block back to global main memory
        for (size_t i = 0; i < i_count; i++)
        {
          ax[i_start + i] = block_ax[i];
          ay[i_start + i] = block_ay[i];
          az[i_start + i] = block_az[i];
        }
      }
    }

void compute_accelerations_blocks_rsqrt_mau4(const size_t  n,
                                                 const dtype   g,
                                                 const dtype   mass,
                                                 const dtype   eps,
                                                 const dtype * restrict x,
                                                 const dtype * restrict y,
                                                 const dtype * restrict z,
                                                 dtype       * restrict ax,
                                                 dtype       * restrict ay,
                                                 dtype       * restrict az)
    {
      const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const dtype  eps2   = eps * eps;

      // 1. Iterate over i-blocks
      for (size_t b_i = 0; b_i < blocks; b_i++)
      {
        const size_t i_start = b_i * BLOCK_SIZE;
        const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
        const size_t i_count = i_end - i_start;

        // Local accumulators for the current i-block to prevent main memory thrashing
        dtype block_ax[BLOCK_SIZE] = {0.0};
        dtype block_ay[BLOCK_SIZE] = {0.0};
        dtype block_az[BLOCK_SIZE] = {0.0};

        // 2. Iterate over j-blocks
        for (size_t b_j = 0; b_j < blocks; b_j++)
        {
          const size_t j_start = b_j * BLOCK_SIZE;
          const size_t j_end   = (j_start + BLOCK_SIZE <= n) ? (j_start + BLOCK_SIZE) : n;

          for (size_t i = i_start; i < i_end; i++)
          {
            const size_t i_rel = i - i_start;
            const dtype xi = x[i];
            const dtype yi = y[i];
            const dtype zi = z[i];

            // 3. Vector-friendly MAU Arrays (Zero Register Spilling)
            dtype axc[4] = {0.0};
            dtype ayc[4] = {0.0};
            dtype azc[4] = {0.0};

            size_t j = j_start;
            for (; j + 3u < j_end; j += 4u)
            {
              // Instructs GCC to vectorize the inner 4 computations
              #pragma GCC unroll 4
              for (size_t u = 0; u < 4u; ++u)
              {
                const dtype dx   = x[j + u] - xi;
                const dtype dy   = y[j + u] - yi;
                const dtype dz   = z[j + u] - zi;
                const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
                const dtype invr = dtype_rsqrt(r2); // This will call your new AVX-512 intrinsic
                const dtype s    = g * mass * invr * invr * invr;

                axc[u] += dx * s;
                ayc[u] += dy * s;
                azc[u] += dz * s;
              }
            }

            // 4. Remainder loop for j-block boundaries that don't divide by 8
            dtype axi = 0.0;
            dtype ayi = 0.0;
            dtype azi = 0.0;

            for (; j < j_end; ++j)
            {
              const dtype dx   = x[j] - xi;
              const dtype dy   = y[j] - yi;
              const dtype dz   = z[j] - zi;
              const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = dtype_rsqrt(r2);
              const dtype s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }

            // 5. Horizontal reduction: sum the 4 parallel accumulators
            #pragma GCC unroll 4
            for (size_t u = 0; u < 4u; ++u)
            {
              axi += axc[u];
              ayi += ayc[u];
              azi += azc[u];
            }

            // Add the result of this j-block into the local i-block buffer
            block_ax[i_rel] += axi;
            block_ay[i_rel] += ayi;
            block_az[i_rel] += azi;
          }
        }

        // 6. Write the completed i-block back to global main memory
        for (size_t i = 0; i < i_count; i++)
        {
          ax[i_start + i] = block_ax[i];
          ay[i_start + i] = block_ay[i];
          az[i_start + i] = block_az[i];
        }
      }
    }


void compute_accelerations_blocks_rsqrt_mau8(const size_t  n,
                                                 const dtype   g,
                                                 const dtype   mass,
                                                 const dtype   eps,
                                                 const dtype * restrict x,
                                                 const dtype * restrict y,
                                                 const dtype * restrict z,
                                                 dtype       * restrict ax,
                                                 dtype       * restrict ay,
                                                 dtype       * restrict az)
    {
      const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const dtype  eps2   = eps * eps;

      // 1. Iterate over i-blocks
      for (size_t b_i = 0; b_i < blocks; b_i++)
      {
        const size_t i_start = b_i * BLOCK_SIZE;
        const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
        const size_t i_count = i_end - i_start;

        // Local accumulators for the current i-block to prevent main memory thrashing
        dtype block_ax[BLOCK_SIZE] = {0.0};
        dtype block_ay[BLOCK_SIZE] = {0.0};
        dtype block_az[BLOCK_SIZE] = {0.0};

        // 2. Iterate over j-blocks
        for (size_t b_j = 0; b_j < blocks; b_j++)
        {
          const size_t j_start = b_j * BLOCK_SIZE;
          const size_t j_end   = (j_start + BLOCK_SIZE <= n) ? (j_start + BLOCK_SIZE) : n;

          for (size_t i = i_start; i < i_end; i++)
          {
            const size_t i_rel = i - i_start;
            const dtype xi = x[i];
            const dtype yi = y[i];
            const dtype zi = z[i];

            // 3. Vector-friendly MAU Arrays (Zero Register Spilling)
            dtype axc[8] = {0.0};
            dtype ayc[8] = {0.0};
            dtype azc[8] = {0.0};

            size_t j = j_start;
            for (; j + 7u < j_end; j += 8u)
            {
              // Instructs GCC to vectorize the inner 8 computations
              #pragma GCC unroll 8
              for (size_t u = 0; u < 8u; ++u)
              {
                const dtype dx   = x[j + u] - xi;
                const dtype dy   = y[j + u] - yi;
                const dtype dz   = z[j + u] - zi;
                const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
                const dtype invr = dtype_rsqrt(r2); // This will call your new AVX-512 intrinsic
                const dtype s    = g * mass * invr * invr * invr;

                axc[u] += dx * s;
                ayc[u] += dy * s;
                azc[u] += dz * s;
              }
            }

            // 4. Remainder loop for j-block boundaries that don't divide by 8
            dtype axi = 0.0;
            dtype ayi = 0.0;
            dtype azi = 0.0;

            for (; j < j_end; ++j)
            {
              const dtype dx   = x[j] - xi;
              const dtype dy   = y[j] - yi;
              const dtype dz   = z[j] - zi;
              const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = dtype_rsqrt(r2);
              const dtype s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }

            // 5. Horizontal reduction: sum the 8 parallel accumulators
            #pragma GCC unroll 8
            for (size_t u = 0; u < 8u; ++u)
            {
              axi += axc[u];
              ayi += ayc[u];
              azi += azc[u];
            }

            // Add the result of this j-block into the local i-block buffer
            block_ax[i_rel] += axi;
            block_ay[i_rel] += ayi;
            block_az[i_rel] += azi;
          }
        }

        // 6. Write the completed i-block back to global main memory
        for (size_t i = 0; i < i_count; i++)
        {
          ax[i_start + i] = block_ax[i];
          ay[i_start + i] = block_ay[i];
          az[i_start + i] = block_az[i];
        }
      }
    }


void compute_accelerations_blocks_rsqrt_mau16(const size_t  n,
                                                 const dtype   g,
                                                 const dtype   mass,
                                                 const dtype   eps,
                                                 const dtype * restrict x,
                                                 const dtype * restrict y,
                                                 const dtype * restrict z,
                                                 dtype       * restrict ax,
                                                 dtype       * restrict ay,
                                                 dtype       * restrict az)
    {
      const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const dtype  eps2   = eps * eps;

      // 1. Iterate over i-blocks
      for (size_t b_i = 0; b_i < blocks; b_i++)
      {
        const size_t i_start = b_i * BLOCK_SIZE;
        const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
        const size_t i_count = i_end - i_start;

        // Local accumulators for the current i-block to prevent main memory thrashing
        dtype block_ax[BLOCK_SIZE] = {0.0};
        dtype block_ay[BLOCK_SIZE] = {0.0};
        dtype block_az[BLOCK_SIZE] = {0.0};

        // 2. Iterate over j-blocks
        for (size_t b_j = 0; b_j < blocks; b_j++)
        {
          const size_t j_start = b_j * BLOCK_SIZE;
          const size_t j_end   = (j_start + BLOCK_SIZE <= n) ? (j_start + BLOCK_SIZE) : n;

          for (size_t i = i_start; i < i_end; i++)
          {
            const size_t i_rel = i - i_start;
            const dtype xi = x[i];
            const dtype yi = y[i];
            const dtype zi = z[i];

            // 3. Vector-friendly MAU Arrays (Zero Register Spilling)
            dtype axc[16] = {0.0};
            dtype ayc[16] = {0.0};
            dtype azc[16] = {0.0};

            size_t j = j_start;
            for (; j + 15u < j_end; j += 16u)
            {
              // Instructs GCC to vectorize the inner 16 computations
              #pragma GCC unroll 16
              for (size_t u = 0; u < 16u; ++u)
              {
                const dtype dx   = x[j + u] - xi;
                const dtype dy   = y[j + u] - yi;
                const dtype dz   = z[j + u] - zi;
                const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
                const dtype invr = dtype_rsqrt(r2); // This will call your new AVX-512 intrinsic
                const dtype s    = g * mass * invr * invr * invr;

                axc[u] += dx * s;
                ayc[u] += dy * s;
                azc[u] += dz * s;
              }
            }

            // 4. Remainder loop for j-block boundaries that don't divide by 8
            dtype axi = 0.0;
            dtype ayi = 0.0;
            dtype azi = 0.0;

            for (; j < j_end; ++j)
            {
              const dtype dx   = x[j] - xi;
              const dtype dy   = y[j] - yi;
              const dtype dz   = z[j] - zi;
              const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = dtype_rsqrt(r2);
              const dtype s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }

            // 5. Horizontal reduction: sum the 16 parallel accumulators
            #pragma GCC unroll 16
            for (size_t u = 0; u < 16u; ++u)
            {
              axi += axc[u];
              ayi += ayc[u];
              azi += azc[u];
            }

            // Add the result of this j-block into the local i-block buffer
            block_ax[i_rel] += axi;
            block_ay[i_rel] += ayi;
            block_az[i_rel] += azi;
          }
        }

        // 6. Write the completed i-block back to global main memory
        for (size_t i = 0; i < i_count; i++)
        {
          ax[i_start + i] = block_ax[i];
          ay[i_start + i] = block_ay[i];
          az[i_start + i] = block_az[i];
        }
      }
    }


/* =========================================================================
 * 5. DKD Integration and Energy Diagnostics
 * ========================================================================= */

void drift (particles_t *p, dtype dt)
{
  const size_t  n  = p->n;
  dtype  *x  = p->x;
  dtype  *y  = p->y;
  dtype  *z  = p->z;
  const dtype  *vx = p->vx;
  const dtype  *vy = p->vy;
  const dtype  *vz = p->vz;
  size_t  i;

  for (i = 0u; i < n; ++i)
  {
    x[i] += dt * vx[i];
    y[i] += dt * vy[i];
    z[i] += dt * vz[i];
  }
}

void kick (particles_t *p, dtype dt)
{
  size_t   n  = p->n;
  dtype  * vx = p->vx;
  dtype  * vy = p->vy;
  dtype  * vz = p->vz;
  dtype  * ax = p->ax;
  dtype  * ay = p->ay;
  dtype  * az = p->az;
  size_t   i;

  for (i = 0u; i < n; ++i)
  {
    vx[i] += dt * ax[i];
    vy[i] += dt * ay[i];
    vz[i] += dt * az[i];
  }
}

void leapfrog_dkd_step (particles_t   *p,
                        const dtype    g,
                        const dtype  eps,
                        const dtype   dt,
                        profiler_t   *profiler,
                        const size_t profiler_flag,
                        const size_t   step,
                        const kernel_t  compute_accelerations
                       )
{
  double t0 = 0.0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift (p, (dtype) 0.5 * dt);
  if (profiler_flag) profiler->first_drift_time[step] = get_time() - t0;

  // Accelerations
  if (profiler_flag)
  {
    profiler_papi_start(profiler);
    t0 = get_time();
  }
  compute_accelerations (p->n, g, p->mass, eps,
                         p->x, p->y, p->z,
                         p->ax, p->ay, p->az);
  if (profiler_flag)
  {
    profiler->force_time[step] = get_time() - t0;
    profiler_papi_stop(profiler, step);
  }

  // Kick
  if (profiler_flag) t0 = get_time();
  kick (p, dt);
  if (profiler_flag) profiler->kick_time[step] = get_time() - t0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift (p, (dtype) 0.5 * dt);
  if (profiler_flag) profiler->second_drift_time[step] = get_time() - t0;
}

dtype kinetic_energy (const particles_t *p)
{
  size_t        n    = p->n;
  dtype         mass = p->mass;
  long double   sum  = 0.0L;
  size_t        i;

  for (i = 0u; i < n; ++i)
  {
    const long double  vx = (long double) p->vx[i];
    const long double  vy = (long double) p->vy[i];
    const long double  vz = (long double) p->vz[i];

    sum += vx * vx + vy * vy + vz * vz;
  }

  return (dtype) (0.5L * (long double) mass * sum);
}

dtype potential_energy_naive (particles_t *p, dtype g, dtype eps)
{
  size_t        n    = p->n;
  dtype         eps2 = eps * eps;
  dtype         m2   = p->mass * p->mass;
  long double   sum  = 0.0L;
  size_t        i;
  size_t        j;

  for (i = 0u; i < n; ++i)
  {
    dtype  xi = p->x[i];
    dtype  yi = p->y[i];
    dtype  zi = p->z[i];

    for (j = i + 1u; j < n; ++j)
    {
      dtype  dx   = p->x[j] - xi;
      dtype  dy   = p->y[j] - yi;
      dtype  dz   = p->z[j] - zi;
      dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
      dtype  invr = (dtype) 1.0 / dtype_sqrt (r2);

      sum -= (long double) g * (long double) m2 * (long double) invr;
    }
  }

  return (dtype) sum;
}

dtype total_energy (particles_t *p, dtype g, dtype eps, dtype *kinetic, dtype *potential)
{
  *kinetic   = kinetic_energy (p);
  *potential = potential_energy_naive (p, g, eps);

  return *kinetic + *potential;
}