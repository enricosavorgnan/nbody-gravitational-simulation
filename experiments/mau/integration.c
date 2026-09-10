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
 * 2. MAU (Multiple Accumulator Units) over the target (i) dimension
 *
 * The accumulator chains are unrolled over i, not over j: K target particles
 * share a single sweep of the j-stream, so
 *
 *   - x[j] / y[j] / z[j] are loaded once per K interactions instead of once
 *     per interaction, and every cache line of the j-stream is consumed by K
 *     targets before it is dropped;
 *   - the K chains are mutually independent, which is the point of MAU;
 *   - the inner j-loop keeps the flat single-body shape that the vectorizer
 *     widens in compute_accelerations_naive -- no #pragma unroll, no inner
 *     u-loop, no split j-remainder.
 *
 * Accumulators are named scalars produced by token pasting, never arrays: an
 * array only reaches a register if SRA and full unrolling both fire, and at
 * K >= 8 they cannot fit anyway (3 * K live accumulators plus 6 * K temporaries
 * against 32 zmm registers), so 8 and 16 are expected to spill to the stack.
 *
 * All four K variants are generated from one body on purpose, so the only
 * difference between MAU 2/4/8/16 is K, and the only difference between the
 * MAU and the Rsqrt-MAU family is the inverse-square-root callback.
 * ========================================================================= */

static inline dtype mau_invsqrt_exact (const dtype r2)
{
  return (dtype) 1.0 / dtype_sqrt (r2);
}

static inline dtype mau_invsqrt_fast (const dtype r2)
{
  return dtype_rsqrt (r2);
}

/* Per-target macros.  Every one of them takes the same (k, INV) argument list
 * -- INV is deliberately unused by some -- so that a single repeater macro can
 * drive all of them. */
#define MAU_LOAD(k, INV)                                                       \
  const dtype xi##k = x[i + k##u];                                             \
  const dtype yi##k = y[i + k##u];                                             \
  const dtype zi##k = z[i + k##u];

#define MAU_ZERO(k, INV)                                                       \
  dtype ax##k = 0.0;                                                           \
  dtype ay##k = 0.0;                                                           \
  dtype az##k = 0.0;

#define MAU_INTERACT(k, INV)                                                   \
  const dtype dx##k = xj - xi##k;                                              \
  const dtype dy##k = yj - yi##k;                                              \
  const dtype dz##k = zj - zi##k;                                              \
  const dtype r2##k = dx##k * dx##k + dy##k * dy##k + dz##k * dz##k + eps2;    \
  const dtype iv##k = INV (r2##k);                                             \
  const dtype s##k  = gm * iv##k * iv##k * iv##k;                              \
  ax##k += dx##k * s##k;                                                       \
  ay##k += dy##k * s##k;                                                       \
  az##k += dz##k * s##k;

#define MAU_STORE(k, INV)                                                      \
  ax[i + k##u] = ax##k;                                                        \
  ay[i + k##u] = ay##k;                                                        \
  az[i + k##u] = az##k;

#define MAU_REPEAT_2(M, INV)   M (0, INV) M (1, INV)

#define MAU_REPEAT_4(M, INV)   MAU_REPEAT_2 (M, INV)                           \
                               M (2, INV) M (3, INV)

#define MAU_REPEAT_8(M, INV)   MAU_REPEAT_4 (M, INV)                           \
                               M (4, INV) M (5, INV) M (6, INV) M (7, INV)

#define MAU_REPEAT_16(M, INV)  MAU_REPEAT_8 (M, INV)                           \
                               M (8, INV)  M (9, INV)  M (10, INV) M (11, INV) \
                               M (12, INV) M (13, INV) M (14, INV) M (15, INV)

/* The shared kernel body.  K is the i-tile width, REPEAT the matching
 * repeater, INV either mau_invsqrt_exact or mau_invsqrt_fast.
 *
 * The j == i self-interaction needs no branch: dx = dy = dz = 0 makes s finite
 * through the Plummer softening and contributes exactly zero, as in the naive
 * kernel. */
#define MAU_BODY(K, REPEAT, INV)                                               \
  const dtype eps2 = eps * eps;                                                \
  const dtype gm   = g * mass;                                                 \
  size_t      i    = 0u;                                                       \
                                                                               \
  for (; i + (K##u - 1u) < n; i += K##u)                                       \
  {                                                                            \
    REPEAT (MAU_LOAD, INV)                                                     \
    REPEAT (MAU_ZERO, INV)                                                     \
                                                                               \
    for (size_t j = 0u; j < n; ++j)                                            \
    {                                                                          \
      const dtype xj = x[j];                                                   \
      const dtype yj = y[j];                                                   \
      const dtype zj = z[j];                                                   \
                                                                               \
      REPEAT (MAU_INTERACT, INV)                                               \
    }                                                                          \
                                                                               \
    REPEAT (MAU_STORE, INV)                                                    \
  }                                                                            \
                                                                               \
  for (; i < n; ++i)                                                           \
  {                                                                            \
    const dtype xi  = x[i];                                                    \
    const dtype yi  = y[i];                                                    \
    const dtype zi  = z[i];                                                    \
    dtype       axi = 0.0;                                                     \
    dtype       ayi = 0.0;                                                     \
    dtype       azi = 0.0;                                                     \
                                                                               \
    for (size_t j = 0u; j < n; ++j)                                            \
    {                                                                          \
      const dtype dx = x[j] - xi;                                              \
      const dtype dy = y[j] - yi;                                              \
      const dtype dz = z[j] - zi;                                              \
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;                     \
      const dtype iv = INV (r2);                                               \
      const dtype s  = gm * iv * iv * iv;                                      \
                                                                               \
      axi += dx * s;                                                           \
      ayi += dy * s;                                                           \
      azi += dz * s;                                                           \
    }                                                                          \
                                                                               \
    ax[i] = axi;                                                               \
    ay[i] = ayi;                                                               \
    az[i] = azi;                                                               \
  }


void compute_accelerations_mau2 (const size_t  n,
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
  MAU_BODY (2, MAU_REPEAT_2, mau_invsqrt_exact)
}


void compute_accelerations_mau4 (const size_t  n,
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
  MAU_BODY (4, MAU_REPEAT_4, mau_invsqrt_exact)
}


void compute_accelerations_mau8 (const size_t  n,
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
  MAU_BODY (8, MAU_REPEAT_8, mau_invsqrt_exact)
}


void compute_accelerations_mau16 (const size_t  n,
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
  MAU_BODY (16, MAU_REPEAT_16, mau_invsqrt_exact)
}


/* =========================================================================
 * 3. Fast RSQRT + MAU over the target (i) dimension
 *
 * Identical to section 2 apart from the inverse-square-root callback.  This is
 * the combination worth measuring: the naive kernel is bound by vsqrtpd /
 * vdivpd throughput, so removing the divide is what exposes the issue width
 * that the i-tiling then feeds.
 * ========================================================================= */
void compute_accelerations_rsqrt_mau2 (const size_t  n,
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
  MAU_BODY (2, MAU_REPEAT_2, mau_invsqrt_fast)
}


void compute_accelerations_rsqrt_mau4 (const size_t  n,
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
  MAU_BODY (4, MAU_REPEAT_4, mau_invsqrt_fast)
}


void compute_accelerations_rsqrt_mau8 (const size_t  n,
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
  MAU_BODY (8, MAU_REPEAT_8, mau_invsqrt_fast)
}


void compute_accelerations_rsqrt_mau16 (const size_t  n,
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
  MAU_BODY (16, MAU_REPEAT_16, mau_invsqrt_fast)
}


#undef MAU_LOAD
#undef MAU_ZERO
#undef MAU_INTERACT
#undef MAU_STORE
#undef MAU_REPEAT_2
#undef MAU_REPEAT_4
#undef MAU_REPEAT_8
#undef MAU_REPEAT_16
#undef MAU_BODY


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