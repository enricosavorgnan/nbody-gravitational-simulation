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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0;
    dtype ay0 = 0.0, ay1 = 0.0;
    dtype az0 = 0.0, az1 = 0.0;

    size_t j = 0u;
    for (; j + 1u < n; j += 2u)
    {
      const dtype dx0   = x[j + 0u] - xi;
      const dtype dy0   = y[j + 0u] - yi;
      const dtype dz0   = z[j + 0u] - zi;
      const dtype r2_0  = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
      const dtype invr0 = (dtype) 1.0 / dtype_sqrt (r2_0);
      const dtype s0    = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

      const dtype dx1   = x[j + 1u] - xi;
      const dtype dy1   = y[j + 1u] - yi;
      const dtype dz1   = z[j + 1u] - zi;
      const dtype r2_1  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
      const dtype invr1 = (dtype) 1.0 / dtype_sqrt (r2_1);
      const dtype s1    = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = ax0 + ax1;
    ay[i] = ay0 + ay1;
    az[i] = az0 + az1;
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
    dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
    dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;

    size_t j = 0u;
    for (; j + 3u < n; j += 4u)
    {
      const dtype dx0   = x[j + 0u] - xi;
      const dtype dy0   = y[j + 0u] - yi;
      const dtype dz0   = z[j + 0u] - zi;
      const dtype r2_0  = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
      const dtype invr0 = (dtype) 1.0 / dtype_sqrt (r2_0);
      const dtype s0    = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

      const dtype dx1   = x[j + 1u] - xi;
      const dtype dy1   = y[j + 1u] - yi;
      const dtype dz1   = z[j + 1u] - zi;
      const dtype r2_1  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
      const dtype invr1 = (dtype) 1.0 / dtype_sqrt (r2_1);
      const dtype s1    = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;

      const dtype dx2   = x[j + 2u] - xi;
      const dtype dy2   = y[j + 2u] - yi;
      const dtype dz2   = z[j + 2u] - zi;
      const dtype r2_2  = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
      const dtype invr2 = (dtype) 1.0 / dtype_sqrt (r2_2);
      const dtype s2    = g * mass * invr2 * invr2 * invr2;
      ax2 += dx2 * s2; ay2 += dy2 * s2; az2 += dz2 * s2;

      const dtype dx3   = x[j + 3u] - xi;
      const dtype dy3   = y[j + 3u] - yi;
      const dtype dz3   = z[j + 3u] - zi;
      const dtype r2_3  = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
      const dtype invr3 = (dtype) 1.0 / dtype_sqrt (r2_3);
      const dtype s3    = g * mass * invr3 * invr3 * invr3;
      ax3 += dx3 * s3; ay3 += dy3 * s3; az3 += dz3 * s3;
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = (ax0 + ax1) + (ax2 + ax3);
    ay[i] = (ay0 + ay1) + (ay2 + ay3);
    az[i] = (az0 + az1) + (az2 + az3);
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
    dtype ax4 = 0.0, ax5 = 0.0, ax6 = 0.0, ax7 = 0.0;
    dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
    dtype ay4 = 0.0, ay5 = 0.0, ay6 = 0.0, ay7 = 0.0;
    dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;
    dtype az4 = 0.0, az5 = 0.0, az6 = 0.0, az7 = 0.0;

    size_t j = 0u;
    for (; j + 7u < n; j += 8u)
    {
      const dtype dx0 = x[j+0] - xi; const dtype dy0 = y[j+0] - yi; const dtype dz0 = z[j+0] - zi;
      const dtype r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0 + eps2;
      const dtype invr0 = (dtype) 1.0 / dtype_sqrt(r2_0);
      const dtype s0 = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

      const dtype dx1 = x[j+1] - xi; const dtype dy1 = y[j+1] - yi; const dtype dz1 = z[j+1] - zi;
      const dtype r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1 + eps2;
      const dtype invr1 = (dtype) 1.0 / dtype_sqrt(r2_1);
      const dtype s1 = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;

      const dtype dx2 = x[j+2] - xi; const dtype dy2 = y[j+2] - yi; const dtype dz2 = z[j+2] - zi;
      const dtype r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2 + eps2;
      const dtype invr2 = (dtype) 1.0 / dtype_sqrt(r2_2);
      const dtype s2 = g * mass * invr2 * invr2 * invr2;
      ax2 += dx2 * s2; ay2 += dy2 * s2; az2 += dz2 * s2;

      const dtype dx3 = x[j+3] - xi; const dtype dy3 = y[j+3] - yi; const dtype dz3 = z[j+3] - zi;
      const dtype r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3 + eps2;
      const dtype invr3 = (dtype) 1.0 / dtype_sqrt(r2_3);
      const dtype s3 = g * mass * invr3 * invr3 * invr3;
      ax3 += dx3 * s3; ay3 += dy3 * s3; az3 += dz3 * s3;

      const dtype dx4 = x[j+4] - xi; const dtype dy4 = y[j+4] - yi; const dtype dz4 = z[j+4] - zi;
      const dtype r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4 + eps2;
      const dtype invr4 = (dtype) 1.0 / dtype_sqrt(r2_4);
      const dtype s4 = g * mass * invr4 * invr4 * invr4;
      ax4 += dx4 * s4; ay4 += dy4 * s4; az4 += dz4 * s4;

      const dtype dx5 = x[j+5] - xi; const dtype dy5 = y[j+5] - yi; const dtype dz5 = z[j+5] - zi;
      const dtype r2_5 = dx5*dx5 + dy5*dy5 + dz5*dz5 + eps2;
      const dtype invr5 = (dtype) 1.0 / dtype_sqrt(r2_5);
      const dtype s5 = g * mass * invr5 * invr5 * invr5;
      ax5 += dx5 * s5; ay5 += dy5 * s5; az5 += dz5 * s5;

      const dtype dx6 = x[j+6] - xi; const dtype dy6 = y[j+6] - yi; const dtype dz6 = z[j+6] - zi;
      const dtype r2_6 = dx6*dx6 + dy6*dy6 + dz6*dz6 + eps2;
      const dtype invr6 = (dtype) 1.0 / dtype_sqrt(r2_6);
      const dtype s6 = g * mass * invr6 * invr6 * invr6;
      ax6 += dx6 * s6; ay6 += dy6 * s6; az6 += dz6 * s6;

      const dtype dx7 = x[j+7] - xi; const dtype dy7 = y[j+7] - yi; const dtype dz7 = z[j+7] - zi;
      const dtype r2_7 = dx7*dx7 + dy7*dy7 + dz7*dz7 + eps2;
      const dtype invr7 = (dtype) 1.0 / dtype_sqrt(r2_7);
      const dtype s7 = g * mass * invr7 * invr7 * invr7;
      ax7 += dx7 * s7; ay7 += dy7 * s7; az7 += dz7 * s7;
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = ((ax0 + ax1) + (ax2 + ax3)) + ((ax4 + ax5) + (ax6 + ax7));
    ay[i] = ((ay0 + ay1) + (ay2 + ay3)) + ((ay4 + ay5) + (ay6 + ay7));
    az[i] = ((az0 + az1) + (az2 + az3)) + ((az4 + az5) + (az6 + az7));
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
    dtype ax4 = 0.0, ax5 = 0.0, ax6 = 0.0, ax7 = 0.0;
    dtype ax8 = 0.0, ax9 = 0.0, ax10 = 0.0, ax11 = 0.0;
    dtype ax12 = 0.0, ax13 = 0.0, ax14 = 0.0, ax15 = 0.0;

    dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
    dtype ay4 = 0.0, ay5 = 0.0, ay6 = 0.0, ay7 = 0.0;
    dtype ay8 = 0.0, ay9 = 0.0, ay10 = 0.0, ay11 = 0.0;
    dtype ay12 = 0.0, ay13 = 0.0, ay14 = 0.0, ay15 = 0.0;

    dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;
    dtype az4 = 0.0, az5 = 0.0, az6 = 0.0, az7 = 0.0;
    dtype az8 = 0.0, az9 = 0.0, az10 = 0.0, az11 = 0.0;
    dtype az12 = 0.0, az13 = 0.0, az14 = 0.0, az15 = 0.0;

    size_t j = 0u;
    for (; j + 15u < n; j += 16u)
    {
#define INTERACTION_STEP_MAU(k) \
      do { \
        const dtype dx##k   = x[j + (k)] - xi; \
        const dtype dy##k   = y[j + (k)] - yi; \
        const dtype dz##k   = z[j + (k)] - zi; \
        const dtype r2_##k  = dx##k * dx##k + dy##k * dy##k + dz##k * dz##k + eps2; \
        const dtype invr##k = (dtype) 1.0 / dtype_sqrt (r2_##k); \
        const dtype s##k    = g * mass * invr##k * invr##k * invr##k; \
        ax##k += dx##k * s##k; ay##k += dy##k * s##k; az##k += dz##k * s##k; \
      } while (0)

      INTERACTION_STEP_MAU(0);
      INTERACTION_STEP_MAU(1);
      INTERACTION_STEP_MAU(2);
      INTERACTION_STEP_MAU(3);
      INTERACTION_STEP_MAU(4);
      INTERACTION_STEP_MAU(5);
      INTERACTION_STEP_MAU(6);
      INTERACTION_STEP_MAU(7);
      INTERACTION_STEP_MAU(8);
      INTERACTION_STEP_MAU(9);
      INTERACTION_STEP_MAU(10);
      INTERACTION_STEP_MAU(11);
      INTERACTION_STEP_MAU(12);
      INTERACTION_STEP_MAU(13);
      INTERACTION_STEP_MAU(14);
      INTERACTION_STEP_MAU(15);
#undef INTERACTION_STEP_MAU
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = (((ax0 + ax1) + (ax2 + ax3)) + ((ax4 + ax5) + (ax6 + ax7))) +
            (((ax8 + ax9) + (ax10 + ax11)) + ((ax12 + ax13) + (ax14 + ax15)));
    ay[i] = (((ay0 + ay1) + (ay2 + ay3)) + ((ay4 + ay5) + (ay6 + ay7))) +
            (((ay8 + ay9) + (ay10 + ay11)) + ((ay12 + ay13) + (ay14 + ay15)));
    az[i] = (((az0 + az1) + (az2 + az3)) + ((az4 + az5) + (az6 + az7))) +
            (((az8 + az9) + (az10 + az11)) + ((az12 + az13) + (az14 + az15)));
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0;
    dtype ay0 = 0.0, ay1 = 0.0;
    dtype az0 = 0.0, az1 = 0.0;

    size_t j = 0u;
    for (; j + 1u < n; j += 2u)
    {
      const dtype dx0   = x[j + 0u] - xi;
      const dtype dy0   = y[j + 0u] - yi;
      const dtype dz0   = z[j + 0u] - zi;
      const dtype r2_0  = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
      const dtype invr0 = dtype_rsqrt (r2_0);
      const dtype s0    = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

      const dtype dx1   = x[j + 1u] - xi;
      const dtype dy1   = y[j + 1u] - yi;
      const dtype dz1   = z[j + 1u] - zi;
      const dtype r2_1  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
      const dtype invr1 = dtype_rsqrt (r2_1);
      const dtype s1    = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = ax0 + ax1;
    ay[i] = ay0 + ay1;
    az[i] = az0 + az1;
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
    dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
    dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;

    size_t j = 0u;
    for (; j + 3u < n; j += 4u)
    {
      const dtype dx0   = x[j + 0u] - xi;
      const dtype dy0   = y[j + 0u] - yi;
      const dtype dz0   = z[j + 0u] - zi;
      const dtype r2_0  = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
      const dtype invr0 = dtype_rsqrt (r2_0);
      const dtype s0    = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

      const dtype dx1   = x[j + 1u] - xi;
      const dtype dy1   = y[j + 1u] - yi;
      const dtype dz1   = z[j + 1u] - zi;
      const dtype r2_1  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
      const dtype invr1 = dtype_rsqrt (r2_1);
      const dtype s1    = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;

      const dtype dx2   = x[j + 2u] - xi;
      const dtype dy2   = y[j + 2u] - yi;
      const dtype dz2   = z[j + 2u] - zi;
      const dtype r2_2  = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
      const dtype invr2 = dtype_rsqrt (r2_2);
      const dtype s2    = g * mass * invr2 * invr2 * invr2;
      ax2 += dx2 * s2; ay2 += dy2 * s2; az2 += dz2 * s2;

      const dtype dx3   = x[j + 3u] - xi;
      const dtype dy3   = y[j + 3u] - yi;
      const dtype dz3   = z[j + 3u] - zi;
      const dtype r2_3  = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
      const dtype invr3 = dtype_rsqrt (r2_3);
      const dtype s3    = g * mass * invr3 * invr3 * invr3;
      ax3 += dx3 * s3; ay3 += dy3 * s3; az3 += dz3 * s3;
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = (ax0 + ax1) + (ax2 + ax3);
    ay[i] = (ay0 + ay1) + (ay2 + ay3);
    az[i] = (az0 + az1) + (az2 + az3);
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
    dtype ax4 = 0.0, ax5 = 0.0, ax6 = 0.0, ax7 = 0.0;
    dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
    dtype ay4 = 0.0, ay5 = 0.0, ay6 = 0.0, ay7 = 0.0;
    dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;
    dtype az4 = 0.0, az5 = 0.0, az6 = 0.0, az7 = 0.0;

    size_t j = 0u;
    for (; j + 7u < n; j += 8u)
    {
      const dtype dx0 = x[j+0] - xi; const dtype dy0 = y[j+0] - yi; const dtype dz0 = z[j+0] - zi;
      const dtype r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0 + eps2;
      const dtype invr0 = dtype_rsqrt(r2_0);
      const dtype s0 = g * mass * invr0 * invr0 * invr0;
      ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

      const dtype dx1 = x[j+1] - xi; const dtype dy1 = y[j+1] - yi; const dtype dz1 = z[j+1] - zi;
      const dtype r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1 + eps2;
      const dtype invr1 = dtype_rsqrt(r2_1);
      const dtype s1 = g * mass * invr1 * invr1 * invr1;
      ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;

      const dtype dx2 = x[j+2] - xi; const dtype dy2 = y[j+2] - yi; const dtype dz2 = z[j+2] - zi;
      const dtype r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2 + eps2;
      const dtype invr2 = dtype_rsqrt(r2_2);
      const dtype s2 = g * mass * invr2 * invr2 * invr2;
      ax2 += dx2 * s2; ay2 += dy2 * s2; az2 += dz2 * s2;

      const dtype dx3 = x[j+3] - xi; const dtype dy3 = y[j+3] - yi; const dtype dz3 = z[j+3] - zi;
      const dtype r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3 + eps2;
      const dtype invr3 = dtype_rsqrt(r2_3);
      const dtype s3 = g * mass * invr3 * invr3 * invr3;
      ax3 += dx3 * s3; ay3 += dy3 * s3; az3 += dz3 * s3;

      const dtype dx4 = x[j+4] - xi; const dtype dy4 = y[j+4] - yi; const dtype dz4 = z[j+4] - zi;
      const dtype r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4 + eps2;
      const dtype invr4 = dtype_rsqrt(r2_4);
      const dtype s4 = g * mass * invr4 * invr4 * invr4;
      ax4 += dx4 * s4; ay4 += dy4 * s4; az4 += dz4 * s4;

      const dtype dx5 = x[j+5] - xi; const dtype dy5 = y[j+5] - yi; const dtype dz5 = z[j+5] - zi;
      const dtype r2_5 = dx5*dx5 + dy5*dy5 + dz5*dz5 + eps2;
      const dtype invr5 = dtype_rsqrt(r2_5);
      const dtype s5 = g * mass * invr5 * invr5 * invr5;
      ax5 += dx5 * s5; ay5 += dy5 * s5; az5 += dz5 * s5;

      const dtype dx6 = x[j+6] - xi; const dtype dy6 = y[j+6] - yi; const dtype dz6 = z[j+6] - zi;
      const dtype r2_6 = dx6*dx6 + dy6*dy6 + dz6*dz6 + eps2;
      const dtype invr6 = dtype_rsqrt(r2_6);
      const dtype s6 = g * mass * invr6 * invr6 * invr6;
      ax6 += dx6 * s6; ay6 += dy6 * s6; az6 += dz6 * s6;

      const dtype dx7 = x[j+7] - xi; const dtype dy7 = y[j+7] - yi; const dtype dz7 = z[j+7] - zi;
      const dtype r2_7 = dx7*dx7 + dy7*dy7 + dz7*dz7 + eps2;
      const dtype invr7 = dtype_rsqrt(r2_7);
      const dtype s7 = g * mass * invr7 * invr7 * invr7;
      ax7 += dx7 * s7; ay7 += dy7 * s7; az7 += dz7 * s7;
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = ((ax0 + ax1) + (ax2 + ax3)) + ((ax4 + ax5) + (ax6 + ax7));
    ay[i] = ((ay0 + ay1) + (ay2 + ay3)) + ((ay4 + ay5) + (ay6 + ay7));
    az[i] = ((az0 + az1) + (az2 + az3)) + ((az4 + az5) + (az6 + az7));
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

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
    dtype ax4 = 0.0, ax5 = 0.0, ax6 = 0.0, ax7 = 0.0;
    dtype ax8 = 0.0, ax9 = 0.0, ax10 = 0.0, ax11 = 0.0;
    dtype ax12 = 0.0, ax13 = 0.0, ax14 = 0.0, ax15 = 0.0;

    dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
    dtype ay4 = 0.0, ay5 = 0.0, ay6 = 0.0, ay7 = 0.0;
    dtype ay8 = 0.0, ay9 = 0.0, ay10 = 0.0, ay11 = 0.0;
    dtype ay12 = 0.0, ay13 = 0.0, ay14 = 0.0, ay15 = 0.0;

    dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;
    dtype az4 = 0.0, az5 = 0.0, az6 = 0.0, az7 = 0.0;
    dtype az8 = 0.0, az9 = 0.0, az10 = 0.0, az11 = 0.0;
    dtype az12 = 0.0, az13 = 0.0, az14 = 0.0, az15 = 0.0;

    size_t j = 0u;
    for (; j + 15u < n; j += 16u)
    {
#define INTERACTION_STEP_RSQRT_MAU(k) \
      do { \
        const dtype dx##k   = x[j + (k)] - xi; \
        const dtype dy##k   = y[j + (k)] - yi; \
        const dtype dz##k   = z[j + (k)] - zi; \
        const dtype r2_##k  = dx##k * dx##k + dy##k * dy##k + dz##k * dz##k + eps2; \
        const dtype invr##k = dtype_rsqrt (r2_##k); \
        const dtype s##k    = g * mass * invr##k * invr##k * invr##k; \
        ax##k += dx##k * s##k; ay##k += dy##k * s##k; az##k += dz##k * s##k; \
      } while (0)

      INTERACTION_STEP_RSQRT_MAU(0);
      INTERACTION_STEP_RSQRT_MAU(1);
      INTERACTION_STEP_RSQRT_MAU(2);
      INTERACTION_STEP_RSQRT_MAU(3);
      INTERACTION_STEP_RSQRT_MAU(4);
      INTERACTION_STEP_RSQRT_MAU(5);
      INTERACTION_STEP_RSQRT_MAU(6);
      INTERACTION_STEP_RSQRT_MAU(7);
      INTERACTION_STEP_RSQRT_MAU(8);
      INTERACTION_STEP_RSQRT_MAU(9);
      INTERACTION_STEP_RSQRT_MAU(10);
      INTERACTION_STEP_RSQRT_MAU(11);
      INTERACTION_STEP_RSQRT_MAU(12);
      INTERACTION_STEP_RSQRT_MAU(13);
      INTERACTION_STEP_RSQRT_MAU(14);
      INTERACTION_STEP_RSQRT_MAU(15);
#undef INTERACTION_STEP_RSQRT_MAU
    }

    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;
      ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
    }

    ax[i] = (((ax0 + ax1) + (ax2 + ax3)) + ((ax4 + ax5) + (ax6 + ax7))) +
            (((ax8 + ax9) + (ax10 + ax11)) + ((ax12 + ax13) + (ax14 + ax15)));
    ay[i] = (((ay0 + ay1) + (ay2 + ay3)) + ((ay4 + ay5) + (ay6 + ay7))) +
            (((ay8 + ay9) + (ay10 + ay11)) + ((ay12 + ay13) + (ay14 + ay15)));
    az[i] = (((az0 + az1) + (az2 + az3)) + ((az4 + az5) + (az6 + az7))) +
            (((az8 + az9) + (az10 + az11)) + ((az12 + az13) + (az14 + az15)));
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

  for (size_t b_i = 0; b_i < blocks; b_i++)
  {
    const size_t i_start = b_i * BLOCK_SIZE;
    const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
    const size_t i_count = i_end - i_start;

    dtype block_ax[BLOCK_SIZE];
    dtype block_ay[BLOCK_SIZE];
    dtype block_az[BLOCK_SIZE];

    for (size_t i = 0; i < i_count; i++)
    {
      block_ax[i] = 0.0;
      block_ay[i] = 0.0;
      block_az[i] = 0.0;
    }

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

        dtype ax0 = 0.0, ax1 = 0.0;
        dtype ay0 = 0.0, ay1 = 0.0;
        dtype az0 = 0.0, az1 = 0.0;

        size_t j = j_start;
        for (; j + 1u < j_end; j += 2u)
        {
          const dtype dx0   = x[j + 0u] - xi;
          const dtype dy0   = y[j + 0u] - yi;
          const dtype dz0   = z[j + 0u] - zi;
          const dtype r2_0  = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
          const dtype invr0 = dtype_rsqrt (r2_0);
          const dtype s0    = g * mass * invr0 * invr0 * invr0;
          ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

          const dtype dx1   = x[j + 1u] - xi;
          const dtype dy1   = y[j + 1u] - yi;
          const dtype dz1   = z[j + 1u] - zi;
          const dtype r2_1  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
          const dtype invr1 = dtype_rsqrt (r2_1);
          const dtype s1    = g * mass * invr1 * invr1 * invr1;
          ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;
        }

        for (; j < j_end; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt (r2);
          const dtype s    = g * mass * invr * invr * invr;
          ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
        }

        block_ax[i_rel] += ax0 + ax1;
        block_ay[i_rel] += ay0 + ay1;
        block_az[i_rel] += az0 + az1;
      }
    }

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

  for (size_t b_i = 0; b_i < blocks; b_i++)
  {
    const size_t i_start = b_i * BLOCK_SIZE;
    const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
    const size_t i_count = i_end - i_start;

    dtype block_ax[BLOCK_SIZE];
    dtype block_ay[BLOCK_SIZE];
    dtype block_az[BLOCK_SIZE];

    for (size_t i = 0; i < i_count; i++)
    {
      block_ax[i] = 0.0;
      block_ay[i] = 0.0;
      block_az[i] = 0.0;
    }

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

        dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
        dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
        dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;

        size_t j = j_start;
        for (; j + 3u < j_end; j += 4u)
        {
          const dtype dx0   = x[j + 0u] - xi;
          const dtype dy0   = y[j + 0u] - yi;
          const dtype dz0   = z[j + 0u] - zi;
          const dtype r2_0  = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
          const dtype invr0 = dtype_rsqrt (r2_0);
          const dtype s0    = g * mass * invr0 * invr0 * invr0;
          ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

          const dtype dx1   = x[j + 1u] - xi;
          const dtype dy1   = y[j + 1u] - yi;
          const dtype dz1   = z[j + 1u] - zi;
          const dtype r2_1  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
          const dtype invr1 = dtype_rsqrt (r2_1);
          const dtype s1    = g * mass * invr1 * invr1 * invr1;
          ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;

          const dtype dx2   = x[j + 2u] - xi;
          const dtype dy2   = y[j + 2u] - yi;
          const dtype dz2   = z[j + 2u] - zi;
          const dtype r2_2  = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps2;
          const dtype invr2 = dtype_rsqrt (r2_2);
          const dtype s2    = g * mass * invr2 * invr2 * invr2;
          ax2 += dx2 * s2; ay2 += dy2 * s2; az2 += dz2 * s2;

          const dtype dx3   = x[j + 3u] - xi;
          const dtype dy3   = y[j + 3u] - yi;
          const dtype dz3   = z[j + 3u] - zi;
          const dtype r2_3  = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps2;
          const dtype invr3 = dtype_rsqrt (r2_3);
          const dtype s3    = g * mass * invr3 * invr3 * invr3;
          ax3 += dx3 * s3; ay3 += dy3 * s3; az3 += dz3 * s3;
        }

        for (; j < j_end; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt (r2);
          const dtype s    = g * mass * invr * invr * invr;
          ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
        }

        block_ax[i_rel] += (ax0 + ax1) + (ax2 + ax3);
        block_ay[i_rel] += (ay0 + ay1) + (ay2 + ay3);
        block_az[i_rel] += (az0 + az1) + (az2 + az3);
      }
    }

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

  for (size_t b_i = 0; b_i < blocks; b_i++)
  {
    const size_t i_start = b_i * BLOCK_SIZE;
    const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
    const size_t i_count = i_end - i_start;

    dtype block_ax[BLOCK_SIZE];
    dtype block_ay[BLOCK_SIZE];
    dtype block_az[BLOCK_SIZE];

    for (size_t i = 0; i < i_count; i++)
    {
      block_ax[i] = 0.0;
      block_ay[i] = 0.0;
      block_az[i] = 0.0;
    }

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

        dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
        dtype ax4 = 0.0, ax5 = 0.0, ax6 = 0.0, ax7 = 0.0;
        dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
        dtype ay4 = 0.0, ay5 = 0.0, ay6 = 0.0, ay7 = 0.0;
        dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;
        dtype az4 = 0.0, az5 = 0.0, az6 = 0.0, az7 = 0.0;

        size_t j = j_start;
        for (; j + 7u < j_end; j += 8u)
        {
          const dtype dx0 = x[j+0] - xi; const dtype dy0 = y[j+0] - yi; const dtype dz0 = z[j+0] - zi;
          const dtype r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0 + eps2;
          const dtype invr0 = dtype_rsqrt(r2_0);
          const dtype s0 = g * mass * invr0 * invr0 * invr0;
          ax0 += dx0 * s0; ay0 += dy0 * s0; az0 += dz0 * s0;

          const dtype dx1 = x[j+1] - xi; const dtype dy1 = y[j+1] - yi; const dtype dz1 = z[j+1] - zi;
          const dtype r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1 + eps2;
          const dtype invr1 = dtype_rsqrt(r2_1);
          const dtype s1 = g * mass * invr1 * invr1 * invr1;
          ax1 += dx1 * s1; ay1 += dy1 * s1; az1 += dz1 * s1;

          const dtype dx2 = x[j+2] - xi; const dtype dy2 = y[j+2] - yi; const dtype dz2 = z[j+2] - zi;
          const dtype r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2 + eps2;
          const dtype invr2 = dtype_rsqrt(r2_2);
          const dtype s2 = g * mass * invr2 * invr2 * invr2;
          ax2 += dx2 * s2; ay2 += dy2 * s2; az2 += dz2 * s2;

          const dtype dx3 = x[j+3] - xi; const dtype dy3 = y[j+3] - yi; const dtype dz3 = z[j+3] - zi;
          const dtype r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3 + eps2;
          const dtype invr3 = dtype_rsqrt(r2_3);
          const dtype s3 = g * mass * invr3 * invr3 * invr3;
          ax3 += dx3 * s3; ay3 += dy3 * s3; az3 += dz3 * s3;

          const dtype dx4 = x[j+4] - xi; const dtype dy4 = y[j+4] - yi; const dtype dz4 = z[j+4] - zi;
          const dtype r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4 + eps2;
          const dtype invr4 = dtype_rsqrt(r2_4);
          const dtype s4 = g * mass * invr4 * invr4 * invr4;
          ax4 += dx4 * s4; ay4 += dy4 * s4; az4 += dz4 * s4;

          const dtype dx5 = x[j+5] - xi; const dtype dy5 = y[j+5] - yi; const dtype dz5 = z[j+5] - zi;
          const dtype r2_5 = dx5*dx5 + dy5*dy5 + dz5*dz5 + eps2;
          const dtype invr5 = dtype_rsqrt(r2_5);
          const dtype s5 = g * mass * invr5 * invr5 * invr5;
          ax5 += dx5 * s5; ay5 += dy5 * s5; az5 += dz5 * s5;

          const dtype dx6 = x[j+6] - xi; const dtype dy6 = y[j+6] - yi; const dtype dz6 = z[j+6] - zi;
          const dtype r2_6 = dx6*dx6 + dy6*dy6 + dz6*dz6 + eps2;
          const dtype invr6 = dtype_rsqrt(r2_6);
          const dtype s6 = g * mass * invr6 * invr6 * invr6;
          ax6 += dx6 * s6; ay6 += dy6 * s6; az6 += dz6 * s6;

          const dtype dx7 = x[j+7] - xi; const dtype dy7 = y[j+7] - yi; const dtype dz7 = z[j+7] - zi;
          const dtype r2_7 = dx7*dx7 + dy7*dy7 + dz7*dz7 + eps2;
          const dtype invr7 = dtype_rsqrt(r2_7);
          const dtype s7 = g * mass * invr7 * invr7 * invr7;
          ax7 += dx7 * s7; ay7 += dy7 * s7; az7 += dz7 * s7;
        }

        for (; j < j_end; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt (r2);
          const dtype s    = g * mass * invr * invr * invr;
          ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
        }

        block_ax[i_rel] += ((ax0 + ax1) + (ax2 + ax3)) + ((ax4 + ax5) + (ax6 + ax7));
        block_ay[i_rel] += ((ay0 + ay1) + (ay2 + ay3)) + ((ay4 + ay5) + (ay6 + ay7));
        block_az[i_rel] += ((az0 + az1) + (az2 + az3)) + ((az4 + az5) + (az6 + az7));
      }
    }

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

  for (size_t b_i = 0; b_i < blocks; b_i++)
  {
    const size_t i_start = b_i * BLOCK_SIZE;
    const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
    const size_t i_count = i_end - i_start;

    dtype block_ax[BLOCK_SIZE];
    dtype block_ay[BLOCK_SIZE];
    dtype block_az[BLOCK_SIZE];

    for (size_t i = 0; i < i_count; i++)
    {
      block_ax[i] = 0.0;
      block_ay[i] = 0.0;
      block_az[i] = 0.0;
    }

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

        dtype ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
        dtype ax4 = 0.0, ax5 = 0.0, ax6 = 0.0, ax7 = 0.0;
        dtype ax8 = 0.0, ax9 = 0.0, ax10 = 0.0, ax11 = 0.0;
        dtype ax12 = 0.0, ax13 = 0.0, ax14 = 0.0, ax15 = 0.0;

        dtype ay0 = 0.0, ay1 = 0.0, ay2 = 0.0, ay3 = 0.0;
        dtype ay4 = 0.0, ay5 = 0.0, ay6 = 0.0, ay7 = 0.0;
        dtype ay8 = 0.0, ay9 = 0.0, ay10 = 0.0, ay11 = 0.0;
        dtype ay12 = 0.0, ay13 = 0.0, ay14 = 0.0, ay15 = 0.0;

        dtype az0 = 0.0, az1 = 0.0, az2 = 0.0, az3 = 0.0;
        dtype az4 = 0.0, az5 = 0.0, az6 = 0.0, az7 = 0.0;
        dtype az8 = 0.0, az9 = 0.0, az10 = 0.0, az11 = 0.0;
        dtype az12 = 0.0, az13 = 0.0, az14 = 0.0, az15 = 0.0;

        size_t j = j_start;
        for (; j + 15u < j_end; j += 16u)
        {
#define INTERACTION_STEP_BLOCK_RSQRT_MAU(k) \
          do { \
            const dtype dx##k   = x[j + (k)] - xi; \
            const dtype dy##k   = y[j + (k)] - yi; \
            const dtype dz##k   = z[j + (k)] - zi; \
            const dtype r2_##k  = dx##k * dx##k + dy##k * dy##k + dz##k * dz##k + eps2; \
            const dtype invr##k = dtype_rsqrt (r2_##k); \
            const dtype s##k    = g * mass * invr##k * invr##k * invr##k; \
            ax##k += dx##k * s##k; ay##k += dy##k * s##k; az##k += dz##k * s##k; \
          } while (0)

          INTERACTION_STEP_BLOCK_RSQRT_MAU(0);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(1);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(2);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(3);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(4);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(5);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(6);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(7);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(8);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(9);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(10);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(11);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(12);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(13);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(14);
          INTERACTION_STEP_BLOCK_RSQRT_MAU(15);
#undef INTERACTION_STEP_BLOCK_RSQRT_MAU
        }

        for (; j < j_end; ++j)
        {
          const dtype dx   = x[j] - xi;
          const dtype dy   = y[j] - yi;
          const dtype dz   = z[j] - zi;
          const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
          const dtype invr = dtype_rsqrt (r2);
          const dtype s    = g * mass * invr * invr * invr;
          ax0 += dx * s; ay0 += dy * s; az0 += dz * s;
        }

        block_ax[i_rel] += (((ax0 + ax1) + (ax2 + ax3)) + ((ax4 + ax5) + (ax6 + ax7))) +
                           (((ax8 + ax9) + (ax10 + ax11)) + ((ax12 + ax13) + (ax14 + ax15)));
        block_ay[i_rel] += (((ay0 + ay1) + (ay2 + ay3)) + ((ay4 + ay5) + (ay6 + ay7))) +
                           (((ay8 + ay9) + (ay10 + ay11)) + ((ay12 + ay13) + (ay14 + ay15)));
        block_az[i_rel] += (((az0 + az1) + (az2 + az3)) + ((az4 + az5) + (az6 + az7))) +
                           (((az8 + az9) + (az10 + az11)) + ((az12 + az13) + (az14 + az15)));
      }
    }

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