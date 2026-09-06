// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif


void compute_accelerations_naive (const size_t  n,          // number of particles
                                  const dtype   g,          // gravitational constant
                                  const dtype   mass,       // mass of every source particle
                                  const dtype   eps,        // Plummer softening length
                                  const dtype * restrict x,          // x positions, read-only
                                  const dtype * restrict y,          // y positions, read-only
                                  const dtype * restrict z,          // z positions, read-only
                                  dtype * restrict ax,               // x acceleration, overwritten
                                  dtype * restrict ay,               // y acceleration, overwritten
                                  dtype * restrict az                // z acceleration, overwritten
					 )
{
  const dtype  eps2 = eps * eps;
  size_t       i;
  size_t       j;

  for (i = 0u; i < n; ++i)
    {
      const dtype  xi  = x[i];
      const dtype  yi  = y[i];
      const dtype  zi  = z[i];
      dtype        axi = 0.0;
      dtype        ayi = 0.0;
      dtype        azi = 0.0;

      for (j = 0u; j < n; ++j)
        {
          if (j != i)
            {
              const dtype  dx   = x[j] - xi;
              const dtype  dy   = y[j] - yi;
              const dtype  dz   = z[j] - zi;
              const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype  invr = 1.0 / dtype_sqrt (r2);
              const dtype  s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }
        }

      ax[i] = axi;
      ay[i] = ayi;
      az[i] = azi;
    }
}


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
  const size_t n2  = n & ~(size_t) 1u; // Round down to multiple of 2

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[2] = { 0 };
    dtype ayc[2] = { 0 };
    dtype azc[2] = { 0 };

    size_t j = 0u;
    for (; j < n2; j += 2u)
    {
      #pragma GCC unroll 2
      for (size_t u = 0u; u < 2u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (just 1 single operation in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }
    ax[i] = axc[0] + axc[1];
    ay[i] = ayc[0] + ayc[1];
    az[i] = azc[0] + azc[1];
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
  const size_t n4  = n & ~(size_t) 3u; // Round down to multiple of 4

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[4] = { 0 };
    dtype ayc[4] = { 0 };
    dtype azc[4] = { 0 };

    size_t j = 0u;
    for (; j < n4; j += 4u)
    {
      #pragma GCC unroll 4
      for (size_t u = 0u; u < 4u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (max 3 loops in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }

    ax[i] = ((axc[0] + axc[1]) + (axc[2] + axc[3]));
    ay[i] = ((ayc[0] + ayc[1]) + (ayc[2] + ayc[3]));
    az[i] = ((azc[0] + azc[1]) + (azc[2] + azc[3]));
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
  const size_t n8  = n & ~(size_t) 7u; // Round down to multiple of 8

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[8] = { 0 };
    dtype ayc[8] = { 0 };
    dtype azc[8] = { 0 };

    size_t j = 0u;
    for (; j < n8; j += 8u)
    {
#pragma GCC unroll 8
      for (size_t u = 0u; u < 8u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (max 7 loops in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }
    ax[i] = ((axc[0] + axc[1]) + (axc[2] + axc[3])) + ((axc[4] + axc[5]) + (axc[6] + axc[7]));
    ay[i] = ((ayc[0] + ayc[1]) + (ayc[2] + ayc[3])) + ((ayc[4] + ayc[5]) + (ayc[6] + ayc[7]));
    az[i] = ((azc[0] + azc[1]) + (azc[2] + azc[3])) + ((azc[4] + azc[5]) + (azc[6] + azc[7]));
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
  const size_t n16  = n & ~(size_t) 15u; // Round down to multiple of 2

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[16] = { 0 };
    dtype ayc[16] = { 0 };
    dtype azc[16] = { 0 };

    size_t j = 0u;
    for (; j < n16; j += 16u)
    {
#pragma GCC unroll 16
      for (size_t u = 0u; u < 16u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (max 15 loops in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }
    ax[i] = (((axc[0] + axc[1]) + (axc[2] + axc[3])) + ((axc[4] + axc[5]) + (axc[6] + axc[7])))
          + (((axc[8] + axc[9]) + (axc[10] + axc[11])) + ((axc[12] + axc[13]) + (axc[14] + axc[15])));
    ay[i] = (((ayc[0] + ayc[1]) + (ayc[2] + ayc[3])) + ((ayc[4] + ayc[5]) + (ayc[6] + ayc[7])))
          + (((ayc[8] + ayc[9]) + (ayc[10] + ayc[11])) + ((ayc[12] + ayc[13]) + (ayc[14] + ayc[15])));
    az[i] = (((azc[0] + azc[1]) + (azc[2] + azc[3])) + ((azc[4] + azc[5]) + (azc[6] + azc[7])))
          + (((azc[8] + azc[9]) + (azc[10] + azc[11])) + ((azc[12] + azc[13]) + (azc[14] + azc[15])));
  }
}


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
  const size_t n2  = n & ~(size_t) 1u; // Round down to multiple of 2

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[2] = { 0 };
    dtype ayc[2] = { 0 };
    dtype azc[2] = { 0 };

    size_t j = 0u;
    for (; j < n2; j += 2u)
    {
#pragma GCC unroll 2
      for (size_t u = 0u; u < 2u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = dtype_rsqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (just 1 single operation in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }
    ax[i] = axc[0] + axc[1];
    ay[i] = ayc[0] + ayc[1];
    az[i] = azc[0] + azc[1];
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
  const size_t n4  = n & ~(size_t) 3u; // Round down to multiple of 4

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[4] = { 0 };
    dtype ayc[4] = { 0 };
    dtype azc[4] = { 0 };

    size_t j = 0u;
    for (; j < n4; j += 4u)
    {
#pragma GCC unroll 4
      for (size_t u = 0u; u < 4u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = dtype_rsqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (max 3 loops in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }

    ax[i] = ((axc[0] + axc[1]) + (axc[2] + axc[3]));
    ay[i] = ((ayc[0] + ayc[1]) + (ayc[2] + ayc[3]));
    az[i] = ((azc[0] + azc[1]) + (azc[2] + azc[3]));
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
  const size_t n8  = n & ~(size_t) 7u; // Round down to multiple of 8

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[8] = { 0 };
    dtype ayc[8] = { 0 };
    dtype azc[8] = { 0 };

    size_t j = 0u;
    for (; j < n8; j += 8u)
    {
#pragma GCC unroll 8
      for (size_t u = 0u; u < 8u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = dtype_rsqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (max 7 loops in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }
    ax[i] = ((axc[0] + axc[1]) + (axc[2] + axc[3])) + ((axc[4] + axc[5]) + (axc[6] + axc[7]));
    ay[i] = ((ayc[0] + ayc[1]) + (ayc[2] + ayc[3])) + ((ayc[4] + ayc[5]) + (ayc[6] + ayc[7]));
    az[i] = ((azc[0] + azc[1]) + (azc[2] + azc[3])) + ((azc[4] + azc[5]) + (azc[6] + azc[7]));
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
  const size_t n16  = n & ~(size_t) 15u; // Round down to multiple of 2

  for (size_t i = 0u; i < n; i++)
  {
    const dtype xi = x[i];
    const dtype yi = y[i];
    const dtype zi = z[i];

    dtype axc[16] = { 0 };
    dtype ayc[16] = { 0 };
    dtype azc[16] = { 0 };

    size_t j = 0u;
    for (; j < n16; j += 16u)
    {
#pragma GCC unroll 16
      for (size_t u = 0u; u < 16u; ++u)
      {
        const dtype dx   = x[j + u] - xi;
        const dtype dy   = y[j + u] - yi;
        const dtype dz   = z[j + u] - zi;
        const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
        const dtype invr = dtype_rsqrt (r2);
        const dtype s    = g * mass * invr * invr * invr;

        axc[u] += dx * s;
        ayc[u] += dy * s;
        azc[u] += dz * s;
      }
    }
    // Last loops (max 15 loops in this case)
    for (; j < n; ++j)
    {
      const dtype dx   = x[j] - xi;
      const dtype dy   = y[j] - yi;
      const dtype dz   = z[j] - zi;
      const dtype r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt (r2);
      const dtype s    = g * mass * invr * invr * invr;

      axc[0] += dx * s;
      ayc[0] += dy * s;
      azc[0] += dz * s;
    }
    ax[i] = (((axc[0] + axc[1]) + (axc[2] + axc[3])) + ((axc[4] + axc[5]) + (axc[6] + axc[7])))
          + (((axc[8] + axc[9]) + (axc[10] + axc[11])) + ((axc[12] + axc[13]) + (axc[14] + axc[15])));
    ay[i] = (((ayc[0] + ayc[1]) + (ayc[2] + ayc[3])) + ((ayc[4] + ayc[5]) + (ayc[6] + ayc[7])))
          + (((ayc[8] + ayc[9]) + (ayc[10] + ayc[11])) + ((ayc[12] + ayc[13]) + (ayc[14] + ayc[15])));
    az[i] = (((azc[0] + azc[1]) + (azc[2] + azc[3])) + ((azc[4] + azc[5]) + (azc[6] + azc[7])))
          + (((azc[8] + azc[9]) + (azc[10] + azc[11])) + ((azc[12] + azc[13]) + (azc[14] + azc[15])));
  }
}



void compute_accelerations_blocks_rsqrt_mau2();
void compute_accelerations_blocks_rsqrt_mau4();
void compute_accelerations_blocks_rsqrt_mau8();
void compute_accelerations_blocks_rsqrt_mau16();

/* DKD */

void drift (particles_t *p,       // particle positions are modified in place
                   dtype        dt       // drift interval, often 0.5 * full step
		   )
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

void kick (particles_t *p,       // particle velocities are modified in place
                  dtype        dt       // full kick interval
		  )
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

void leapfrog_dkd_step (particles_t   *p,             // complete particle state, modified in place
                        const dtype    g,             // gravitational constant
                        const dtype  eps,             // softening length
                        const dtype   dt,             // full time-step
                        profiler_t   *profiler,       // optional profiler for per-step timing
                        const size_t profiler_flag,   // whether to use the profiler
                        const size_t   step,          // current step index for profiler
                        const kernel_t  compute_accelerations    // kernel function to compute accelerations

			       )
{
  double t0 = 0.0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift (p, (dtype) 0.5 * dt);
  if (profiler_flag) profiler->first_drift_time[step] = get_time() - t0;

  // Accelerations
  if (profiler_flag) t0 = get_time();
  compute_accelerations (p->n, g, p->mass, eps,
                               p->x, p->y, p->z,
                               p->ax, p->ay, p->az);
  if (profiler_flag) profiler->force_time[step] = get_time() - t0;

  // Kick
  if (profiler_flag) t0 = get_time();
  kick (p, dt);
  if (profiler_flag) profiler->kick_time[step] = get_time() - t0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift (p, (dtype) 0.5 * dt);
  if (profiler_flag) profiler->second_drift_time[step] = get_time() - t0;

}

dtype kinetic_energy (const particles_t *p    // particle velocities are read-only
			     )
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

dtype potential_energy_naive (particles_t *p,        // particle positions are read-only
                                     dtype        g,        // gravitational constant
                                     dtype        eps       // softening length
				     )
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

dtype total_energy (particles_t *p,           // complete particle state, read-only
                           dtype        g,           // gravitational constant
                           dtype        eps,         // softening length
                           dtype       *kinetic,     // output kinetic energy
                           dtype       *potential    // output potential energy
			   )
{
  *kinetic   = kinetic_energy (p);
  *potential = potential_energy_naive (p, g, eps);

  return *kinetic + *potential;
}