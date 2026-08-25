// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

/* ACCELERATION */

void compute_accelerations_rsqrt_check(const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  const dtype * restrict x_r,
                                  const dtype * restrict y_r,
                                  const dtype * restrict z_r,
                                  dtype * restrict ax_r,
                                  dtype * restrict ay_r,
                                  dtype * restrict az_r,
                                  const dtype * restrict x_s,
                                  const dtype * restrict y_s,
                                  const dtype * restrict z_s,
                                  dtype * restrict ax_s,
                                  dtype * restrict ay_s,
                                  dtype * restrict az_s

           )
{
  const dtype  eps2 = eps * eps;
  size_t       i;
  size_t       j;

  for (i = 0u; i < n; ++i)
  {
    const dtype  xi_r  = x_r[i];
    const dtype  yi_r  = y_r[i];
    const dtype  zi_r  = z_r[i];

    const dtype  xi_s  = x_s[i];
    const dtype  yi_s  = y_s[i];
    const dtype  zi_s  = z_s[i];

    dtype        axi_r = 0.0;
    dtype        ayi_r = 0.0;
    dtype        azi_r = 0.0;
    dtype        axi_s = 0.0;
    dtype        ayi_s = 0.0;
    dtype        azi_s = 0.0;

    for (j = 0u; j < n; ++j)
    {
      if (j != i)
      {
        const dtype  dx_r   = x_r[j] - xi_r;
        const dtype  dy_r   = y_r[j] - yi_r;
        const dtype  dz_r   = z_r[j] - zi_r;
        const dtype  r2_r   = dx_r * dx_r + dy_r * dy_r + dz_r * dz_r + eps2;
        const dtype  invr_r = (dtype) dtype_rsqrt(r2_r);
        const dtype  s_r    = g * mass * invr_r * invr_r * invr_r;

        const dtype  dx_s   = x_s[j] - xi_s;
        const dtype  dy_s   = y_s[j] - yi_s;
        const dtype  dz_s   = z_s[j] - zi_s;
        const dtype  r2_s   = dx_s * dx_s + dy_s * dy_s + dz_s * dz_s + eps2;
        const dtype  invr_s = (dtype) 1.0 / dtype_sqrt(r2_s);
        const dtype  s_s    = g * mass * invr_s * invr_s * invr_s;

        axi_r += dx_r * s_r;
        ayi_r += dy_r * s_r;
        azi_r += dz_r * s_r;

        axi_s += dx_s * s_s;
        ayi_s += dy_s * s_s;
        azi_s += dz_s * s_s;
      }
    }

    ax_r[i] = axi_r;
    ay_r[i] = ayi_r;
    az_r[i] = azi_r;

    ax_s[i] = axi_s;
    ay_s[i] = ayi_s;
    az_s[i] = azi_s;
  }
}


void compute_accelerations_rsqrt_third_law_check(const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  const dtype * restrict x_r,
                                  const dtype * restrict y_r,
                                  const dtype * restrict z_r,
                                  dtype * restrict ax_r,
                                  dtype * restrict ay_r,
                                  dtype * restrict az_r,
                                  const dtype * restrict x_s,
                                  const dtype * restrict y_s,
                                  const dtype * restrict z_s,
                                  dtype * restrict ax_s,
                                  dtype * restrict ay_s,
                                  dtype * restrict az_s
           )
{
  const dtype  eps2 = eps * eps;
  size_t       i;
  size_t       j;

  // Initialize acceleration arrays to zero
  const size_t bytes = n * sizeof(dtype);
  memset(ax_r, 0, bytes);
  memset(ay_r, 0, bytes);
  memset(az_r, 0, bytes);
  memset(ax_s, 0, bytes);
  memset(ay_s, 0, bytes);
  memset(az_s, 0, bytes);


  for (i = 0u; i < n; ++i)
  {
    const dtype  xi_r  = x_r[i];
    const dtype  yi_r  = y_r[i];
    const dtype  zi_r  = z_r[i];

    const dtype  xi_s  = x_s[i];
    const dtype  yi_s  = y_s[i];
    const dtype  zi_s  = z_s[i];

    dtype        axi_r = (dtype) 0.0;
    dtype        ayi_r = (dtype) 0.0;
    dtype        azi_r = (dtype) 0.0;

    dtype        axi_s = (dtype) 0.0;
    dtype        ayi_s = (dtype) 0.0;
    dtype        azi_s = (dtype) 0.0;

    for (j = i + 1; j < n; ++j)
    {
      // Compute distances and forces
      const dtype  dx_r   = x_r[j] - xi_r;
      const dtype  dy_r   = y_r[j] - yi_r;
      const dtype  dz_r   = z_r[j] - zi_r;
      const dtype  r2_r   = dx_r * dx_r + dy_r * dy_r + dz_r * dz_r + eps2;
      const dtype  invr_r = dtype_rsqrt(r2_r);
      const dtype  s_r    = g * mass * invr_r * invr_r * invr_r;

      const dtype  dx_s   = x_s[j] - xi_s;
      const dtype  dy_s   = y_s[j] - yi_s;
      const dtype  dz_s   = z_s[j] - zi_s;
      const dtype  r2_s   = dx_s * dx_s + dy_s * dy_s + dz_s * dz_s + eps2;
      const dtype  invr_s = 1. / dtype_sqrt(r2_s);
      const dtype  s_s    = g * mass * invr_s * invr_s * invr_s;

      // Accumulate to registers for Is
      axi_r += dx_r * s_r;
      ayi_r += dy_r * s_r;
      azi_r += dz_r * s_r;

      axi_s += dx_s * s_s;
      ayi_s += dy_s * s_s;
      azi_s += dz_s * s_s;

      // Accumulate to memory for Js
      ax_r[j] -= dx_r * s_r;
      ay_r[j] -= dy_r * s_r;
      az_r[j] -= dz_r * s_r;

      ax_s[j] -= dx_s * s_s;
      ay_s[j] -= dy_s * s_s;
      az_s[j] -= dz_s * s_s;
    }

    // Flush registers to memory
    ax_r[i] += axi_r;
    ay_r[i] += ayi_r;
    az_r[i] += azi_r;

    ax_s[i] += axi_s;
    ay_s[i] += ayi_s;
    az_s[i] += azi_s;
  }
}



void drift (particles_t *p,       // particle positions are modified in place
                   dtype        dt       // drift interval, often 0.5 * full step
       )
{
  const size_t  n  = p->n;
  dtype  *x_r  = p->x_r;
  dtype  *y_r  = p->y_r;
  dtype  *z_r  = p->z_r;
  const dtype  *vx_r = p->vx_r;
  const dtype  *vy_r = p->vy_r;
  const dtype  *vz_r = p->vz_r;
  
  dtype  *x_s  = p->x_s;
  dtype  *y_s  = p->y_s;
  dtype  *z_s  = p->z_s;
  const dtype  *vx_s = p->vx_s;
  const dtype  *vy_s = p->vy_s;
  const dtype  *vz_s = p->vz_s;
  size_t  i;

  for (i = 0u; i < n; ++i)
  {
    x_r[i] += dt * vx_r[i];
    y_r[i] += dt * vy_r[i];
    z_r[i] += dt * vz_r[i];
    
    x_s[i] += dt * vx_s[i];
    y_s[i] += dt * vy_s[i];
    z_s[i] += dt * vz_s[i];
  }
}


/*
 * Kick all velocities using the current accelerations. This is the K in DKD
 */
void kick (particles_t *p,       // particle velocities are modified in place
                  dtype        dt       // full kick interval
		  )
{
  size_t   n  = p->n;
  dtype  * vx_r = p->vx_r;
  dtype  * vy_r = p->vy_r;
  dtype  * vz_r = p->vz_r;
  dtype  * ax_r = p->ax_r;
  dtype  * ay_r = p->ay_r;
  dtype  * az_r = p->az_r;
  
  dtype  * vx_s = p->vx_s;
  dtype  * vy_s = p->vy_s;
  dtype  * vz_s = p->vz_s;
  dtype  * ax_s = p->ax_s;
  dtype  * ay_s = p->ay_s;
  dtype  * az_s = p->az_s;
  
  size_t   i;

  for (i = 0u; i < n; ++i)
  {
    vx_r[i] += dt * ax_r[i];
    vy_r[i] += dt * ay_r[i];
    vz_r[i] += dt * az_r[i];
    
    vx_s[i] += dt * ax_s[i];
    vy_s[i] += dt * ay_s[i];
    vz_s[i] += dt * az_s[i];
  }
}

/*
 * Compute one DKD leapfrog step:
 *
 *   1. drift positions by dt/2;
 *   2. compute accelerations at the half-step positions;
 *   3. kick velocities by dt;
 *   4. drift positions by dt/2 with the updated velocities.
 *
 * This keeps positions and velocities synchronized at integer time levels
 */
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
                               p->x_r, p->y_r, p->z_r,
                               p->x_s, p->y_s, p->z_s,
                               p->ax_r, p->ay_r, p->az_r,
                               p->ax_s, p->ay_s, p->az_s);
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

/*
 * Kinetic energy of the equal-mass system.
 * A long-double accumulator is used so that summation roundoff in the check is less likely to hide
 * errors caused by the integration or the force kernel.
 */
void kinetic_energy (const particles_t *p,
                      dtype *sum_r_out,
                      dtype *sum_s_out
			     )
{
  size_t        n    = p->n;
  dtype         mass = p->mass;
  long double   sum_r  = 0.0L;
  long double   sum_s  = 0.0L;
  size_t        i;

  for (i = 0u; i < n; ++i)
    {
      const long double  vx_r = (long double) p->vx_r[i];
      const long double  vy_r = (long double) p->vy_r[i];
      const long double  vz_r = (long double) p->vz_r[i];

      sum_r += vx_r * vx_r + vy_r * vy_r + vz_r * vz_r;
    
      const long double  vx_s = (long double) p->vx_s[i];
      const long double  vy_s = (long double) p->vy_s[i];
      const long double  vz_s = (long double) p->vz_s[i];

      sum_s += vx_s * vx_s + vy_s * vy_s + vz_s * vz_s;
    }

  *sum_r_out = 0.5L * (dtype) mass * sum_r;
  *sum_s_out = 0.5L * (dtype) mass * sum_s;
}

/*
 * Simple O(N^2) potential-energy diagnostic for the same softened potential used
 * by the force kernel.
 * Not performance critical if called only every K steps, and keeping it independent
 * of compute_accelerations_naive makes it a useful correctness check during optimization.
 */
void potential_energy_naive (particles_t *p,
                              dtype        g,
                              dtype        eps,
                              dtype *sum_r_out,
                              dtype *sum_s_out
				     )
{
  size_t        n    = p->n;
  dtype         eps2 = eps * eps;
  dtype         m2   = p->mass * p->mass;
  long double   sum_r  = 0.0L;
  long double   sum_s  = 0.0L;
  size_t        i;
  size_t        j;

  for (i = 0u; i < n; ++i)
    {
      dtype  xi_r = p->x_r[i];
      dtype  yi_r = p->y_r[i];
      dtype  zi_r = p->z_r[i];
    
      dtype  xi_s = p->x_s[i];
      dtype  yi_s = p->y_s[i];
      dtype  zi_s = p->z_s[i];

      for (j = i + 1u; j < n; ++j)
        {
          dtype  dx_r   = p->x_r[j] - xi_r;
          dtype  dy_r   = p->y_r[j] - yi_r;
          dtype  dz_r   = p->z_r[j] - zi_r;
          dtype  r2_r   = dx_r * dx_r + dy_r * dy_r + dz_r * dz_r + eps2;
          dtype  invr_r = dtype_rsqrt (r2_r);

          dtype  dx_s   = p->x_s[j] - xi_s;
          dtype  dy_s   = p->y_s[j] - yi_s;
          dtype  dz_s   = p->z_s[j] - zi_s;
          dtype  r2_s   = dx_s * dx_s + dy_s * dy_s + dz_s * dz_s + eps2;
          dtype  invr_s = 1. / dtype_sqrt (r2_s);

          sum_r -= (long double) g * (long double) m2 * (long double) invr_r;
          sum_s -= (long double) g * (long double) m2 * (long double) invr_s;
        }
    }

  *sum_r_out = (dtype) sum_r;
  *sum_s_out = (dtype) sum_s;
}

/*
 * Total mechanical energy, returned together with kinetic and potential parts for reporting.
 * The relative drift of this quantity is the main verification metric.
 */
void total_energy (particles_t *p,
                           dtype        g,
                           dtype        eps,
                           dtype  *kinetic_r,
                           dtype  *kinetic_s,
                           dtype  *potential_r,
                           dtype  *potential_s
			   )
{
  kinetic_energy (p, kinetic_r, kinetic_s);
  potential_energy_naive (p, g, eps, potential_r, potential_s);
}