// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

/* ACCELERATION */

void compute_accelerations_omp_br(const size_t  n,
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
    const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const dtype  eps2   = eps * eps;

    // Initialize accelerations to 0
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    ax[i] = 0.0;
    ay[i] = 0.0;
    az[i] = 0.0;
  }

    // Loop over i-th blocks
    // Scheduler is static here because the total work is the same for all blocks
#pragma omp parallel for schedule(static)
  for (size_t b_i = 0; b_i < blocks; b_i++)
  {
    const size_t i_start = b_i * BLOCK_SIZE;
    const size_t i_end   = (i_start + BLOCK_SIZE <= n) ? (i_start + BLOCK_SIZE) : n;
    const size_t i_count = i_end - i_start;

    dtype block_ax[BLOCK_SIZE];
    dtype block_ay[BLOCK_SIZE];
    dtype block_az[BLOCK_SIZE];

    // Initialize the block accelerations to 0
    for (size_t i = 0; i < i_count; i++)
    {
      block_ax[i] = 0.0;
      block_ay[i] = 0.0;
      block_az[i] = 0.0;
    }

    // Loop over j-th blocks
    for (size_t b_j = 0; b_j < blocks; b_j++)
    {
      const size_t j_start = b_j * BLOCK_SIZE;
      const size_t j_end   = (j_start + BLOCK_SIZE <= n) ? (j_start + BLOCK_SIZE) : n;

      // Inner Loops
      for (size_t i = i_start; i < i_end; i++)
      {
        const size_t i_rel = i - i_start;
        const dtype xi  = x[i];
        const dtype yi  = y[i];
        const dtype zi  = z[i];
        dtype       axi = 0.0;
        dtype       ayi = 0.0;
        dtype       azi = 0.0;

        for (size_t j = j_start; j < j_end; j++)
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

        block_ax[i_rel] += axi;
        block_ay[i_rel] += ayi;
        block_az[i_rel] += azi;
      }
    }

    // Final update
    for (size_t i = 0; i < i_count; i++)
    {
      ax[i_start + i] = block_ax[i];
      ay[i_start + i] = block_ay[i];
      az[i_start + i] = block_az[i];
    }
  }
}



void compute_accelerations_omp_rt(const size_t  n,          // number of particles
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
      const dtype eps2 = eps * eps;
      int num_threads = omp_get_max_threads();

      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
      }

      // Allocate thread-local buffers
      // Buffer are allocated so to avoid false sharing and to be NUMA-friendly:
      // each thread works on its own private buffer, at the end all threads sum their contributions.
      static dtype* thread_buffers = NULL;
      #pragma omp single
      if (thread_buffers == NULL) {
          thread_buffers = calloc(num_threads * 3 * n, sizeof(dtype));
      }

      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        dtype* my_ax = thread_buffers + (tid * 3 * n);
        dtype* my_ay = my_ax + n;
        dtype* my_az = my_ay + n;

        // First-touch NUMA policy
        // (Locks this RAM to the current CPU's CCD)
        memset(my_ax, 0, 3 * n * sizeof(dtype));

        // The scheduler is here dynamic because there is a strong unbalance between the first and the
        // last iterations due to the triangularity of the workloads.
        // The chunk size is set to block_size, so that each thread works on a contiguous block of data.
        #pragma omp for schedule(dynamic, BLOCK_SIZE)
        for (size_t i = 0; i < n; ++i)
        {
          const dtype xi = x[i];
          const dtype yi = y[i];
          const dtype zi = z[i];

          dtype axi = 0.0;
          dtype ayi = 0.0;
          dtype azi = 0.0;

          // The compiler is free to vectorize this loop, as there are no data dependencies
          #pragma GCC ivdep
          for (size_t j = i + 1; j < n; ++j)
          {
            const dtype dx = x[j] - xi;
            const dtype dy = y[j] - yi;
            const dtype dz = z[j] - zi;
            const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

            const dtype invr = dtype_rsqrt(r2);
            const dtype s = g * mass * invr * invr * invr;

            const dtype fx = dx * s;
            const dtype fy = dy * s;
            const dtype fz = dz * s;

            // Accumulate for 'i'
            axi += fx;
            ayi += fy;
            azi += fz;

            // Commit reaction force for 'j' into thread's private array
            my_ax[j] -= fx;
            my_ay[j] -= fy;
            my_az[j] -= fz;
          }

          // Commit the 'i' accumulator into thread's private array
          my_ax[i] += axi;
          my_ay[i] += ayi;
          my_az[i] += azi;
        }

        // Global Reduction Phase
        // Once all threads finished, they sum their private arrays to global.
        // This work can be parallelized with a static schedule.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
          dtype sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

          for (int t = 0; t < num_threads; ++t) {
            size_t offset = (t * 3 * n) + i;
            sum_x += thread_buffers[offset];
            sum_y += thread_buffers[offset + n];
            sum_z += thread_buffers[offset + 2 * n];
          }

          ax[i] += sum_x;
          ay[i] += sum_y;
          az[i] += sum_z;
        }
      }
    }


void compute_accelerations_omp_brt(const size_t  n,
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
      int num_threads = omp_get_max_threads();

      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
      }

      // Allocate thread buffers, all at once
      static dtype* thread_buffers = NULL;
      if (thread_buffers == NULL) {
          thread_buffers = calloc(num_threads * 3 * n, sizeof(dtype));
      }

      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        dtype* my_ax = thread_buffers + (tid * 3 * n);
        dtype* my_ay = my_ax + n;
        dtype* my_az = my_ay + n;

        // Set values to 0
        // This because of the First-touch NUMA policy
        memset(my_ax, 0, 3 * n * sizeof(dtype));

        // Dynamic scheduling for the triangular workload
        #pragma omp for schedule(dynamic, BLOCK_SIZE)
        for (size_t b_i = 0; b_i < n; b_i += BLOCK_SIZE)
        {
          const size_t i_end = (b_i + BLOCK_SIZE < n) ? (b_i + BLOCK_SIZE) : n;

          dtype block_ax_i[BLOCK_SIZE] = {0.0};
          dtype block_ay_i[BLOCK_SIZE] = {0.0};
          dtype block_az_i[BLOCK_SIZE] = {0.0};

          // Diagonal Blocks
          for (size_t i = b_i; i < i_end; ++i)
          {
            const size_t ii = i - b_i;
            const dtype xi = x[i]; const dtype yi = y[i]; const dtype zi = z[i];

            for (size_t j = i + 1; j < i_end; ++j)
            {
              const size_t jj = j - b_i;
              const dtype dx = x[j] - xi; const dtype dy = y[j] - yi; const dtype dz = z[j] - zi;
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

              const dtype invr = dtype_rsqrt(r2);
              const dtype s = g * mass * invr * invr * invr;

              const dtype fx = dx * s; const dtype fy = dy * s; const dtype fz = dz * s;

              block_ax_i[ii] += fx; block_ay_i[ii] += fy; block_az_i[ii] += fz;
              block_ax_i[jj] -= fx; block_ay_i[jj] -= fy; block_az_i[jj] -= fz;
            }
          }

          // Off-Diagonal Blocks
          for (size_t b_j = b_i + BLOCK_SIZE; b_j < n; b_j += BLOCK_SIZE)
          {
            const size_t j_end = (b_j + BLOCK_SIZE < n) ? (b_j + BLOCK_SIZE) : n;

            dtype block_ax_j[BLOCK_SIZE] = {0.0};
            dtype block_ay_j[BLOCK_SIZE] = {0.0};
            dtype block_az_j[BLOCK_SIZE] = {0.0};

            for (size_t i = b_i; i < i_end; ++i)
            {
              const size_t ii = i - b_i;
              const dtype xi = x[i]; const dtype yi = y[i]; const dtype zi = z[i];
              dtype axi = 0.0, ayi = 0.0, azi = 0.0;

              // The compiler is free to vectorize this loop, as there are no data dependencies
              #pragma GCC ivdep
              for (size_t j = b_j; j < j_end; ++j)
              {
                const size_t jj = j - b_j;
                const dtype dx = x[j] - xi; const dtype dy = y[j] - yi; const dtype dz = z[j] - zi;
                const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

                const dtype invr = dtype_rsqrt(r2);
                const dtype s = g * mass * invr * invr * invr;
                const dtype fx = dx * s; const dtype fy = dy * s; const dtype fz = dz * s;

                axi += fx; ayi += fy; azi += fz;
                block_ax_j[jj] -= fx; block_ay_j[jj] -= fy; block_az_j[jj] -= fz;
              }
              block_ax_i[ii] += axi; block_ay_i[ii] += ayi; block_az_i[ii] += azi;
            }

            // Commit the j-block to thread's local buffer
            for (size_t j = b_j; j < j_end; ++j)
            {
              const size_t jj = j - b_j;
              my_ax[j] += block_ax_j[jj];
              my_ay[j] += block_ay_j[jj];
              my_az[j] += block_az_j[jj];
            }
          }

          // Commit the i-block to thread's local buffer
          for (size_t i = b_i; i < i_end; ++i)
          {
            const size_t ii = i - b_i;
            my_ax[i] += block_ax_i[ii];
            my_ay[i] += block_ay_i[ii];
            my_az[i] += block_az_i[ii];
          }
        }

        // Global reduction
        // Wait for all threads to finish computing forces,
        // then sum everything back to global 'ax'
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
          dtype sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
          for (int t = 0; t < num_threads; ++t) {
            size_t offset = (t * 3 * n) + i;
            sum_x += thread_buffers[offset];
            sum_y += thread_buffers[offset + n];
            sum_z += thread_buffers[offset + 2 * n];
          }
          ax[i] += sum_x;
          ay[i] += sum_y;
          az[i] += sum_z;
        }
      }
    }


// Reduction Version of the ORT kernel
// The idea here is to exploit the OMP reduction clause to avoid explicit synchronization
// However, we lose some control over
void compute_accelerations_omp_rt_red(const size_t  n,
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

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    ax[i] = 0.0;
    ay[i] = 0.0;
    az[i] = 0.0;
  }

  // Dynamic scheduling with Automatic Array Reduction
#pragma omp parallel for schedule(dynamic, BLOCK_SIZE) reduction(+:ax[0:n], ay[0:n], az[0:n])
  for (size_t i = 0; i < n; ++i)
  {
    const dtype xi = x[i]; const dtype yi = y[i]; const dtype zi = z[i];
    dtype axi = 0.0, ayi = 0.0, azi = 0.0;

#pragma GCC ivdep
    for (size_t j = i + 1; j < n; ++j)
    {
      const dtype dx = x[j] - xi; const dtype dy = y[j] - yi; const dtype dz = z[j] - zi;
      const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
      const dtype invr = dtype_rsqrt(r2);
      const dtype s = g * mass * invr * invr * invr;
      const dtype fx = dx * s; const dtype fy = dy * s; const dtype fz = dz * s;

      axi += fx; ayi += fy; azi += fz;

      // OpenMP safely writes this to a secret private array for this thread
      ax[j] -= fx; ay[j] -= fy; az[j] -= fz;
    }
    ax[i] += axi; ay[i] += ayi; az[i] += azi;
  }
}


void compute_accelerations_omp_brt_red(const size_t  n,
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

// #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      ax[i] = 0.0;
      ay[i] = 0.0;
      az[i] = 0.0;
    }

      // Dynamic scheduling with Automatic Array Reduction
    #pragma omp parallel for schedule(dynamic, BLOCK_SIZE) reduction(+:ax[0:n], ay[0:n], az[0:n])
      for (size_t b_i = 0; b_i < n; b_i += BLOCK_SIZE)
      {
        const size_t i_end = (b_i + BLOCK_SIZE < n) ? (b_i + BLOCK_SIZE) : n;

        dtype block_ax_i[BLOCK_SIZE] = {0.0};
        dtype block_ay_i[BLOCK_SIZE] = {0.0};
        dtype block_az_i[BLOCK_SIZE] = {0.0};

        // 1. Diagonal Block
        for (size_t i = b_i; i < i_end; ++i)
        {
          const size_t ii = i - b_i;
          const dtype xi = x[i]; const dtype yi = y[i]; const dtype zi = z[i];

          for (size_t j = i + 1; j < i_end; ++j)
          {
            const size_t jj = j - b_i;
            const dtype dx = x[j] - xi; const dtype dy = y[j] - yi; const dtype dz = z[j] - zi;
            const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
            const dtype invr = dtype_rsqrt(r2);
            const dtype s = g * mass * invr * invr * invr;
            const dtype fx = dx * s; const dtype fy = dy * s; const dtype fz = dz * s;

            block_ax_i[ii] += fx; block_ay_i[ii] += fy; block_az_i[ii] += fz;
            block_ax_i[jj] -= fx; block_ay_i[jj] -= fy; block_az_i[jj] -= fz;
          }
        }

        // 2. Off-Diagonal Blocks
        for (size_t b_j = b_i + BLOCK_SIZE; b_j < n; b_j += BLOCK_SIZE)
        {
          const size_t j_end = (b_j + BLOCK_SIZE < n) ? (b_j + BLOCK_SIZE) : n;

          dtype block_ax_j[BLOCK_SIZE] = {0.0};
          dtype block_ay_j[BLOCK_SIZE] = {0.0};
          dtype block_az_j[BLOCK_SIZE] = {0.0};

          for (size_t i = b_i; i < i_end; ++i)
          {
            const size_t ii = i - b_i;
            const dtype xi = x[i]; const dtype yi = y[i]; const dtype zi = z[i];
            dtype axi = 0.0, ayi = 0.0, azi = 0.0;

    #pragma GCC ivdep
            for (size_t j = b_j; j < j_end; ++j)
            {
              const size_t jj = j - b_j;
              const dtype dx = x[j] - xi; const dtype dy = y[j] - yi; const dtype dz = z[j] - zi;
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;
              const dtype invr = dtype_rsqrt(r2);
              const dtype s = g * mass * invr * invr * invr;
              const dtype fx = dx * s; const dtype fy = dy * s; const dtype fz = dz * s;

              axi += fx; ayi += fy; azi += fz;
              block_ax_j[jj] -= fx; block_ay_j[jj] -= fy; block_az_j[jj] -= fz;
            }
            block_ax_i[ii] += axi; block_ay_i[ii] += ayi; block_az_i[ii] += azi;
          }

          // OpenMP safely handles the global write to 'ax[j]' behind the scenes!
          for (size_t j = b_j; j < j_end; ++j)
          {
            const size_t jj = j - b_j;
            ax[j] += block_ax_j[jj];
            ay[j] += block_ay_j[jj];
            az[j] += block_az_j[jj];
          }
        }

        // OpenMP safely handles the global write to 'ax[i]' behind the scenes!
        for (size_t i = b_i; i < i_end; ++i)
        {
          const size_t ii = i - b_i;
          ax[i] += block_ax_i[ii];
          ay[i] += block_ay_i[ii];
          az[i] += block_az_i[ii];
        }
      }
    }


/* DKD */

/*
 * Drift all particles by a time interval using the current velocities.
 * The DKD leapfrog workflow calls it twice per step: a half-drift before the
 * force evaluation and a half-drift after the kick.
 *
 * Again: are data qualifiers missed for optimization?
 */
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

/*
 * Kick all velocities using the current accelerations. This is the K in DKD
 */
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

/*
 * Kinetic energy of the equal-mass system.
 * A long-double accumulator is used so that summation roundoff in the check is less likely to hide
 * errors caused by the integration or the force kernel.
 */
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

/*
 * Simple O(N^2) potential-energy diagnostic for the same softened potential used
 * by the force kernel.
 * Not performance critical if called only every K steps, and keeping it independent
 * of compute_accelerations_naive makes it a useful correctness check during optimization.
 */
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

/*
 * Total mechanical energy, returned together with kinetic and potential parts for reporting.
 * The relative drift of this quantity is the main verification metric.
 */
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