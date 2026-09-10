// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

/* ACCELERATION */


void compute_accelerations_omp_br_cross(const size_t  local_n,
                                        const size_t  visit_n,
                                        const dtype   g,
                                        const dtype   mass,
                                        const dtype   eps,
                                        const dtype * restrict local_x,
                                        const dtype * restrict local_y,
                                        const dtype * restrict local_z,
                                        const dtype * restrict visit_x,
                                        const dtype * restrict visit_y,
                                        const dtype * restrict visit_z,
                                        dtype * restrict ax,
                                        dtype * restrict ay,
                                        dtype * restrict az)
{
  const size_t blocks_i = (local_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t blocks_j = (visit_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const dtype  eps2   = eps * eps;

  #pragma omp parallel for schedule(static)
  for (size_t b_i = 0; b_i < blocks_i; b_i++)
  {
    const size_t i_start = b_i * BLOCK_SIZE;
    const size_t i_end   = (i_start + BLOCK_SIZE <= local_n) ? (i_start + BLOCK_SIZE) : local_n;

    for (size_t b_j = 0; b_j < blocks_j; b_j++)
    {
      const size_t j_start = b_j * BLOCK_SIZE;
      const size_t j_end   = (j_start + BLOCK_SIZE <= visit_n) ? (j_start + BLOCK_SIZE) : visit_n;

      for (size_t i = i_start; i < i_end; ++i)
      {
        const dtype xi = local_x[i];
        const dtype yi = local_y[i];
        const dtype zi = local_z[i];
        dtype axi = 0.0, ayi = 0.0, azi = 0.0;

        #pragma GCC ivdep
        for (size_t j = j_start; j < j_end; ++j)
        {
          const dtype dx = visit_x[j] - xi;
          const dtype dy = visit_y[j] - yi;
          const dtype dz = visit_z[j] - zi;
          const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

          const dtype invr = dtype_rsqrt(r2);
          const dtype s = g * mass * invr * invr * invr;

          axi += dx * s; ayi += dy * s; azi += dz * s;
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
      }
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

// DKD Leapfrog step
// This uses MPI to
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

  // ACCELERATIONS
  if (profiler_flag)
  {
    profiler_papi_start(profiler);
    t0 = get_time();
  }

  // MPI Definitions
  int rank=0;
  int size=1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const size_t n_local=p->n;

  // Refresh accelerations before doing computations
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n_local; i++)
  {
    p->ax[i] = 0.0;
    p->ay[i] = 0.0;
    p->az[i] = 0.0;
  }

  // Local computation (if running on 1 node)
  if (size==1) compute_accelerations(n_local, g, p->mass, eps, p->x, p->y, p->z, p->ax, p->ay, p->az);
  // More than 1 node
  else
  {
    // Allocate double buffers (sending and receiving)
    static dtype *buf_x[2] = {NULL, NULL};
    static dtype *buf_y[2] = {NULL, NULL};
    static dtype *buf_z[2] = {NULL, NULL};

    if (buf_x[0] == NULL) {
      buf_x[0] = malloc(n_local * sizeof(dtype));
      buf_x[1] = malloc(n_local * sizeof(dtype));
      buf_y[0] = malloc(n_local * sizeof(dtype));
      buf_y[1] = malloc(n_local * sizeof(dtype));
      buf_z[0] = malloc(n_local * sizeof(dtype));
      buf_z[1] = malloc(n_local * sizeof(dtype));
    }

    // Copy local particles to initial foreground buffer
    memcpy(buf_x[0], p->x, n_local * sizeof(dtype));
    memcpy(buf_y[0], p->y, n_local * sizeof(dtype));
    memcpy(buf_z[0], p->z, n_local * sizeof(dtype));

    int curr = 0;
    MPI_Request req[6];
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    // Asynchronous Ring Shift Loop
    for (int s = 1; s < size; ++s)
    {
      // Receive from previous
      MPI_Irecv(buf_x[1 - curr], n_local, MPI_DTYPE, prev, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(buf_y[1 - curr], n_local, MPI_DTYPE, prev, 1, MPI_COMM_WORLD, &req[1]);
      MPI_Irecv(buf_z[1 - curr], n_local, MPI_DTYPE, prev, 2, MPI_COMM_WORLD, &req[2]);
      // Send to next
      MPI_Isend(buf_x[curr], n_local, MPI_DTYPE, next, 0, MPI_COMM_WORLD, &req[3]);
      MPI_Isend(buf_y[curr], n_local, MPI_DTYPE, next, 1, MPI_COMM_WORLD, &req[4]);
      MPI_Isend(buf_z[curr], n_local, MPI_DTYPE, next, 2, MPI_COMM_WORLD, &req[5]);

      // Compute accelerations with current buffer
      if (s==1) compute_accelerations(n_local, g, p->mass, eps, p->x, p->y, p->z, p->ax, p->ay, p->az);
      // Compute accelerations with cross buffer
      else compute_accelerations_omp_br_cross(n_local, n_local, g, p->mass, eps, p->x, p->y, p->z, buf_x[curr], buf_y[curr], buf_z[curr], p->ax, p->ay, p->az);

      // Wait for network
      MPI_Waitall(6, req, MPI_STATUSES_IGNORE);
      // Swap buffers
      curr = 1-curr;
    }
    // Final cross-interaction
    compute_accelerations_omp_br_cross(n_local, n_local, g, p->mass, eps, p->x, p->y, p->z, buf_x[curr], buf_y[curr], buf_z[curr], p->ax, p->ay, p->az);
  }
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