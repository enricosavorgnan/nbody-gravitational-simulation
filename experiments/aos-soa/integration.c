// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif


void compute_accelerations_naive_aos (const size_t  n,
                                  const dtype   g,
                                  const dtype   mass, 
                                  const dtype   eps,
                                  particle_t   *p
					 )
{
  const dtype  eps2 = eps * eps;

  for (size_t i = 0u; i < n; ++i)
    {
      const dtype  xi  = p[i].x;
      const dtype  yi  = p[i].y;
      const dtype  zi  = p[i].z;
      dtype        axi = 0.0;
      dtype        ayi = 0.0;
      dtype        azi = 0.0;

      for (size_t j = 0u; j < n; ++j)
        {
          if (j != i)
            {
              const dtype  dx   = p[j].x - xi;
              const dtype  dy   = p[j].y - yi;
              const dtype  dz   = p[j].z - zi;
              const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
              const dtype  invr = 1.0 / dtype_sqrt (r2);
              const dtype  s    = g * mass * invr * invr * invr;

              axi += dx * s;
              ayi += dy * s;
              azi += dz * s;
            }
        }

      p[i].ax = axi;
      p[i].ay = ayi;
      p[i].az = azi;
    }
}



void compute_accelerations_third_law_aos(const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  particle_t    *p
           )
{
  const dtype  eps2 = eps * eps;

  for (size_t i = 0u; i < n; ++i)
  {
    p[i].ax = (dtype) 0.0;
    p[i].ay = (dtype) 0.0;
    p[i].az = (dtype) 0.0;
  }

  for (size_t i = 0u; i < n; ++i)
  {
    const dtype  xi  = p[i].x;  
    const dtype  yi  = p[i].y;
    const dtype  zi  = p[i].z;
    
    dtype        axi = (dtype) 0.0;
    dtype        ayi = (dtype) 0.0;
    dtype        azi = (dtype) 0.0;

    for (size_t j = i + 1; j < n; ++j)
    {
      // Compute distances and forces
      const dtype  dx   = p[j].x - xi;
      const dtype  dy   = p[j].y - yi;
      const dtype  dz   = p[j].z - zi;
      const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype  invr = 1.0 / dtype_sqrt (r2);
      const dtype  s    = g * mass * invr * invr * invr;

      // Accumulate to registers for Is
      axi += dx * s;
      ayi += dy * s;
      azi += dz * s;

      // Accumulate to memory for Js
      p[j].ax -= dx * s;
      p[j].ay -= dy * s;
      p[j].az -= dz * s;
    }

    // Flush registers to memory
    p[i].ax += axi;
    p[i].ay += ayi;
    p[i].az += azi;
  }
}

/* DKD */
void drift_aos (particle_t *p, dtype dt, const size_t n)
{
  for (size_t i = 0u; i < n; ++i)
    {
      p[i].x += dt * p[i].vx;
      p[i].y += dt * p[i].vy;
      p[i].z += dt * p[i].vz;
    }
}


void kick_aos (particle_t *p, dtype dt, const size_t n)
{
  
  for (size_t i = 0u; i < n; ++i)
    {
      p[i].vx += dt * p[i].ax;
      p[i].vy += dt * p[i].ay;
      p[i].vz += dt * p[i].az;
    }
}


void leapfrog_dkd_step_aos (particle_t   *p,
                        const dtype    g,            
                        const dtype  eps,             
                        const dtype   dt,
                        const size_t n,
                        profiler_t   *profiler,       
                        const size_t profiler_flag,   
                        const size_t   step,          
                        const kernel_t_aos  compute_accelerations

			       )
{
  double t0 = 0.0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift_aos (p, (dtype) 0.5 * dt, n);
  if (profiler_flag) profiler->first_drift_time[step] = get_time() - t0;

  // Accelerations
  if (profiler_flag)
  {
    t0 = get_time();
    profiler_papi_start(profiler);
  }
  compute_accelerations (n, g, p->mass, eps, p);

  if (profiler_flag)
  {
    profiler_papi_stop(profiler, step);
    profiler->force_time[step] = get_time() - t0;
  }

  // Kick
  if (profiler_flag) t0 = get_time();
  kick_aos (p, dt, n);
  if (profiler_flag) profiler->kick_time[step] = get_time() - t0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift_aos (p, (dtype) 0.5 * dt, n);
  if (profiler_flag) profiler->second_drift_time[step] = get_time() - t0;

}


dtype kinetic_energy_aos (const particle_t *p, const size_t n)
{
  dtype         mass = p->mass;
  long double   sum  = 0.0L;
  size_t        i;

  for (i = 0u; i < n; ++i)
    {
      const long double  vx = (long double) p[i].vx;
      const long double  vy = (long double) p[i].vy;
      const long double  vz = (long double) p[i].vz;

      sum += vx * vx + vy * vy + vz * vz;
    }

  return (dtype) (0.5L * (long double) mass * sum);
}


dtype potential_energy_naive_aos (particle_t *p,
                                     dtype g,        
                                     dtype eps,
                                     size_t  n
				     )
{
  dtype         eps2 = eps * eps;
  dtype         m2   = p->mass * p->mass;
  long double   sum  = 0.0L;
  size_t        i;
  size_t        j;

  for (i = 0u; i < n; ++i)
    {
      dtype  xi = p[i].x;
      dtype  yi = p[i].y;
      dtype  zi = p[i].z;

      for (j = i + 1u; j < n; ++j)
        {
          dtype  dx   = p[j].x - xi;
          dtype  dy   = p[j].y - yi;
          dtype  dz   = p[j].z - zi;
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
dtype total_energy_aos (particle_t *p,
                           dtype        g,           
                           dtype        eps,   
                           const size_t n,
                           dtype       *kinetic,    
                           dtype       *potential
			   )
{
  *kinetic   = kinetic_energy_aos (p, n);
  *potential = potential_energy_naive_aos (p, g, eps, n);

  return *kinetic + *potential;
}



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



void compute_accelerations_third_law(const size_t  n,          // number of particles
                                  const dtype   g,             // gravitational constant
                                  const dtype   mass,          // mass of every source particle
                                  const dtype   eps,           // Plummer softening length
                                  const dtype * restrict x,    // x positions, read-only
                                  const dtype * restrict y,    // y positions, read-only
                                  const dtype * restrict z,    // z positions, read-only
                                  dtype * restrict ax,         // x acceleration, overwritten
                                  dtype * restrict ay,         // y acceleration, overwritten
                                  dtype * restrict az          // z acceleration, overwritten
           )
{
  const dtype  eps2 = eps * eps;
  size_t       i;
  size_t       j;

  // Initialize acceleration arrays to zero
  const size_t bytes = n * sizeof(dtype);
  memset(ax, 0, bytes);
  memset(ay, 0, bytes);
  memset(az, 0, bytes);


  for (i = 0u; i < n; ++i)
  {
    const dtype  xi  = x[i];
    const dtype  yi  = y[i];
    const dtype  zi  = z[i];
    dtype        axi = (dtype) 0.0;
    dtype        ayi = (dtype) 0.0;
    dtype        azi = (dtype) 0.0;

    for (j = i + 1; j < n; ++j)
    {
      // Compute distances and forces
      const dtype  dx   = x[j] - xi;
      const dtype  dy   = y[j] - yi;
      const dtype  dz   = z[j] - zi;
      const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype  invr = 1.0 / dtype_sqrt (r2);
      const dtype  s    = g * mass * invr * invr * invr;

      // Accumulate to registers for Is
      axi += dx * s;
      ayi += dy * s;
      azi += dz * s;

      // Accumulate to memory for Js
      ax[j] -= dx * s;
      ay[j] -= dy * s;
      az[j] -= dz * s;
    }

    // Flush registers to memory
    ax[i] += axi;
    ay[i] += ayi;
    az[i] += azi;
  }
}

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