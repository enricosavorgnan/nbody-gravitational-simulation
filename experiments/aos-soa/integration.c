// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

void compute_accelerations_naive (const size_t  n,
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



void compute_accelerations_third_law(const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  particle_t    *p
           )
{
  const dtype  eps2 = eps * eps;

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
void drift (particle_t *p, dtype dt, const size_t n)
{
  for (size_t i = 0u; i < n; ++i)
    {
      p[i].x += dt * p[i].vx;
      p[i].y += dt * p[i].vy;
      p[i].z += dt * p[i].vz;
    }
}


void kick (particle_t *p, dtype dt, const size_t n)
{
  
  for (size_t i = 0u; i < n; ++i)
    {
      p[i].vx += dt * p[i].ax;
      p[i].vy += dt * p[i].ay;
      p[i].vz += dt * p[i].az;
    }
}


void leapfrog_dkd_step (particle_t   *p,             
                        const dtype    g,            
                        const dtype  eps,             
                        const dtype   dt,
                        const size_t n,
                        profiler_t   *profiler,       
                        const size_t profiler_flag,   
                        const size_t   step,          
                        const kernel_t  compute_accelerations    

			       )
{
  double t0 = 0.0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift (p, (dtype) 0.5 * dt, n);
  if (profiler_flag) profiler->first_drift_time[step] = get_time() - t0;

  // Accelerations
  if (profiler_flag) t0 = get_time();
  compute_accelerations (n, g, p->mass, eps, p);
  if (profiler_flag) profiler->force_time[step] = get_time() - t0;

  // Kick
  if (profiler_flag) t0 = get_time();
  kick (p, dt, n);
  if (profiler_flag) profiler->kick_time[step] = get_time() - t0;

  // Drift
  if (profiler_flag) t0 = get_time();
  drift (p, (dtype) 0.5 * dt, n);
  if (profiler_flag) profiler->second_drift_time[step] = get_time() - t0;

}


dtype kinetic_energy (const particle_t *p, const size_t n)
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


dtype potential_energy_naive (particle_t *p,
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
dtype total_energy (particle_t *p,
                           dtype        g,           
                           dtype        eps,   
                           const size_t n,
                           dtype       *kinetic,    
                           dtype       *potential
			   )
{
  *kinetic   = kinetic_energy (p, n);
  *potential = potential_energy_naive (p, g, eps, n);

  return *kinetic + *potential;
}