// Integration & Acceleration Stuff

#include "./headers/integration.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

// MPI ENGINE
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



// This implementation is MPI-parallelized
void leapfrog_dkd_step_mpi(particles_t   *p,
                            const dtype    g,
                            const dtype  eps,
                            const dtype   dt,
                            profiler_t   *profiler,
                            const size_t profiler_flag,
                            const size_t   step)
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
  if (size==1) compute_accelerations_omp_br(n_local, g, p->mass, eps, p->x, p->y, p->z, p->ax, p->ay, p->az);
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
      if (s==1) compute_accelerations_omp_br(n_local, g, p->mass, eps, p->x, p->y, p->z, p->ax, p->ay, p->az);
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


dtype checked_global_energy(particles_t *p, dtype g, dtype eps, int rank, int size, dtype *out_kinetic, dtype *out_potential) {
  // 1. Sum up all the local Kinetic Energies
  dtype local_kinetic = kinetic_energy(p);
  dtype global_kinetic = 0.0;
  MPI_Reduce(&local_kinetic, &global_kinetic, 1, MPI_DTYPE, MPI_SUM, 0, MPI_COMM_WORLD);

  dtype global_potential = 0.0;
  dtype *global_x = NULL, *global_y = NULL, *global_z = NULL;

  // 2. Rank 0 allocates enough memory to hold the ENTIRE system
  if (rank == 0) {
    global_x = malloc(p->n * size * sizeof(dtype));
    global_y = malloc(p->n * size * sizeof(dtype));
    global_z = malloc(p->n * size * sizeof(dtype));
  }

  // 3. Gather all positions from the cluster onto Rank 0
  MPI_Gather(p->x, p->n, MPI_DTYPE, global_x, p->n, MPI_DTYPE, 0, MPI_COMM_WORLD);
  MPI_Gather(p->y, p->n, MPI_DTYPE, global_y, p->n, MPI_DTYPE, 0, MPI_COMM_WORLD);
  MPI_Gather(p->z, p->n, MPI_DTYPE, global_z, p->n, MPI_DTYPE, 0, MPI_COMM_WORLD);

  // 4. Rank 0 computes the true Potential Energy of the entire universe
  if (rank == 0) {
    particles_t global_p = *p;
    global_p.n = p->n * size;
    global_p.x = global_x;
    global_p.y = global_y;
    global_p.z = global_z;

    global_potential = potential_energy_naive(&global_p, g, eps);

    free(global_x); free(global_y); free(global_z);
  }

  if (out_kinetic) *out_kinetic = global_kinetic;
  if (out_potential) *out_potential = global_potential;

  return global_kinetic + global_potential;
}



// SERIAL/OMP ENGINE

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




// COMMON ENGINE
void compute_accelerations_baseline (const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  const dtype * x,
                                  const dtype * y,
                                  const dtype * z,
                                  dtype * ax,
                                  dtype * ay,
                                  dtype * az
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
      if (j!=i)
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

      ax[i] = axi;
      ay[i] = ayi;
      az[i] = azi;
    }
}



void compute_accelerations_third_law(const size_t  n,
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



void compute_accelerations_rsqrt(const size_t  n,
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
      const dtype  dx   = x[j] - xi;
      const dtype  dy   = y[j] - yi;
      const dtype  dz   = z[j] - zi;
      const dtype  r2   = dx * dx + dy * dy + dz * dz + eps2;
      const dtype  invr = (dtype) dtype_rsqrt(r2);
      const dtype  s    = g * mass * invr * invr * invr;

      axi += dx * s;
      ayi += dy * s;
      azi += dz * s;
    }

    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
}


void compute_accelerations_blocks(const size_t  n,
                                  const dtype  g,
                                  const dtype  mass,                 
                                  const dtype  eps,                   
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
          const dtype invr = (dtype) 1.0 / dtype_sqrt (r2);
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

    for (size_t i = 0; i < i_count; i++)
    {
      ax[i_start + i] = block_ax[i];
      ay[i_start + i] = block_ay[i];
      az[i_start + i] = block_az[i];
    }
  }
}


void compute_accelerations_rsqrt_third_law(const size_t  n,ì
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
      const dtype  invr = dtype_rsqrt(r2);
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


void compute_accelerations_blocks_rsqrt(const size_t  n,                    
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

    for (size_t i = 0; i < i_count; i++)
    {
      ax[i_start + i] = block_ax[i];
      ay[i_start + i] = block_ay[i];
      az[i_start + i] = block_az[i];
    }
  }
}


void compute_accelerations_blocks_third_law(const size_t  n,
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

      // 1. Third Law requires us to zero the global arrays first,
      // because we will be using += to accumulate forces globally.
      for (size_t i = 0u; i < n; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
      }

      // 2. Iterate over block i
      for (size_t b_i = 0; b_i < n; b_i += BLOCK_SIZE)
      {
        const size_t i_end = (b_i + BLOCK_SIZE < n) ? (b_i + BLOCK_SIZE) : n;

        // Local accumulator for the i-block. Easily fits in L1 cache (128 elements = ~1KB).
        dtype block_ax_i[BLOCK_SIZE] = {0.0};
        dtype block_ay_i[BLOCK_SIZE] = {0.0};
        dtype block_az_i[BLOCK_SIZE] = {0.0};

        // 3. Diagonal Block (b_i == b_j): compute upper triangle to avoid self-interaction
        for (size_t i = b_i; i < i_end; ++i)
        {
          const size_t ii = i - b_i;
          const dtype xi = x[i];
          const dtype yi = y[i];
          const dtype zi = z[i];

          for (size_t j = i + 1; j < i_end; ++j)
          {
            const size_t jj = j - b_i;
            const dtype dx = x[j] - xi;
            const dtype dy = y[j] - yi;
            const dtype dz = z[j] - zi;
            const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

            const dtype invr = (dtype) 1.0 / dtype_sqrt(r2); // Uses your AVX-512 replacement
            const dtype s = g * mass * invr * invr * invr;

            const dtype fx = dx * s;
            const dtype fy = dy * s;
            const dtype fz = dz * s;

            block_ax_i[ii] += fx;
            block_ay_i[ii] += fy;
            block_az_i[ii] += fz;

            block_ax_i[jj] -= fx;
            block_ay_i[jj] -= fy;
            block_az_i[jj] -= fz;
          }
        }

        // 4. Off-Diagonal Blocks (b_j > b_i): compute full N x M block interactions
        for (size_t b_j = b_i + BLOCK_SIZE; b_j < n; b_j += BLOCK_SIZE)
        {
          const size_t j_end = (b_j + BLOCK_SIZE < n) ? (b_j + BLOCK_SIZE) : n;

          // Local accumulator for the j-block. Also stays locked in L1 cache.
          dtype block_ax_j[BLOCK_SIZE] = {0.0};
          dtype block_ay_j[BLOCK_SIZE] = {0.0};
          dtype block_az_j[BLOCK_SIZE] = {0.0};

          for (size_t i = b_i; i < i_end; ++i)
          {
            const size_t ii = i - b_i;
            const dtype xi = x[i];
            const dtype yi = y[i];
            const dtype zi = z[i];

            // Isolate the i-accumulation to allow GCC to auto-vectorize the j loop
            dtype axi = 0.0;
            dtype ayi = 0.0;
            dtype azi = 0.0;

            // With -ffast-math, GCC perfectly vectorizes this loop because it is a
            // purely streaming vector operation reading 'x' and writing to 'block_ax_j'
            #pragma GCC ivdep
            for (size_t j = b_j; j < j_end; ++j)
            {
              const size_t jj = j - b_j;
              const dtype dx = x[j] - xi;
              const dtype dy = y[j] - yi;
              const dtype dz = z[j] - zi;
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

              const dtype invr = (dtype) 1.0 / dtype_sqrt(r2);
              const dtype s = g * mass * invr * invr * invr;

              const dtype fx = dx * s;
              const dtype fy = dy * s;
              const dtype fz = dz * s;

              axi += fx;
              ayi += fy;
              azi += fz;

              // Contiguous write to L1 cache, NO global memory thrashing
              block_ax_j[jj] -= fx;
              block_ay_j[jj] -= fy;
              block_az_j[jj] -= fz;
            }

            // Commit the reduced i-forces to the local i-buffer
            block_ax_i[ii] += axi;
            block_ay_i[ii] += ayi;
            block_az_i[ii] += azi;
          }

          // 5. Commit the finished j-block to global memory
          // (Happens only once per block pair, saving billions of L1 misses)
          for (size_t j = b_j; j < j_end; ++j)
          {
            const size_t jj = j - b_j;
            ax[j] += block_ax_j[jj];
            ay[j] += block_ay_j[jj];
            az[j] += block_az_j[jj];
          }
        }

        // 6. Commit the finished i-block to global memory
        for (size_t i = b_i; i < i_end; ++i)
        {
          const size_t ii = i - b_i;
          ax[i] += block_ax_i[ii];
          ay[i] += block_ay_i[ii];
          az[i] += block_az_i[ii];
        }
      }
    }


void compute_accelerations_blocks_rsqrt_third_law(const size_t  n,
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

      // 1. Third Law requires us to zero the global arrays first,
      // because we will be using += to accumulate forces globally.
      for (size_t i = 0u; i < n; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
      }

      // 2. Iterate over block i
      for (size_t b_i = 0; b_i < n; b_i += BLOCK_SIZE)
      {
        const size_t i_end = (b_i + BLOCK_SIZE < n) ? (b_i + BLOCK_SIZE) : n;

        // Local accumulator for the i-block. Easily fits in L1 cache (128 elements = ~1KB).
        dtype block_ax_i[BLOCK_SIZE] = {0.0};
        dtype block_ay_i[BLOCK_SIZE] = {0.0};
        dtype block_az_i[BLOCK_SIZE] = {0.0};

        // 3. Diagonal Block (b_i == b_j): compute upper triangle to avoid self-interaction
        for (size_t i = b_i; i < i_end; ++i)
        {
          const size_t ii = i - b_i;
          const dtype xi = x[i];
          const dtype yi = y[i];
          const dtype zi = z[i];

          for (size_t j = i + 1; j < i_end; ++j)
          {
            const size_t jj = j - b_i;
            const dtype dx = x[j] - xi;
            const dtype dy = y[j] - yi;
            const dtype dz = z[j] - zi;
            const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

            const dtype invr = dtype_rsqrt(r2); // Uses your AVX-512 replacement
            const dtype s = g * mass * invr * invr * invr;

            const dtype fx = dx * s;
            const dtype fy = dy * s;
            const dtype fz = dz * s;

            block_ax_i[ii] += fx;
            block_ay_i[ii] += fy;
            block_az_i[ii] += fz;

            block_ax_i[jj] -= fx;
            block_ay_i[jj] -= fy;
            block_az_i[jj] -= fz;
          }
        }

        // 4. Off-Diagonal Blocks (b_j > b_i): compute full N x M block interactions
        for (size_t b_j = b_i + BLOCK_SIZE; b_j < n; b_j += BLOCK_SIZE)
        {
          const size_t j_end = (b_j + BLOCK_SIZE < n) ? (b_j + BLOCK_SIZE) : n;

          // Local accumulator for the j-block. Also stays locked in L1 cache.
          dtype block_ax_j[BLOCK_SIZE] = {0.0};
          dtype block_ay_j[BLOCK_SIZE] = {0.0};
          dtype block_az_j[BLOCK_SIZE] = {0.0};

          for (size_t i = b_i; i < i_end; ++i)
          {
            const size_t ii = i - b_i;
            const dtype xi = x[i];
            const dtype yi = y[i];
            const dtype zi = z[i];

            // Isolate the i-accumulation to allow GCC to auto-vectorize the j loop
            dtype axi = 0.0;
            dtype ayi = 0.0;
            dtype azi = 0.0;

            // With -ffast-math, GCC perfectly vectorizes this loop because it is a
            // purely streaming vector operation reading 'x' and writing to 'block_ax_j'
            #pragma GCC ivdep
            for (size_t j = b_j; j < j_end; ++j)
            {
              const size_t jj = j - b_j;
              const dtype dx = x[j] - xi;
              const dtype dy = y[j] - yi;
              const dtype dz = z[j] - zi;
              const dtype r2 = dx * dx + dy * dy + dz * dz + eps2;

              const dtype invr = dtype_rsqrt(r2);
              const dtype s = g * mass * invr * invr * invr;

              const dtype fx = dx * s;
              const dtype fy = dy * s;
              const dtype fz = dz * s;

              axi += fx;
              ayi += fy;
              azi += fz;

              // Contiguous write to L1 cache, NO global memory thrashing
              block_ax_j[jj] -= fx;
              block_ay_j[jj] -= fy;
              block_az_j[jj] -= fz;
            }

            // Commit the reduced i-forces to the local i-buffer
            block_ax_i[ii] += axi;
            block_ay_i[ii] += ayi;
            block_az_i[ii] += azi;
          }

          // 5. Commit the finished j-block to global memory
          // (Happens only once per block pair, saving billions of L1 misses)
          for (size_t j = b_j; j < j_end; ++j)
          {
            const size_t jj = j - b_j;
            ax[j] += block_ax_j[jj];
            ay[j] += block_ay_j[jj];
            az[j] += block_az_j[jj];
          }
        }

        // 6. Commit the finished i-block to global memory
        for (size_t i = b_i; i < i_end; ++i)
        {
          const size_t ii = i - b_i;
          ax[i] += block_ax_i[ii];
          ay[i] += block_ay_i[ii];
          az[i] += block_az_i[ii];
        }
      }
    }


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



void compute_accelerations_omp_rt(const size_t  n,
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
        #pragma omp for schedule(dynamic, 1)
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

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      ax[i] = 0.0;
      ay[i] = 0.0;
      az[i] = 0.0;
    }

      // Dynamic scheduling with Automatic Array Reduction
      #pragma omp parallel for schedule(dynamic, 1) reduction(+:ax[0:n], ay[0:n], az[0:n])
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

// Drift
void drift (particles_t *p,
            dtype dt)
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

// Kick
void kick (particles_t *p,
                  dtype        dt
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

// Kinetic Energy
dtype kinetic_energy (const particles_t *p
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

// Potential Energy
dtype potential_energy_naive (particles_t *p,ì
                                     dtype        g,         
                                     dtype        eps ì
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

// Total Energy
dtype total_energy (particles_t *p,     ì
                           dtype        g,            
                           dtype        eps,  ì
                           dtype       *kinetic,ì
                           dtype       *potentialì
			   )
{
  *kinetic   = kinetic_energy (p);
  *potential = potential_energy_naive (p, g, eps);

  return *kinetic + *potential;
}

