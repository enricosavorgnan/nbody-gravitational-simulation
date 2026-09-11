#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "common.h"
#include "utils.h"
#include "particles.h"
#include "profiling.h"

#if USE_MPI==1
typedef void (*kernel_t)(
    const size_t  local_n,
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
    dtype * restrict az);


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
                                        dtype * restrict az);


void leapfrog_dkd_step_mpi(particles_t   *p,
                            const dtype    g,
                            const dtype  eps,
                            const dtype   dt,
                            profiler_t   *profiler,
                            const size_t profiler_flag,
                            const size_t   step);


dtype checked_global_energy(particles_t *p,
                            dtype g,
                            dtype eps,
                            int rank,
                            int size,
                            dtype *out_kinetic,
                            dtype *out_potential);
#else
typedef void (*kernel_t)(
    const size_t  n,                                  // number of particles)
    const dtype   g,                                  // gravitational constant
    const dtype   mass,                               // mass of every source particle
    const dtype   eps,                                // Plummer softening length
    const dtype   * restrict x,                       // x positions, read-only
    const dtype   * restrict y,                       // y positions, read-only
    const dtype   * restrict z,                       // z positions, read-only
    dtype   * restrict ax,                            // x acceleration, overwritten
    dtype   * restrict  ay,                           // y acceleration, overwritten
    dtype   * restrict az                             // z acceleration, overwritten
);

// SERIAL/OMP ENGINE

void leapfrog_dkd_step (particles_t   *p,
                        const dtype    g,
                        const dtype  eps,
                        const dtype   dt,
                        profiler_t   *profiler,
                        const size_t profiler_flag,
                        const size_t   step,
                        const kernel_t  compute_accelerations);


void compute_accelerations_baseline(const size_t  n,
                                    const dtype   g,
                                    const dtype   mass,
                                    const dtype   eps,
                                    const dtype * x,
                                    const dtype * y,
                                    const dtype * z,
                                    dtype * ax,
                                    dtype * ay,
                                    dtype * az);


void compute_accelerations_naive(const size_t  n,
                                 const dtype   g,
                                 const dtype   mass,
                                 const dtype   eps,
                                 const dtype * restrict x,
                                 const dtype * restrict y,
                                 const dtype * restrict z,
                                 dtype * restrict ax,
                                 dtype * restrict ay,
                                 dtype * restrict az);


void compute_accelerations_third_law(const size_t  n,
                                      const dtype   g,
                                      const dtype   mass,
                                      const dtype   eps,
                                      const dtype * restrict x,
                                      const dtype * restrict y,
                                      const dtype * restrict z,
                                      dtype * restrict ax,
                                      dtype * restrict ay,
                                      dtype * restrict az);


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
           );


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
                                 );


void compute_accelerations_rsqrt_third_law(const size_t  n,
                                           const dtype   g,
                                           const dtype   mass,
                                           const dtype   eps,
                                           const dtype * restrict x,
                                           const dtype * restrict y,
                                           const dtype * restrict z,
                                           dtype * restrict ax,
                                           dtype * restrict ay,
                                           dtype * restrict az);


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
                                        );


void compute_accelerations_blocks_third_law(const size_t  n,
                                            const dtype   g,
                                            const dtype   mass,
                                            const dtype   eps,
                                            const dtype * restrict x,
                                            const dtype * restrict y,
                                            const dtype * restrict z,
                                            dtype       * restrict ax,
                                            dtype       * restrict ay,
                                            dtype       * restrict az);


void compute_accelerations_blocks_rsqrt_third_law(const size_t  n,
                                                  const dtype   g,
                                                  const dtype   mass,
                                                  const dtype   eps,
                                                  const dtype * restrict x,
                                                  const dtype * restrict y,
                                                  const dtype * restrict z,
                                                  dtype       * restrict ax,
                                                  dtype       * restrict ay,
                                                  dtype       * restrict az);


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
                                  );



void compute_accelerations_omp_rt(const size_t  n,
                                  const dtype   g,
                                  const dtype   mass,
                                  const dtype   eps,
                                  const dtype * restrict x,
                                  const dtype * restrict y,
                                  const dtype * restrict z,
                                  dtype * restrict ax,
                                  dtype * restrict ay,
                                  dtype * restrict az);


void compute_accelerations_omp_brt(const size_t  n,
                                   const dtype   g,
                                   const dtype   mass,
                                   const dtype   eps,
                                   const dtype * restrict x,
                                   const dtype * restrict y,
                                   const dtype * restrict z,
                                   dtype       * restrict ax,
                                   dtype       * restrict ay,
                                   dtype       * restrict az);


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
                                      dtype       * restrict az);


void compute_accelerations_omp_brt_red(const size_t  n,
                                       const dtype   g,
                                       const dtype   mass,
                                       const dtype   eps,
                                       const dtype * restrict x,
                                       const dtype * restrict y,
                                       const dtype * restrict z,
                                       dtype       * restrict ax,
                                       dtype       * restrict ay,
                                       dtype       * restrict az);

#endif

// COMMON

// Drift
void drift(particles_t *p,
           dtype dt);

// Kick
void kick(particles_t *p,
          dtype dt);

// Kinetic Energy
dtype kinetic_energy(const particles_t *p);

// Potential Energy
dtype potential_energy_naive(particles_t *p,
                             dtype g,
                             dtype eps);

// Total Energy
dtype total_energy(particles_t *p,
                   dtype g,
                   dtype eps,
                   dtype *kinetic,
                   dtype *potential);


#endif  //INTEGRATION_H