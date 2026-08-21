#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <stdlib.h>
#include <string.h>
#inclue <math.h>

#include "common.h"
#include "utils.h"
#include "particles.h"
#include "profiling.h"


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


void compute_accelerations_naive (const size_t  n,                           // number of particles
                                  const dtype   g,                           // gravitational constant
                                  const dtype   mass,                        // mass of every source particle
                                  const dtype   eps,                         // Plummer softening length
                                  const dtype   * restrict x,                          // x positions, read-only
                                  const dtype   * restrict y,                          // y positions, read-only
                                  const dtype   * restrict z,                          // z positions, read-only
                                  dtype   * restrict ax,                               // x acceleration, overwritten
                                  dtype   * restrict ay,                               // y acceleration, overwritten
                                  dtype   * restrict az                                // z acceleration, overwritten
                  );

void compute_accelerations_third_law(const size_t  n,                         // number of particles
                                  const dtype   g,                            // gravitational constant
                                  const dtype   mass,                         // mass of every source particle
                                  const dtype   eps,                          // Plummer softening length
                                  const dtype   * restrict x,                           // x positions, read-only
                                  const dtype   * restrict y,                           // y positions, read-only
                                  const dtype   * restrict z,                           // z positions, read-only
                                  dtype   * restrict ax,                                // x acceleration, overwritten
                                  dtype   * restrict ay,                                // y acceleration, overwritten
                                  dtype   * restrict az                                 // z acceleration, overwritten
                  );

void compute_accelerations_rsqrt_third_law(const size_t  n,                   // number of particles
                                  const dtype   g,                            // gravitational constant
                                  const dtype   mass,                         // mass of every source particle
                                  const dtype   eps,                          // Plummer softening length
                                  const dtype   * restrict x,                           // x positions, read-only
                                  const dtype   * restrict y,                           // y positions, read-only
                                  const dtype   * restrict z,                           // z positions, read-only
                                  dtype   * restrict ax,                                // x acceleration, overwritten
                                  dtype   * restrict ay,                                // y acceleration, overwritten
                                  dtype   * restrict az                                 // z acceleration, overwritten
                  );


void compute_accelerations_blocks_rsqrt_third_law(const size_t  n,            // number of particles
                                  const dtype   g,                            // gravitational constant
                                  const dtype   mass,                         // mass of every source particle
                                  const dtype   eps,                          // Plummer softening length
                                  const dtype   * restrict x,                           // x positions, read-only
                                  const dtype   * restrict y,                           // y positions, read-only
                                  const dtype   * restrict z,                           // z positions, read-only
                                  dtype   * restrict ax,                                // x acceleration, overwritten
                                  dtype   * restrict ay,                                // y acceleration, overwritten
                                  dtype   * restrict az                                 // z acceleration, overwritten
                  );


void compute_accelerations_omp(const size_t  n,                               // number of particles)
                                  const dtype   g,                            // gravitational constant
                                  const dtype   mass,                         // mass of every source particle
                                  const dtype   eps,                          // Plummer softening length
                                  const dtype   * restrict x,                           // x positions, read-only
                                  const dtype   * restrict y,                           // y positions, read-only
                                  const dtype   * restrict z,                           // z positions, read-only
                                  dtype   * restrict ax,                                // x acceleration, overwritten
                                  dtype   * restrict ay,                                // y acceleration, overwritten
                                  dtype   * restrict az                                 // z acceleration, overwritten
                  );


void drift (particles_t *p,                                                    // particle positions are modified in place
                   dtype        dt                                             // full drift interval
                  );

void kick (particles_t *p,                                                     // particle velocities are modified in place
                  dtype        dt                                              // full kick interval
                 );

void leapfrog_dkd_step (particles_t *p,                                         // complete particle state, modified in place
                        const dtype        g,                                   // gravitational constant
                        const dtype        eps,                                 // softening length
                        const dtype        dt,                                  // full time-step
                        profiler_t         *profiler,                           // optional profiler for per-step timing
                        const size_t        profiler_flag,                      // whether to profile this step
                        const size_t        step,                               // current step index for profiler
                        const kernel_t      compute_accelerations               // kernel function to compute accelerations
                        );

dtype kinetic_energy (const particles_t *p                                      // particle velocities are read-only
                            );

dtype potential_energy_naive (particles_t *p,                                   // particle positions are read-only
                                     dtype        g,                            // gravitational constant
                                     dtype        eps                           // softening length
                                    );

dtype total_energy (particles_t *p,                                 // particle positions and velocities are read-only
                     dtype        g,                                // gravitational constant
                     dtype        eps,                              // softening length
                     dtype       *kinetic,                          // optional output of kinetic energy
                     dtype       *potential                         // optional output of potential energy
                    );

#endif  //INTEGRATION_H