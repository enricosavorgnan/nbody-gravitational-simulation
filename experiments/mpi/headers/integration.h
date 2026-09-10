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
                        const size_t        step                               // current step index for profiler
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