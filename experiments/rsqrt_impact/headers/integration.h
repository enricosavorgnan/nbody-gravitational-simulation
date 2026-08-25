#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "utils.h"
#include "particles.h"
#include "profiling.h"


typedef void (*kernel_t)(
    const size_t  n,                                  // number of particles)
    const dtype   g,                                  // gravitational constant
    const dtype   mass,                               // mass of every source particle
    const dtype   eps,                                // Plummer softening length
    const dtype   * restrict x_r,                       // x positions, read-only
    const dtype   * restrict y_r,                       // y positions, read-only
    const dtype   * restrict z_r,                       // z positions, read-only
    dtype   * restrict ax_r,                            // x acceleration, overwritten
    dtype   * restrict  ay_r,                           // y acceleration, overwritten
    dtype   * restrict az_r,                             // z acceleration, overwritten
    const dtype   * restrict x_s,                       // x positions, read-only
    const dtype   * restrict y_s,                       // y positions, read-only
    const dtype   * restrict z_s,                       // z positions, read-only
    dtype   * restrict ax_s,                            // x acceleration, overwritten
    dtype   * restrict  ay_s,                           // y acceleration, overwritten
    dtype   * restrict az_s  
    );



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
                                  dtype * restrict az_s                                // z acceleration, overwritten
                  );



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
                                  dtype * restrict az_s                            // z acceleration, overwritten
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

void kinetic_energy (const particles_t *p,
                      dtype *sum_r_out,
                      dtype *sum_s_out                                  // particle velocities are read-only
                            );

void potential_energy_naive (particles_t *p,
                              dtype        g,
                              dtype        eps,
                              dtype *sum_r_out,
                              dtype *sum_s_out                        // softening length
                                    );

void total_energy (particles_t *p,
                           dtype        g,
                           dtype        eps,
                           dtype  *kinetic_r,
                           dtype  *kinetic_s,
                           dtype  *potential_r,
                           dtype  *potential_s                     // optional output of potential energy
                    );

#endif  //INTEGRATION_H