#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "utils.h"
#include "particles.h"
#include "profiling.h"


// Pragmas

#ifndef DO_UNROLL
#define DO_UNROLL 0
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#if defined(__clang__)
  #if DO_UNROLL == -1
    /* Clang auto-unroll heuristic */
    #define UNROLL_PRAGMA _Pragma("clang loop unroll(enable)")
  #elif DO_UNROLL > 0
    /* Explicit unroll count */
    #define UNROLL_PRAGMA _Pragma(STR(clang loop unroll_count(DO_UNROLL)))
  #else
    /* Explicitly disable unrolling */
    #define UNROLL_PRAGMA _Pragma("clang loop unroll(disable)")
  #endif

#elif defined(__GNUC__)
  #if DO_UNROLL == -1
    /* Auto: omit pragma to let GCC optimizer decide */
    #define UNROLL_PRAGMA
  #elif DO_UNROLL > 0
    /* Explicit unroll count */
    #define UNROLL_PRAGMA _Pragma(STR(GCC unroll DO_UNROLL))
  #else
    /* GCC unroll factor 1 disables unrolling */
    #define UNROLL_PRAGMA _Pragma("GCC unroll 1")
  #endif

#else
  #define UNROLL_PRAGMA
#endif


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

void compute_accelerations_rsqrt(const size_t  n,                          // number of particles
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


void compute_accelerations_blocks(const size_t  n,                       // number of particles
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


void compute_accelerations_blocks_third_law(const size_t  n,                   // number of particles
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


void compute_accelerations_blocks_rsqrt(const size_t  n,                   // number of particles
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

//
// void compute_accelerations_omp(const size_t  n,                               // number of particles)
//                                   const dtype   g,                            // gravitational constant
//                                   const dtype   mass,                         // mass of every source particle
//                                   const dtype   eps,                          // Plummer softening length
//                                   const dtype   * restrict x,                           // x positions, read-only
//                                   const dtype   * restrict y,                           // y positions, read-only
//                                   const dtype   * restrict z,                           // z positions, read-only
//                                   dtype   * restrict ax,                                // x acceleration, overwritten
//                                   dtype   * restrict ay,                                // y acceleration, overwritten
//                                   dtype   * restrict az                                 // z acceleration, overwritten
//                   );


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