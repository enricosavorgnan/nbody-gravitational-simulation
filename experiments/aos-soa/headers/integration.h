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
    particle_t  * restrict p                          // particle array, read-write
);


void compute_accelerations_naive (const size_t  n,                           // number of particles
                                  const dtype   g,                           // gravitational constant
                                  const dtype   mass,                        // mass of every source particle
                                  const dtype   eps,                         // Plummer softening length
                                  particle_t  * restrict p                            // particle array, read-write
                                  );

void compute_accelerations_third_law(const size_t  n,                         // number of particles
                                  const dtype   g,                            // gravitational constant
                                  const dtype   mass,                         // mass of every source particle
                                  const dtype   eps,                          
                                  particle_t * restrict p                            // particle array, read-write
                                  );


void drift (particle_t *p,                                                    
                   dtype dt,
                   const size_t n
                   );

void kick (particle_t *p,                                                     
                  dtype        dt,
                  const size_t n
                 );

void leapfrog_dkd_step (particle_t *p,                                         
                        const dtype        g,                                   
                        const dtype        eps,                                 // softening length
                        const dtype        dt,                                  // full time-step
                        const size_t n,
                        profiler_t         *profiler,                          
                        const size_t        profiler_flag,               
                        const size_t        step,               
                        const kernel_t      compute_accelerations               // kernel function to compute accelerations
                        );

dtype kinetic_energy (const particle_t *p, const size_t n);

dtype potential_energy_naive (particle_t *p,
                                     dtype g,                
                                     dtype eps,
                                        size_t  n
                                    );

dtype total_energy (particle_t *p,                                 
                     dtype        g,                              
                     dtype        eps,                              
                        const size_t n,                                      
                     dtype       *kinetic,                         
                     dtype       *potential                     
                    );

#endif  //INTEGRATION_H