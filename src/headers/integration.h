#ifndef INTEGRATION_H
#define INTEGRATION_H
#endif  //INTEGRATION_H

#include <stdlib.h>

#include "./nbody_common.h"
#include "./utils.h"
#include "./particles.h"


static void compute_accelerations_naive (size_t  n,          // number of particles
                                   dtype   g,          // gravitational constant
                                   dtype   mass,       // mass of every source particle
                                   dtype   eps,        // Plummer softening length
                                   dtype * x,          // x positions, read-only
                                   dtype * y,          // y positions, read-only
                                   dtype * z,          // z positions, read-only
                                   dtype * ax,         // x acceleration, overwritten
                                   dtype * ay,         // y acceleration, overwritten
                                   dtype * az          // z acceleration, overwritten
                  );


static void drift (particles_t *p,       // particle positions are modified in place
                   dtype        dt       // full drift interval
                  );

static void kick (particles_t *p,       // particle velocities are modified in place
                  dtype        dt       // full kick interval
                 );

static void leapfrog_dkd_step (particles_t *p,        // complete particle state, modified in place
                               dtype        g,        // gravitational constant
                               dtype        eps,      // softening length
                               dtype        dt        // full time-step
                              );

static dtype kinetic_energy (const particles_t *p    // particle velocities are read-only
                            );

static dtype potential_energy_naive (particles_t *p,        // particle positions are read-only
                                     dtype        g,        // gravitational constant
                                     dtype        eps       // softening length
                                    );

static dtype total_energy (particles_t *p,        // particle positions and velocities are read-only
                     dtype        g,        // gravitational constant
                     dtype        eps,      // softening length
                     dtype       *kinetic,  // optional output of kinetic energy
                     dtype       *potential // optional output of potential energy
                    );
