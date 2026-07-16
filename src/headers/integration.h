#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <stdlib.h>

#include "common.h"
#include "utils.h"
#include "particles.h"


void compute_accelerations_naive (size_t  n,          // number of particles
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


void drift (particles_t *p,       // particle positions are modified in place
                   dtype        dt       // full drift interval
                  );

void kick (particles_t *p,       // particle velocities are modified in place
                  dtype        dt       // full kick interval
                 );

void leapfrog_dkd_step (particles_t *p,        // complete particle state, modified in place
                               dtype        g,        // gravitational constant
                               dtype        eps,      // softening length
                               dtype        dt        // full time-step
                              );

dtype kinetic_energy (const particles_t *p    // particle velocities are read-only
                            );

dtype potential_energy_naive (particles_t *p,        // particle positions are read-only
                                     dtype        g,        // gravitational constant
                                     dtype        eps       // softening length
                                    );

dtype total_energy (particles_t *p,        // particle positions and velocities are read-only
                     dtype        g,        // gravitational constant
                     dtype        eps,      // softening length
                     dtype       *kinetic,  // optional output of kinetic energy
                     dtype       *potential // optional output of potential energy
                    );

#endif  //INTEGRATION_H