#ifndef PARTICLES_H
#define PARTICLES_H

#include <stdlib.h>

#include "common.h"
#include "utils.h"


typedef struct particle_s
{
    dtype mass;
    dtype x;
    dtype y;
    dtype z;
    dtype vx;
    dtype vy;
    dtype vz;
    dtype ax;
    dtype ay;
    dtype az;
} particle_t ;

float dtype_to_storage_float (dtype value, const char *component, size_t i);
particle_t* particles_allocate (size_t n, dtype mass);
void particles_free (particle_t *p);
void particles_read_binary (const char *path, dtype mass, particle_t **p, size_t *n_out);
void particles_write_binary (const char *path, const particle_t *p, size_t n);

#endif  //PARTICLES_H