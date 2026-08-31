#ifndef PARTICLES_H
#define PARTICLES_H

#include <stdlib.h>

#include "common.h"
#include "utils.h"


// ----- AoS -----
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
particle_t* particle_allocate (size_t n, dtype mass);
void particle_free (particle_t *p);
void particle_read_binary (const char *path, dtype mass, particle_t **p, size_t *n_out);
void particle_write_binary (const char *path, const particle_t *p, size_t n);

// ----- SoA -----
typedef struct particles_s
{
    size_t n;
    dtype  mass;
    dtype  *x;
    dtype  *y;
    dtype  *z;
    dtype  *vx;
    dtype  *vy;
    dtype  *vz;
    dtype  *ax;
    dtype  *ay;
    dtype  *az;
} particles_t;

void particles_init_empty (particles_t *p);
void particles_allocate (particles_t  *p, size_t n, dtype mass);
void particles_free (particles_t *p);
void particles_read_binary (const char  *path, dtype mass, particles_t *p);
void particles_write_binary (const char  *path, const particles_t *p);

#endif  //PARTICLES_H