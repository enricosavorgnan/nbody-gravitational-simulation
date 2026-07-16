#ifndef PARTICLES_H
#define PARTICLES_H
#endif  //PARTICLES_H

#include <stdlib.h>

#include "./nbody_common.h"
#include "./utils.h"

typedef struct particles_s
{
    size_t  n;
    dtype   mass;
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


static void particles_init_empty (particles_t *p);
static void particles_allocate (particles_t  *p, size_t n, dtype mass);
static void particles_free (particles_t *p);
static float dtype_to_storage_float (dtype value, const char *component, size_t i);
static void particles_read_binary (const char  *path, dtype mass, particles_t *p);
static void particles_write_binary (const char  *path, const particles_t *p);
