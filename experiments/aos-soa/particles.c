// Particles Stuff

#include "./headers/particles.h"


// Allocate the AoS storage used by the solver.
particle_t* particles_allocate (size_t n, dtype mass)
{
  if (n == 0u)
    die ("the number of particles must be positive");
  if (n > SIZE_MAX / sizeof (dtype))
    die ("particle count is too large");
  if (!(mass > (dtype) 0.0) || !dtype_isfinite (mass))
    die ("particle mass must be positive and finite");

  // Initialize an array of particles (AoS).
  particle_t *particles = checked_aligned_alloc (n * sizeof (particle_t), NBODY_ALIGNMENT);
  for (size_t i = 0u; i < n; ++i)
  {
    particles[i].mass = mass;
    particles[i].x = particles[i].y = particles[i].z = (dtype) 0.0;
    particles[i].vx = particles[i].vy = particles[i].vz = (dtype) 0.0;
    particles[i].ax = particles[i].ay = particles[i].az = (dtype) 0.0;
  }

  return particles;
}

float dtype_to_storage_float (dtype value, const char *component, size_t i)
{
  const double  as_double = (double) value;

  if (!isfinite (as_double) || (fabs (as_double) > (double) FLT_MAX))
    die ("particle %zu component %s cannot be stored as a finite float", i, component);

  return (float) value;
}

// Release all particle arrays and return the container to the empty state.
void particles_free (particle_t *p)
{
  free(p);
}

void particles_read_binary (const char *path, dtype mass, particle_t **p, size_t *n_out)
{
  FILE          *fp;
  unsigned char  magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t       n64;
  size_t         n;

  fp = fopen (path, "rb");
  if (fp == NULL)
    die ("cannot open input file '%s'", path);

  checked_fread (magic, sizeof magic[0], NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
  if (memcmp (magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die ("input file '%s' is not an %s file", path, NBODY_BINARY_VERSION_TEXT);

  checked_fread (&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((n64 == 0u) || (n64 > (uint64_t) SIZE_MAX))
    die ("invalid particle count in '%s'", path);
  n = (size_t) n64;

  *p = particles_allocate (n, mass);
  *n_out = n;

  for (size_t i = 0u; i < n; ++i)
  {
    float record[NBODY_BINARY_COMPONENTS];
    checked_fread (record, sizeof record[0], NBODY_BINARY_COMPONENTS, fp, path, "particle record");

    (*p)[i].x  = (dtype) record[0];
    (*p)[i].y  = (dtype) record[1];
    (*p)[i].z  = (dtype) record[2];
    (*p)[i].vx = (dtype) record[3];
    (*p)[i].vy = (dtype) record[4];
    (*p)[i].vz = (dtype) record[5];
  }

  if (fclose (fp) != 0)
    die ("error while closing input file '%s'", path);
}

void particles_write_binary (const char *path, const particle_t *p, size_t n)
{
  FILE     *fp;
  uint64_t  n64 = (uint64_t) n;

  fp = fopen (path, "wb");
  if (fp == NULL)
    die ("cannot open output file '%s'", path);

  checked_fwrite (nbody_binary_magic, sizeof nbody_binary_magic[0],
                  NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
  checked_fwrite (&n64, sizeof n64, 1u, fp, path, "particle count");

  for (size_t i = 0u; i < n; ++i)
  {
    float record[NBODY_BINARY_COMPONENTS];

    record[0] = dtype_to_storage_float (p[i].x,  "x",  i);
    record[1] = dtype_to_storage_float (p[i].y,  "y",  i);
    record[2] = dtype_to_storage_float (p[i].z,  "z",  i);
    record[3] = dtype_to_storage_float (p[i].vx, "vx", i);
    record[4] = dtype_to_storage_float (p[i].vy, "vy", i);
    record[5] = dtype_to_storage_float (p[i].vz, "vz", i);

    checked_fwrite (record, sizeof record[0], NBODY_BINARY_COMPONENTS, fp, path, "particle record");
  }

  if (fclose (fp) != 0)
    die ("error while closing output file '%s'", path);
}