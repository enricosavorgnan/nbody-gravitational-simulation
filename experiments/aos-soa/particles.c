// Particles Stuff

#include "./headers/particles.h"


// Allocate the AoS storage used by the solver.
particle_t* particle_allocate (size_t n, dtype mass)
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
void particle_free (particle_t *p)
{
  free(p);
}

void particle_read_binary (const char *path, dtype mass, particle_t **p, size_t *n_out)
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

  *p = particle_allocate (n, mass);
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

void particle_write_binary (const char *path, const particle_t *p, size_t n)
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


void particles_init_empty (particles_t *p
				  )
{
  p->n    = 0u;
  p->mass = (dtype) 1.0;
  p->x    = NULL;
  p->y    = NULL;
  p->z    = NULL;
  p->vx   = NULL;
  p->vy   = NULL;
  p->vz   = NULL;
  p->ax   = NULL;
  p->ay   = NULL;
  p->az   = NULL;
}



void particles_allocate (particles_t  *p,       // output container
                                size_t        n,       // number of particles
                                dtype         mass     // mass of each particle
				)
{
  const size_t  bytes = n * sizeof (dtype);

  if (n == 0u)
    die ("the number of particles must be positive");
  if (n > SIZE_MAX / sizeof (dtype))
    die ("particle count is too large");
  if (!(mass > (dtype) 0.0) || !dtype_isfinite (mass))
    die ("particle mass must be positive and finite");

  particles_init_empty (p);
  p->n    = n;
  p->mass = mass;
  p->x    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->y    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->z    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vx   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vy   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vz   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ax   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ay   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->az   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
}


void particles_free (particles_t *p    // container to release
			    )
{
  free (p->x);
  free (p->y);
  free (p->z);
  free (p->vx);
  free (p->vy);
  free (p->vz);
  free (p->ax);
  free (p->ay);
  free (p->az);
  particles_init_empty (p);
}



void particles_read_binary (const char  *path,       // input file path
                                   dtype        mass,       // mass assigned to each particle
                                   particles_t *p           // output particle container
				   )
{
  FILE           *fp;
  unsigned char   magic[NBODY_BINARY_MAGIC_SIZE];
  uint64_t        n64;
  size_t          n;
  size_t          i;

  fp = fopen (path, "rb");
  if (fp == NULL)
    die ("cannot open input file '%s'", path);

  checked_fread (magic, sizeof magic[0], NBODY_BINARY_MAGIC_SIZE,
                 fp, path, "binary magic");
  if (memcmp (magic, nbody_binary_magic, NBODY_BINARY_MAGIC_SIZE) != 0)
    die ("input file '%s' is not an %s file", path, NBODY_BINARY_VERSION_TEXT);

  checked_fread (&n64, sizeof n64, 1u, fp, path, "particle count");
  if ((n64 == 0u) || (n64 > (uint64_t) SIZE_MAX))
    die ("invalid particle count in '%s'", path);
  n = (size_t) n64;

  particles_allocate (p, n, mass);

  for (i = 0u; i < n; ++i)
    {
      float  record[NBODY_BINARY_COMPONENTS];

      checked_fread (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                     fp, path, "particle record");

      /*
       * this check is also very costly made like that.
       * either optimize or render it optional for some debugging mode
       */
      if (!isfinite ((double) record[0]) || !isfinite ((double) record[1]) ||
          !isfinite ((double) record[2]) || !isfinite ((double) record[3]) ||
          !isfinite ((double) record[4]) || !isfinite ((double) record[5]))
        die ("non-finite particle value in '%s' at index %zu", path, i);

      p->x[i]  = (dtype) record[0];
      p->y[i]  = (dtype) record[1];
      p->z[i]  = (dtype) record[2];
      p->vx[i] = (dtype) record[3];
      p->vy[i] = (dtype) record[4];
      p->vz[i] = (dtype) record[5];
    }

  if (fclose (fp) != 0)
    die ("error while closing input file '%s'", path);
}


void particles_write_binary (const char        *path,       // output file path
                                    const particles_t *p           // particle state to write
				    )
{
  FILE          *fp;
  const size_t   n   = p->n;
  uint64_t       n64 = (uint64_t) n;

  if ((size_t) n64 != n)
    die ("particle count cannot be represented in the binary header");

  fp = fopen (path, "wb");
  if (fp == NULL)
    die ("cannot open output file '%s'", path);

  checked_fwrite (nbody_binary_magic, sizeof nbody_binary_magic[0],
                  NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
  checked_fwrite (&n64, sizeof n64, 1u, fp, path, "particle count");

  for (size_t i = 0u; i < n; ++i)
    {
      float  record[NBODY_BINARY_COMPONENTS];

      record[0] = dtype_to_storage_float (p->x[i],  "x",  i);
      record[1] = dtype_to_storage_float (p->y[i],  "y",  i);
      record[2] = dtype_to_storage_float (p->z[i],  "z",  i);
      record[3] = dtype_to_storage_float (p->vx[i], "vx", i);
      record[4] = dtype_to_storage_float (p->vy[i], "vy", i);
      record[5] = dtype_to_storage_float (p->vz[i], "vz", i);
      checked_fwrite (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                      fp, path, "particle record");
    }

  if (fclose (fp) != 0)
    die ("error while closing output file '%s'", path);
}