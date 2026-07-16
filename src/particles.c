// Particles Stuff

#include "./headers/particles.h"

/* ======================================================================================== */
/*
   : ------------------------------------------------------ :
   :  PARTICLES ALLOCATION                                  :
   : I/O                                                    :
   : ------------------------------------------------------ :
 */


/*
 * Initialise an empty particle container.  This function does not allocate; it
 * simply gives every pointer a known value so that particles_free can safely be
 * called after a partial failure path.
 */
void particles_init_empty (particles_t *p    // particle container to initialise
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


/*
 * Allocate the SoA storage used by the solver.  Positions, velocities, and
 * accelerations are separate arrays, not an array of structs, because the
 * direct kernel only needs streams of x/y/z coordinates and accumulators.  This
 * layout is also the natural one for later SIMD and MPI ring-buffer work.
 */
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


/*
 * Release all particle arrays and return the container to the empty state.  No
 * simulation data survive this call.
 */
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


/*
 * Cast one physical quantity to the on-disk type.
 * No finite values and overflows should still fail loudly.
 * How do we deal with that?
 * Check the initial condition generator for a more performant
 * implementation.
 *
 * NOTE: may this be a bottleneck in the I/O ? what your profiling says?
 *       if so, you may consider making this optional only when a "debugging mode"
 *       is set, and to expand to a simple cast otherwise;
 *       Optionally, these sanity checks should be made as a loop over particles that
 *       accumulate failure counter, instead on per-particle function call
 */
float dtype_to_storage_float (dtype       value,       // value to store
                                     const char *component,   // component name for diagnostics
                                     size_t      i            // particle index for diagnostics
				     )
{
  const double  as_double = (double) value;

  if (!isfinite (as_double) || (fabs (as_double) > (double) FLT_MAX))
    die ("particle %zu component %s cannot be stored as a finite float", i, component);

  return (float) value;
}


/*
 * Load particle coordinates and velocities from the binary file.
 * The on-disk values are single precision, then converted to dtype so the same
 * initial-condition file can be used for both float and double solver builds.
 * Acceleration arrays are left uninitialised because every force evaluation
 * overwrites them.
 */
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


/*
 * Write the current particle state in the same binary format accepted by the
 * reader.  Conversion to single precision is done explicitly record by record;
 * this is simple rather than maximally fast;
 * check in the code for initial condition generator
 */
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