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
  p->x_r    = NULL;
  p->y_r    = NULL;
  p->z_r    = NULL;
  p->vx_r   = NULL;
  p->vy_r   = NULL;
  p->vz_r   = NULL;
  p->ax_r   = NULL;
  p->ay_r   = NULL;
  p->az_r   = NULL;
  p->x_s    = NULL;
  p->y_s    = NULL;
  p->z_s    = NULL;
  p->vx_s   = NULL;
  p->vy_s   = NULL;
  p->vz_s   = NULL;
  p->ax_s   = NULL;
  p->ay_s   = NULL;
  p->az_s   = NULL;
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
  p->x_r    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->y_r    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->z_r    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vx_r   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vy_r   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vz_r   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ax_r   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ay_r   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->az_r   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->x_s    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->y_s    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->z_s    = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vx_s   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vy_s   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->vz_s   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ax_s   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->ay_s   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
  p->az_s   = checked_aligned_alloc (bytes, NBODY_ALIGNMENT);
}


/*
 * Release all particle arrays and return the container to the empty state.  No
 * simulation data survive this call.
 */
void particles_free (particles_t *p    // container to release
			    )
{
  free (p->x_r);
  free (p->y_r);
  free (p->z_r);
  free (p->vx_r);
  free (p->vy_r);
  free (p->vz_r);
  free (p->ax_r);
  free (p->ay_r);
  free (p->az_r);
  free (p->x_s);
  free (p->y_s);
  free (p->z_s);
  free (p->vx_s);
  free (p->vy_s);
  free (p->vz_s);
  free (p->ax_s);
  free (p->ay_s);
  free (p->az_s);
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
          !isfinite ((double) record[4]) || !isfinite ((double) record[5]) ||
          !isfinite ((double) record[6]) || !isfinite ((double) record[7]) ||
          !isfinite ((double) record[8]) || !isfinite ((double) record[9]) ||
          !isfinite ((double) record[10]) || !isfinite ((double) record[11])
          )
        die ("non-finite particle value in '%s' at index %zu", path, i);

      p->x_r[i]  = (dtype) record[0];
      p->y_r[i]  = (dtype) record[1];
      p->z_r[i]  = (dtype) record[2];
      p->vx_r[i] = (dtype) record[3];
      p->vy_r[i] = (dtype) record[4];
      p->vz_r[i] = (dtype) record[5];
      p->x_s[i]  = (dtype) record[6];
      p->y_s[i]  = (dtype) record[7];
      p->z_s[i]  = (dtype) record[8];
      p->vx_s[i] = (dtype) record[9];
      p->vy_s[i] = (dtype) record[10];
      p->vz_s[i] = (dtype) record[11];
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

      record[0] = dtype_to_storage_float (p->x_r[i],  "x_r",  i);
      record[1] = dtype_to_storage_float (p->y_r[i],  "y_r",  i);
      record[2] = dtype_to_storage_float (p->z_r[i],  "z_r",  i);
      record[3] = dtype_to_storage_float (p->vx_r[i], "vx_r", i);
      record[4] = dtype_to_storage_float (p->vy_r[i], "vy_r", i);
      record[5] = dtype_to_storage_float (p->vz_r[i], "vz_r", i);
      record[6] = dtype_to_storage_float (p->x_s[i],  "x_s",  i);
      record[7] = dtype_to_storage_float (p->y_s[i],  "y_s",  i);
      record[8] = dtype_to_storage_float (p->z_s[i],  "z_s",  i);
      record[9] = dtype_to_storage_float (p->vx_s[i], "vx_s", i);
      record[10] = dtype_to_storage_float (p->vy_s[i], "vy_s", i);
      record[11] = dtype_to_storage_float (p->vz_s[i], "vz_s", i);
      checked_fwrite (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
                      fp, path, "particle record");
    }

  if (fclose (fp) != 0)
    die ("error while closing output file '%s'", path);
}