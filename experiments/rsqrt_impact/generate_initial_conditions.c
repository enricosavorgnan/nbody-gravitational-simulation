/*
 * generate_ic.c
 *
 * Stand-alone C11 generator for equal-mass Plummer-sphere initial conditions.
 * The output is the native-endian binary format read by nbody_direct_serial.c:
 *
 *   8 bytes       magic "NBODYF1\0"
 *   uint64_t      number of particles
 *   N records     x y z vx vy vz as six IEEE single-precision floats
 *
 * The generator samples the usual Plummer radius distribution and an isotropic
 * equilibrium velocity distribution.
 * The velocity scale follows the same dimensionless convention as the solver:
 * total mass M = N * particle_mass and G is user-settable, defaulting to one in
 * code units.
 *
 * The generated coordinates and velocities use dtype internally.  Select it at
 * compile time with -DNBODY_USE_DOUBLE or -DNBODY_USE_FLOAT; the file is always
 * stored as float32 records.

 */



#include "./headers/common.h"

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NBODY_PI
#define NBODY_PI 3.141592653589793238462643383279502884
#endif


typedef struct rng_s
{
  uint64_t  state;
  bool      has_spare;
  dtype     spare;
} rng_t;


// -------------------------------------------------------------------------------
// UTILS
// -------------------------------------------------------------------------------
static void die (const char *format, ...)
{
  va_list  args;

  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fputc ('\n', stderr);
  exit (EXIT_FAILURE);
}


static void print_usage (const char *program    // argv[0]
			 )
{
  fprintf (stderr,
           "usage: %s --n N --output FILE [options]\n"
           "\n"
           "options:\n"
           "  --n N                 number of particles\n"
           "  --output FILE         output binary particle file (%s)\n"
           "  --seed N              RNG seed (default: 1)\n"
           "  --scale X             Plummer scale radius a (default: 1)\n"
           "  --rmax X              optional truncation radius; <=0 disables (default: 0)\n"
           "  --G X                 gravitational constant used for velocities (default: 1)\n"
           "  --mass X              particle mass used for velocities (default: 1)\n"
           "  --help                show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}


/*
 * Parse a positive particle count or seed-like integer from the command line.
 * The function is intentionally strict so that typos in benchmark scripts do
 * not silently produce a different initial condition.
*/
static size_t parse_size (const char *text,     // decimal text to parse
                          const char *name      // option name used in errors
)
{
  char               *endptr;
  unsigned long long  value;

  errno = 0;
  value = strtoull (text, &endptr, 10);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0'))
    die ("invalid integer for %s: %s", name, text);
  if (value > (unsigned long long) SIZE_MAX)
    die ("integer for %s is too large: %s", name, text);

  return (size_t) value;
}

/*
 * Parse a finite floating-point value and cast it to dtype.  All physical
 * parameters pass through this helper before they are used to scale positions
 * or velocities.
 */
static dtype parse_dtype (const char *text,     // decimal text to parse
                          const char *name      // option name used in errors
)
{
  char    *endptr;
  double   value;

  errno = 0;
  value = strtod (text, &endptr);
  if ((errno != 0) || (endptr == text) || (*endptr != '\0') || !isfinite (value))
    die ("invalid floating-point value for %s: %s", name, text);
  if (fabs (value) > (double) DTYPE_MAX_VALUE)
    die ("floating-point value for %s is outside the selected dtype range: %s", name, text);

  return (dtype) value;
}

/*
 * Extract either --key=value or --key value from argv.  This avoids depending on
 * POSIX getopt while still keeping the interface readable in batch scripts.
 */
static const char *option_value (int        *i,       // current argv index, updated on success
                                 int         argc,    // argc from main
                                 char      **argv,    // argv from main
                                 const char *key      // long option name, including "--"
)
{
  const size_t  key_len = strlen (key);
  const char   *arg     = argv[*i];

  if ((strncmp (arg, key, key_len) == 0) && (arg[key_len] == '='))
    return arg + key_len + 1;

  if (strcmp (arg, key) == 0)
    {
      if (*i + 1 >= argc)
        die ("missing value after %s", key);
      *i += 1;
      return argv[*i];
    }

  return NULL;
}

/*
 * Allocate aligned storage for generated coordinates.  The generator writes a
 * file, but keeping the same alignment convention as the solver makes it easy
 * to reuse the arrays in future in-memory tests.
 */
static dtype *allocate_array (size_t      n,       // number of dtype values
                              const char *name     // array name for diagnostics
)
{
  const size_t  bytes = n * sizeof (dtype);
  size_t        padded;
  dtype        *ptr;

  if (n == 0u)
    die ("cannot allocate zero-length array %s", name);
  if (n > SIZE_MAX / sizeof (dtype))
    die ("array %s is too large", name);

  padded = ((bytes + NBODY_ALIGNMENT - 1u) / NBODY_ALIGNMENT) * NBODY_ALIGNMENT;
  ptr = aligned_alloc (NBODY_ALIGNMENT, padded);
  if (ptr == NULL)
    die ("allocation failed for array %s", name);

  return ptr;
}

/*
 * Write exactly nmemb items to a binary stream.
 */
static void checked_fwrite (const void *ptr,       // source buffer
                            size_t      size,      // item size in bytes
                            size_t      nmemb,     // number of items to write
                            FILE       *fp,        // open output stream
                            const char *path,      // file name for diagnostics
                            const char *what       // logical record name
			    )
{
  const size_t  written = fwrite (ptr, size, nmemb, fp);
  if (written != nmemb)
    die ("write error while writing %s to '%s'", what, path);
}


// -------------------------------------------------------------------------------
// RANDOM NUMBERS GENERATION
// -------------------------------------------------------------------------------

/*
 * SplitMix64 step.  It is small, deterministic across platforms, and adequate
 * for generating reproducible teaching initial conditions.
 */
static uint64_t rng_next_u64 (rng_t *rng    // generator state, modified in place
)
{
  uint64_t  z;

  rng->state += UINT64_C (0x9e3779b97f4a7c15);
  z = rng->state;
  z = (z ^ (z >> 30)) * UINT64_C (0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C (0x94d049bb133111eb);
  return z ^ (z >> 31);
}

/*
 * Uniform variate in the open interval (0, 1).  Avoiding exactly 0 and exactly
 * 1 keeps logarithms, inverse CDFs, and rejection samplers away from singular
 * endpoints.
 */
static double rng_uniform_open (rng_t *rng    // generator state, modified in place
)
{
  const uint64_t  bits = rng_next_u64 (rng) >> 11;

  return ((double) bits + 0.5) * (1.0 / 9007199254740992.0);
}


/*
 * Standard normal variate using Box-Muller, with one saved spare value.  Three
 * independent calls produce the Cartesian components of a Maxwellian velocity.
 */
static dtype rng_normal (rng_t *rng    // generator state, modified in place
)
{
  dtype  u1;
  dtype  u2;
  dtype  radius;
  dtype  angle;

  if (rng->has_spare)
    {
      rng->has_spare = false;
      return rng->spare;
    }

  u1 = (dtype) rng_uniform_open (rng);
  u2 = (dtype) rng_uniform_open (rng);
  radius = dtype_sqrt ((dtype) -2.0 * dtype_log (u1));
  angle  = (dtype) (2.0 * NBODY_PI) * u2;

  rng->spare = radius * dtype_sin (angle);
  rng->has_spare = true;
  return radius * dtype_cos (angle);
}


/*
 * Draw a random unit vector on the sphere.  Positions and velocities both use
 * this helper so that angular sampling is isotropic rather than biased by a
 * naive spherical-coordinate choice.
 */
static void random_unit_vector (rng_t *rng,     // generator state, modified in place
                                dtype *ux,      // output x component
                                dtype *uy,      // output y component
                                dtype *uz       // output z component
)
{
  const dtype  cos_theta = (dtype) (2.0 * rng_uniform_open (rng) - 1.0);
  const dtype  phi       = (dtype) (2.0 * NBODY_PI * rng_uniform_open (rng));
  const dtype  sin_theta = dtype_sqrt (dtype_fmax ((dtype) 0.0,
                                                   (dtype) 1.0 - cos_theta * cos_theta));

  *ux = sin_theta * dtype_cos (phi);
  *uy = sin_theta * dtype_sin (phi);
  *uz = cos_theta;
}


/* ··············································································
 * ··············································································
 *
 *  A C T U A L    P A R T I C L E S    G E N E R A T I O N 
 *
 * ··············································································
 */



/*
 * Fill positions with a uniform ball and velocities with a Maxwellian of
 * one-dimensional dispersion sigma.  The centre-of-mass position and velocity
 * are subtracted at the end.  Subtracting the finite-sample position mean may
 * move a few particles just outside the nominal radius by a tiny amount; for a
 * teaching initial condition this is preferable to carrying a net translation.
 */
static void generate_ball_maxwell (size_t  n,                  // number of particles
                                   dtype   ball_radius,        // radius of uniform ball
                                   dtype   sigma,              // 1D velocity dispersion
                                   rng_t  *rng,                // generator state, modified in place
                                   dtype  *restrict x_r,         // output x positions
                                   dtype  *restrict y_r,         // output y positions
                                   dtype  *restrict z_r,         // output z position
                                   dtype  *restrict vx_r,        // output x velocities
                                   dtype  *restrict vy_r,        // output y velocities
                                   dtype  *restrict vz_r,         // output z velocities
                                   dtype  *restrict x_s,         // output x positions
                                   dtype  *restrict y_s,         // output y positions
                                   dtype  *restrict z_s,         // output z positions
                                   dtype  *restrict vx_s,        // output x velocities
                                   dtype  *restrict vy_s,        // output y velocities
                                   dtype  *restrict vz_s         // output z velocities
)
{
  long double  xcm  = 0.0L;
  long double  ycm  = 0.0L;
  long double  zcm  = 0.0L;
  long double  vxcm = 0.0L;
  long double  vycm = 0.0L;
  long double  vzcm = 0.0L;

  for (size_t i = 0u; i < n; ++i)
    {
      dtype  ux;
      dtype  uy;
      dtype  uz;
      dtype  radius;

      radius = ball_radius * dtype_pow ((dtype) rng_uniform_open (rng), (dtype) (1.0 / 3.0));
      random_unit_vector (rng, &ux, &uy, &uz);
      x_r[i] = radius * ux;
      y_r[i] = radius * uy;
      z_r[i] = radius * uz;

      vx_r[i] = sigma * rng_normal (rng);
      vy_r[i] = sigma * rng_normal (rng);
      vz_r[i] = sigma * rng_normal (rng);
    
      x_s[i] = radius * ux;
      y_s[i] = radius * uy;
      z_s[i] = radius * uz;

      vx_s[i] = sigma * rng_normal (rng);
      vy_s[i] = sigma * rng_normal (rng);
      vz_s[i] = sigma * rng_normal (rng);

      xcm  += (long double) x_r[i];
      ycm  += (long double) y_r[i];
      zcm  += (long double) z_r[i];
      vxcm += (long double) vx_r[i];
      vycm += (long double) vy_r[i];
      vzcm += (long double) vz_r[i];
    }

  xcm  /= (long double) n;
  ycm  /= (long double) n;
  zcm  /= (long double) n;
  vxcm /= (long double) n;
  vycm /= (long double) n;
  vzcm /= (long double) n;

  for (size_t i = 0u; i < n; ++i)
    {
      x_r[i]  -= (dtype) xcm;
      y_r[i]  -= (dtype) ycm;
      z_r[i]  -= (dtype) zcm;
      vx_r[i] -= (dtype) vxcm;
      vy_r[i] -= (dtype) vycm;
      vz_r[i] -= (dtype) vzcm;
    
      x_s[i]  -= (dtype) xcm;
      y_s[i]  -= (dtype) ycm;
      z_s[i]  -= (dtype) zcm;
      vx_s[i] -= (dtype) vxcm;
      vy_s[i] -= (dtype) vycm;
      vz_s[i] -= (dtype) vzcm;
    }
}


/*
 * Sample a Plummer-model radius.  For scale radius a, the cumulative mass
 * profile is M(<r)/M = r^3 / (r^2 + a^2)^(3/2), which inverts to
 * r = a / sqrt(u^(-2/3) - 1).  If rmax > 0, the distribution is rejected until
 * it lies inside the requested truncation radius; by default no truncation is
 * applied.
 */
static dtype sample_plummer_radius (rng_t *rng,       // generator state, modified in place
                                    dtype  scale,     // Plummer scale radius a
                                    dtype  rmax       // optional truncation radius, <=0 disables
)
{
  dtype  radius;

  do
    {
      const dtype  u = (dtype) rng_uniform_open (rng);

      radius = scale / dtype_sqrt (dtype_pow (u, (dtype) (-2.0 / 3.0)) - (dtype) 1.0);
    }
  while ((rmax > (dtype) 0.0) && (radius > rmax));

  return radius;
}

/*
 * Sample the dimensionless speed ratio q = v / v_escape for a Plummer sphere.
 * The density is proportional to q^2 (1 - q^2)^(7/2).  The constant 0.1 is a
 * safe envelope for rejection sampling on 0 <= q <= 1.
 */
static dtype sample_plummer_q (rng_t *rng    // generator state, modified in place
)
{
  for (;;)
    {
      const dtype  q       = (dtype) rng_uniform_open (rng);
      const dtype  y       = (dtype) (0.1 * rng_uniform_open (rng));
      const dtype  density = q * q * dtype_pow ((dtype) 1.0 - q * q, (dtype) 3.5);

      if (y <= density)
        return q;
    }
}

/*
 * Generate all particle coordinates and velocities.  The arrays are modified in
 * place.  Velocities are drawn from the equilibrium Plummer distribution for
 * the chosen total mass and scale radius, then the centre-of-mass position and
 * velocity are removed so that the solver does not spend the first few steps
 * translating the whole system through the box.
 */
static void generate_plummer (size_t  n,                 // number of particles
                              dtype   scale,             // Plummer scale radius
                              dtype   rmax,              // optional position truncation
                              dtype   g,                 // gravitational constant
                              dtype   particle_mass,     // mass of one particle
                              rng_t  *rng,               // generator state, modified in place
                              dtype  *restrict x_r,        // output x positions
                              dtype  *restrict y_r,        // output y positions
                              dtype  *restrict z_r,        // output z positions
                              dtype  *restrict vx_r,       // output x velocities
                              dtype  *restrict vy_r,       // output y velocities
                              dtype  *restrict vz_r,        // output z velocities
                              dtype  *restrict x_s,        // output x positions
                              dtype  *restrict y_s,        // output y positions
                              dtype  *restrict z_s,        // output z positions
                              dtype  *restrict vx_s,       // output x velocities
                              dtype  *restrict vy_s,       // output y velocities
                              dtype  *restrict vz_s        // output z velocities
)
{
  const dtype  total_mass = (dtype) n * particle_mass;
  long double  xcm        = 0.0L;
  long double  ycm        = 0.0L;
  long double  zcm        = 0.0L;
  long double  vxcm       = 0.0L;
  long double  vycm       = 0.0L;
  long double  vzcm       = 0.0L;

  if (!dtype_isfinite (total_mass) || !(total_mass > (dtype) 0.0))
    die ("total mass is not finite in the selected dtype");

  for (size_t i = 0u; i < n; ++i)
    {
      dtype  ux;
      dtype  uy;
      dtype  uz;
      dtype  radius;
      dtype  q;
      dtype  psi;
      dtype  speed;

      radius = sample_plummer_radius (rng, scale, rmax);
      random_unit_vector (rng, &ux, &uy, &uz);
      x_r[i] = radius * ux;
      y_r[i] = radius * uy;
      z_r[i] = radius * uz;
      x_s[i] = radius * ux;
      y_s[i] = radius * uy;
      z_s[i] = radius * uz;

      q     = sample_plummer_q (rng);
      psi   = g * total_mass / dtype_sqrt (radius * radius + scale * scale);
      speed = q * dtype_sqrt ((dtype) 2.0 * psi);
      random_unit_vector (rng, &ux, &uy, &uz);
      vx_r[i] = speed * ux;
      vy_r[i] = speed * uy;
      vz_r[i] = speed * uz;

      vx_s[i] = speed * ux;
      vy_s[i] = speed * uy;
      vz_s[i] = speed * uz;

      xcm  += (long double) x_r[i];
      ycm  += (long double) y_r[i];
      zcm  += (long double) z_r[i];
      vxcm += (long double) vx_r[i];
      vycm += (long double) vy_r[i];
      vzcm += (long double) vz_r[i];
    }

  xcm  /= (long double) n;
  ycm  /= (long double) n;
  zcm  /= (long double) n;
  vxcm /= (long double) n;
  vycm /= (long double) n;
  vzcm /= (long double) n;

  for (size_t i = 0u; i < n; ++i)
    {
      x_r[i]  -= (dtype) xcm;
      y_r[i]  -= (dtype) ycm;
      z_r[i]  -= (dtype) zcm;
      vx_r[i] -= (dtype) vxcm;
      vy_r[i] -= (dtype) vycm;
      vz_r[i] -= (dtype) vzcm;
    
      x_s[i]  -= (dtype) xcm;
      y_s[i]  -= (dtype) ycm;
      z_s[i]  -= (dtype) zcm;
      vx_s[i] -= (dtype) vxcm;
      vy_s[i] -= (dtype) vycm;
      vz_s[i] -= (dtype) vzcm;
    }
}



/* ··············································································
 * ··············································································
 *
 *  V E R I F Y    D A T A    S A N I T Y  
 *
 * ··············································································
 */


static int verify_sanity ( const size_t  n,               // number of particles
			   const dtype  *x_r,               // x positions
			   const dtype  *y_r,               // y positions
			   const dtype  *z_r,               // z positions
			   const dtype  *vx_r,              // x velocities
			   const dtype  *vy_r,              // y velocities
			   const dtype  *vz_r,               // z velocities
			   const dtype  *x_s,               // x positions
         const dtype  *y_s,               // y positions
         const dtype  *z_s,               // z positions
         const dtype  *vx_s,              // x velocities
         const dtype  *vy_s,              // y velocities
         const dtype  *vz_s               // z velocities
			   )
{
  uint64_t failures = 0;
  
  for (size_t i = 0u; i < n; ++i)
  {
    failures += (!isfinite (x_r[i]) || (fabs (x_r[i]) > DTYPE_MAX_VALUE));
    failures += (!isfinite (x_s[i]) || (fabs (x_s[i]) > DTYPE_MAX_VALUE));
  }
  if ( failures ) {
    printf ( "%zu x component cannot be stored as a finite float", failures );
    return 1; }
  
  failures = 0;
  for (size_t i = 0u; i < n; ++i)
  {
    failures += (!isfinite (y_r[i]) || (fabs (y_r[i]) > DTYPE_MAX_VALUE));
    failures += (!isfinite (y_s[i]) || (fabs (y_s[i]) > DTYPE_MAX_VALUE));
  }
  if ( failures ) {
    printf ( "%zu y component cannot be stored as a finite float", failures );
    return 1; }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
  {
    failures += (!isfinite (z_r[i]) || (fabs (z_r[i]) > DTYPE_MAX_VALUE));
    failures += (!isfinite (z_s[i]) || (fabs (z_s[i]) > DTYPE_MAX_VALUE));
  }
  if ( failures ) {
    printf ( "%zu z component cannot be stored as a finite float", failures );
    return 1; }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
  {
    failures += (!isfinite (vx_r[i]) || (fabs (vx_r[i]) > DTYPE_MAX_VALUE));
    failures += (!isfinite (vx_s[i]) || (fabs (vx_s[i]) > DTYPE_MAX_VALUE));
  }
  if ( failures ) {
    printf ( "%zu vx component cannot be stored as a finite float", failures );
    return 1; }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
  {
    failures += (!isfinite (vy_r[i]) || (fabs (vy_r[i]) > DTYPE_MAX_VALUE));
    failures += (!isfinite (vy_s[i]) || (fabs (vy_s[i]) > DTYPE_MAX_VALUE));
  }
  if ( failures ) {
    printf ( "%zu vy component cannot be stored as a finite float", failures );
    return 1; }

  failures = 0;
  for (size_t i = 0u; i < n; ++i)
  {
    failures += (!isfinite (vz_r[i]) || (fabs (vz_r[i]) > DTYPE_MAX_VALUE));
    failures += (!isfinite (vz_s[i]) || (fabs (vz_s[i]) > DTYPE_MAX_VALUE));
  }
  if ( failures ) {
    printf ( "%zu vz component cannot be stored as a finite float", failures );
    return 1; }

  return 0;
}


/* ··············································································
 * ··············································································
 *
 *  W R I T E    P A R T I C L E S    F I L E
 *
 * ··············································································
 */



/*
 * Write the generated initial conditions to disk.  The file contains only the
 * compact binary header and float32 particle records; run metadata belong in
 * the benchmark script or reproducibility manifest, not in this minimal binary
 * exchange format.
 */
static void write_particles_binary (const char  *path,            // output file path
                                    size_t       n,               // number of particles
                                    const dtype *x_r,               // x positions
                                    const dtype *y_r,               // y positions
                                    const dtype *z_r,               // z positions
                                    const dtype *vx_r,              // x velocities
                                    const dtype *vy_r,              // y velocities
                                    const dtype *vz_r,               // z velocities
                                    const dtype *x_s,               // x positions
                                    const dtype *y_s,               // y positions
                                    const dtype *z_s,               // z positions
                                    const dtype *vx_s,              // x velocities
                                    const dtype *vy_s,              // y velocities
                                    const dtype *vz_s    
				    )
{
  FILE      *fp;
  uint64_t   n64 = (uint64_t) n;

  if ((size_t) n64 != n)
    die ("particle count cannot be represented in the binary header");

  int uncorrect_data = verify_sanity ( n, x_r, y_r, z_r, vx_r, vy_r, vz_r, x_s, y_s, z_s, vx_s, vy_s, vz_s );

  if ( !uncorrect_data )
    {
      fp = fopen (path, "wb");
      if (fp == NULL)
	die ("cannot open output file '%s'", path);
  
      checked_fwrite (nbody_binary_magic, sizeof nbody_binary_magic[0],
		      NBODY_BINARY_MAGIC_SIZE, fp, path, "binary magic");
      checked_fwrite (&n64, sizeof n64, 1u, fp, path, "particle count");
      
      for (size_t i = 0u; i < n; ++i)
	{
	  float  record[NBODY_BINARY_COMPONENTS];
	  
	  record[0] =  x_r[i];
	  record[1] =  y_r[i];
	  record[2] =  z_r[i];
	  record[3] = vx_r[i];
	  record[4] = vy_r[i];
	  record[5] = vz_r[i];
    record[6] =  x_s[i];
    record[7] =  y_s[i];
    record[8] =  z_s[i];
    record[9] = vx_s[i];
    record[10] = vy_s[i];
    record[11] = vz_s[i];
	  
	  checked_fwrite (record, sizeof record[0], NBODY_BINARY_COMPONENTS,
			  fp, path, "particle record");
	}

      if (fclose (fp) != 0)
	die ("error while closing output file '%s'", path);
    }
  else
    printf( "Some data are not representable as finite floating points\n" );
}



int main (int argc, char **argv)
{
  const char  *output_path   = NULL;
  int          model         = PLUMMER_SPHERE;
  size_t       n             = 0u;
  uint64_t     seed          = 1u;
  dtype        ball_radius   = (dtype) 1.0;      // for MAXWELL_BALL
  dtype        sigma         = (dtype) -1.0;     // for MAXWELL_BALL
  dtype        scale         = (dtype) 1.0;      // for PLUMMER_SPHERE
  dtype        rmax          = (dtype) 0.0;      // for PLUMMER_SPHERE
  dtype        g             = (dtype) 1.0;
  dtype        particle_mass = (dtype) 1.0;
  bool         sigma_auto;                       // for MAXWELL_BALL
  dtype       *x_r;
  dtype       *y_r;
  dtype       *z_r;
  dtype       *vx_r;
  dtype       *vy_r;
  dtype       *vz_r;
  dtype       *x_s;
  dtype       *y_s;
  dtype       *z_s;
  dtype       *vx_s;
  dtype       *vy_s;
  dtype       *vz_s;
  rng_t        rng;
  int          argi;

  for (argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--model")) != NULL)
        model = parse_size (value, "--model");
      else if ((value = option_value (&argi, argc, argv, "--n")) != NULL)
        n = parse_size (value, "--n");
      else if ((value = option_value (&argi, argc, argv, "--output")) != NULL)
        output_path = value;
      else if ((value = option_value (&argi, argc, argv, "--seed")) != NULL)
        seed = (uint64_t) parse_size (value, "--seed");
      else if ((value = option_value (&argi, argc, argv, "--scale")) != NULL)
        scale = parse_dtype (value, "--scale");
      else if ((value = option_value (&argi, argc, argv, "--rmax")) != NULL)
        rmax = parse_dtype (value, "--rmax");
      else if ((value = option_value (&argi, argc, argv, "--radius")) != NULL)
        ball_radius = parse_dtype (value, "--radius");
      else if ((value = option_value (&argi, argc, argv, "--sigma")) != NULL)
        sigma = parse_dtype (value, "--sigma");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        particle_mass = parse_dtype (value, "--mass");
      else if (strcmp (argv[argi], "--help") == 0)
        {
          print_usage (argv[0]);
          return EXIT_SUCCESS;
        }
      else
        {
          print_usage (argv[0]);
          die ("unknown option: %s", argv[argi]);
        }
    }

  if (n == 0u)
    {
      print_usage (argv[0]);
      die ("missing or invalid --n N");
    }
  if (output_path == NULL)
    {
      print_usage (argv[0]);
      die ("missing required --output FILE");
    }
  if (!(scale > (dtype) 0.0))
    die ("--scale must be positive");
  if (!(g > (dtype) 0.0))
    die ("--G must be positive");
  if (!(particle_mass > (dtype) 0.0))
    die ("--mass must be positive");
  if (!(ball_radius > (dtype) 0.0))
    die ("--radius must be positive");
  sigma_auto = (sigma < (dtype) 0.0);
  if (sigma_auto)
    {
      const dtype  total_mass = (dtype) n * particle_mass;

      if (!dtype_isfinite (total_mass) || !(total_mass > (dtype) 0.0))
        die ("total mass is not finite in the selected dtype");
      sigma = dtype_sqrt (g * total_mass / ((dtype) 5.0 * ball_radius));
    }
  if (!(sigma >= (dtype) 0.0) || !dtype_isfinite (sigma))
    die ("--sigma must be non-negative and finite, or negative to request the auto value");

  
  x_r  = allocate_array (n, "x_r");
  y_r  = allocate_array (n, "y_r");
  z_r  = allocate_array (n, "z_r");
  vx_r = allocate_array (n, "vx_r");
  vy_r = allocate_array (n, "vy_r");
  vz_r = allocate_array (n, "vz_r");
  x_s  = allocate_array (n, "x_s");
  y_s  = allocate_array (n, "y_s");
  z_s  = allocate_array (n, "z_s");
  vx_s = allocate_array (n, "vx_s");
  vy_s = allocate_array (n, "vy_s");
  vz_s = allocate_array (n, "vz_s");

  rng.state     = seed;
  rng.has_spare = false;
  rng.spare     = (dtype) 0.0;

  if ( model == PLUMMER_SPHERE)
    generate_plummer (n, scale, rmax, g, particle_mass, &rng, x_r, y_r, z_r, vx_r, vy_r, vz_r, x_s, y_s, z_s, vx_s, vy_s, vz_s);

  else
    generate_ball_maxwell (n, ball_radius, sigma, &rng, x_r, y_r, z_r, vx_r, vy_r, vz_r, x_s, y_s, z_s, vx_s, vy_s, vz_s);
  
  write_particles_binary (output_path, n, x_r, y_r, z_r, vx_r, vy_r, vz_r, x_s, y_s, z_s, vx_s, vy_s, vz_s);

  free (x_r);
  free (y_r);
  free (z_r);
  free (vx_r);
  free (vy_r);
  free (vz_r);
  free (x_s);
  free (y_s);
  free (z_s);
  free (vx_s);
  free (vy_s);
  free (vz_s);


  return EXIT_SUCCESS;
}
