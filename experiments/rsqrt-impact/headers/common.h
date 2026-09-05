#ifndef NBODY_COMMON_H
#define NBODY_COMMON_H

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef NBODY_ALIGNMENT
#define NBODY_ALIGNMENT 64u
#endif

#ifndef N_RSQRT_LOOP
#define N_RSQRT_LOOP 3
#endif

#define NBODY_BINARY_MAGIC_SIZE 8u
#define NBODY_BINARY_COMPONENTS 6u
#define NBODY_BINARY_VERSION_TEXT "nbody-f32-v1"

#define PLUMMER_SPHERE 0
#define MAXWELL_BALL   1

static const unsigned char  nbody_binary_magic[NBODY_BINARY_MAGIC_SIZE] =
  { 'N', 'B', 'O', 'D', 'Y', 'F', '1', '\0' };

#if defined (NBODY_USE_FLOAT) && defined (NBODY_USE_DOUBLE)
#error "define only one of NBODY_USE_FLOAT and NBODY_USE_DOUBLE"
#endif

#if defined (NBODY_USE_FLOAT)
typedef float  dtype;
#define DTYPE_NAME "float"
#define DTYPE_MAX_VALUE FLT_MAX
#define DTYPE_MIN_NORMAL FLT_MIN
#define DTYPE_PRINTF_FORMAT "%.9g"

static inline dtype dtype_sqrt (dtype x)
{
  return sqrtf (x);
}

// The following is the official implementation of
// Fast Inverse Square Root, taken from Wikipedia:
// https://en.wikipedia.org/wiki/Fast_inverse_square_root
// it should be correct up to the 11th bit
static inline dtype dtype_rsqrt(dtype x)
{
  int32_t i;
  float y = x;

  i = *(int32_t *) &y;
  i = 0x5f3759df - (i >> 1);
  y = *(float *) &i;

  for (int l = 0; l < N_RSQRT_LOOP; l++)
  {
    y = y * ((dtype) 1.5 - (dtype) 0.5 * x * y * y);
  }
  return y;
}

static inline dtype dtype_pow (dtype x,
                               dtype y)
{
  return powf (x, y);
}

static inline dtype dtype_sin (dtype x)
{
  return sinf (x);
}

static inline dtype dtype_cos (dtype x)
{
  return cosf (x);
}

static inline dtype dtype_log (dtype x)
{
  return logf (x);
}

static inline dtype dtype_fabs (dtype x)
{
  return fabsf (x);
}

static inline dtype dtype_fmax (dtype x,
                                dtype y)
{
  return fmaxf (x, y);
}

#else
typedef double dtype;
#define DTYPE_NAME "double"
#define DTYPE_MAX_VALUE DBL_MAX
#define DTYPE_MIN_NORMAL DBL_MIN
#define DTYPE_PRINTF_FORMAT "%.17g"

static inline dtype dtype_sqrt (dtype x)
{
  return sqrt (x);
}

static inline dtype dtype_rsqrt (dtype x)
{
  union {
    int64_t i;
    double f;
  } u;

  u.f = x;
  u.i = 0x5fe6eb50c7b537a9 - (u.i >> 1);

  for (int l = 0; l < N_RSQRT_LOOP; l++)
  {
    u.f = u.f * ((dtype) 1.5 - (dtype) 0.5 * x * u.f * u.f );
  }
  return u.f;
}

static inline dtype dtype_pow (dtype x,
                               dtype y)
{
  return pow (x, y);
}

static inline dtype dtype_sin (dtype x)
{
  return sin (x);
}

static inline dtype dtype_cos (dtype x)
{
  return cos (x);
}

static inline dtype dtype_log (dtype x)
{
  return log (x);
}

static inline dtype dtype_fabs (dtype x)
{
  return fabs (x);
}

static inline dtype dtype_fmax (dtype x,
                                dtype y)
{
  return fmax (x, y);
}

#endif

static inline bool dtype_isfinite (dtype x)
{
  return isfinite ((double) x);
}

#endif
