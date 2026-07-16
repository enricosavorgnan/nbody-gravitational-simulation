/*
 * A collection of utility functions for the N-body simulation
 */

#include "./headers/utils.h"

/* ======================================================================================== */
/*
   : ------------------------------------------------------ :
   :  UTILITIES                                             :
   : ------------------------------------------------------ :
 */


static void die (const char *format, ...)
{
  va_list  args;

  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fputc ('\n', stderr);
  exit (EXIT_FAILURE);
}

/*
 * Parse a size_t command-line value.
 * All user-facing quantities that count particles or steps pass through
 * this function so that overflow and malformed input fail early, before
 * any allocation or simulation state is modified.
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
 * Parse a finite floating-point command-line value and cast it to dtype.
 * The parser reads through double because `strtof` and `strtod` differ only in final
 * rounding for the ranges used here; the explicit range check keeps float-mode
 * builds from silently accepting values that dtype cannot represent.
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
 * Return the value associated with either "--key value" or "--key=value".
 * The caller passes the loop index by address so that the separated-value
 * form consumes the following argv entry exactly once.
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
 * Allocate a cache-line aligned block.  Alignment is not required for scalar
 * correctness, but it makes the serial skeleton a better starting point for
 * vectorisation and OpenMP first-touch experiments.
 */
static void *checked_aligned_alloc (size_t  nbytes,      // requested useful bytes
                                    size_t  alignment    // power-of-two alignment
				    )
{
  void   *ptr;
  size_t  padded;

  if (nbytes == 0u)
    die ("attempted zero-byte allocation");
  if (alignment == 0u)
    die ("invalid zero alignment");
  if (nbytes > SIZE_MAX - alignment)
    die ("allocation size overflow");

  padded = ((nbytes + alignment - 1u) / alignment) * alignment;
  ptr = aligned_alloc (alignment, padded);
  if (ptr == NULL)
    die ("aligned_alloc failed for %zu bytes", padded);

  return ptr;
}

/*
 * Read exactly nmemb items from a binary stream.  Centralizing the check avoids
 * partial binary records being mistaken for valid particles, which is otherwise
 * easy to do when replacing a line-oriented ASCII reader with fread.
 */
static void checked_fread (void       *ptr,       // destination buffer
                           const size_t size,      // item size in bytes
                           size_t      nmemb,     // number of items expected
                           FILE       *fp,        // open input stream
                           const char *path,      // file name for diagnostics
                           const char *what       // logical record name
			   )
{
  const size_t  got = fread (ptr, size, nmemb, fp);

  if (got != nmemb)
    {
      if (ferror (fp))
        die ("read error while reading %s from '%s'", what, path);
      die ("short file while reading %s from '%s'", what, path);
    }
}

/*
 * Write exactly nmemb items to a binary stream.  All output paths go through
 * this helper so that disk-full and permission errors are reported at the point
 * where the data loss happens, not later in a benchmark script.
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

