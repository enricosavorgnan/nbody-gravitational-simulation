/*
 * nbody_direct_serial.c
 *
 * Serial C11 reference implementation for the direct gravitational N-body
 * exercise.  The O(N^2) force kernel is simple;
 * optimizing the kernel is part of the assignment.
 * The kernel is the natural place to discuss SoA data layout, cache locality,
 *  Newton's third law, accumulator dependency chains, rsqrt, OpenMP
 *  reductions, and MPI ring-shift communication.
 *
 * Units are dimensionless.  By default, G = 1, particle mass = 1, and the
 * softened potential is
 *
 *   phi_ij = - G m^2 / sqrt(|r_i-r_j|^2 + eps^2).
 *
 * Binary input/output file format, native endian:
 *
 *   8 bytes       magic "NBODYF1\0"
 *   uint64_t      number of particles
 *   N records     x y z vx vy vz as six IEEE single-precision floats
 *
 * The simulation arithmetic uses dtype, selected at compile time:
 *
 *   -DNBODY_USE_DOUBLE    default double-precision arithmetic
 *   -DNBODY_USE_FLOAT     single-precision arithmetic
 *
 * Files are intentionally still stored in single precision, independently of
 * dtype.
 *
 */

#include <errno.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./headers/common.h"
#include "./headers/profiling.h"
#include "./headers/utils.h"
#include "./headers/particles.h"
#include "./headers/integration.h"


/*
 * Print a compact command-line reference.
 * Defaults are chosen for > small test < runs
 */
static void print_usage (const char *program    // argv[0]
			 )
{
  fprintf (stderr,
           "usage: %s --input FILE [options]\n"
           "\n"
           "options:\n"
           "  --input FILE              input binary particle file (%s)\n"
           "  --output FILE             optional final-state binary file\n"
           "  --nsteps N                number of DKD steps (default: 10)\n"
           "  --dt X                    time step (default: 0.001)\n"
           "  --eps X                   softening length (default: 0.01)\n"
           "  --G X                     gravitational constant (default: 1)\n"
           "  --mass X                  particle mass (default: 1)\n"
           "  --energy-every N          diagnostic period in steps (default: 1)\n"
           "  --energy-tol X            warning tolerance for max relative drift (default: 1e-3)\n"
           "  --kernel STR              acceleration kernel choice: n, t, r, b, rt, bt, br, brt (default: n)\n"
           "  --quiet                   only print final summary\n"
           "  --papi [0|1]              enable PAPI counters (default: 0)\n"
           "  --help                    show this help message\n",
           program, NBODY_BINARY_VERSION_TEXT);
}

// MPI ENGINE
#if USE_MPI==1
static void retrieve_kernel (const char *kernel_choice, kernel_t *kernel)
{
  if (strcmp(kernel_choice, "obrc") == 0)
    *kernel = compute_accelerations_omp_br_cross;
  else
    die ("unknown kernel choice: %s", kernel_choice);
}

static const char *retrieve_kernel_name(const kernel_t kernel)
{
  if (kernel == compute_accelerations_omp_br_cross)
    return "obrc";
  else
    die ("unknown kernel function pointer");
  return "unknown";
}

int main (int argc, char **argv)
{
  const char  *input_path    = NULL;
  const char  *output_path   = NULL;
  size_t       nsteps        = 10u;
  size_t       energy_every  = 1u;
  dtype        dt            = (dtype) 1.0e-3;
  dtype        eps           = (dtype) 1.0e-2;
  dtype        g             = (dtype) 1.0;
  dtype        mass          = (dtype) 1.0;
  dtype        energy_tol    = (dtype) 1.0e-3;
  bool         quiet         = false;
  bool         papi          = false;
  bool         mpi           = false;
  particles_t  particles;
  dtype        kinetic0;
  dtype        potential0;
  dtype        global_energy;

  // Profiling Stuff
  size_t       profiler_flag  = 0u;
  const char  *profiler_path  = NULL;
  double       t0             = 0.0;
  profiler_t   profiler;

  // Kernel Choice
  kernel_t kernel = compute_accelerations_omp_br_cross;

  // Parse CLI
  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--output")) != NULL)
        output_path = value;
      else if ((value = option_value (&argi, argc, argv, "--nsteps")) != NULL)
        nsteps = parse_size (value, "--nsteps");
      else if ((value = option_value (&argi, argc, argv, "--energy-every")) != NULL)
        energy_every = parse_size (value, "--energy-every");
      else if ((value = option_value (&argi, argc, argv, "--dt")) != NULL)
        dt = parse_dtype (value, "--dt");
      else if ((value = option_value (&argi, argc, argv, "--eps")) != NULL)
        eps = parse_dtype (value, "--eps");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        mass = parse_dtype (value, "--mass");
      else if ((value = option_value (&argi, argc, argv, "--energy-tol")) != NULL)
        energy_tol = parse_dtype (value, "--energy-tol");
      else if ((value = option_value (&argi, argc, argv, "--profiler")) != NULL)
        profiler_flag = parse_size (value, "--profiler");
      else if ((value = option_value (&argi, argc, argv, "--profiler-path")) != NULL)
        profiler_path = value;
      else if ((value = option_value (&argi, argc, argv, "--kernel")) != NULL)
        retrieve_kernel(value, &kernel);
      else if ((value = option_value (&argi, argc, argv, "--papi")) != NULL)
      {
        papi = parse_size (value, "--papi");
        if (papi != 0u && papi != 1u)
          die ("--papi must be 0 or 1");
      }
      else if (strcmp (argv[argi], "--quiet") == 0)
        quiet = true;
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

  if (input_path == NULL)
    {
      print_usage (argv[0]);
      die ("missing required --input FILE");
    }
  if (!(dt > (dtype) 0.0))          die ("--dt must be positive");
  if (!(eps >= (dtype) 0.0))        die ("--eps must be non-negative");
  if (!(g > (dtype) 0.0))           die ("--G must be positive");
  if (!(mass > (dtype) 0.0))        die ("--mass must be positive");
  if (energy_every == 0u)           die ("--energy-every must be positive");
  if (!(energy_tol > (dtype) 0.0))  die ("--energy-tol must be positive");

  // Instantiate profiler if requested
  if (profiler_flag)
  {
    profiler_allocate (&profiler, nsteps);
    if (papi) profiler_papi_init(&profiler);
  }

  // Allocate particles' container to an empty state
  particles_init_empty (&particles);

  // MPI definitions
  int rank=0;
  int size=1;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Read particles from input file
  if (profiler_flag) { t0 = get_time();}
  particles_read_binary (input_path, mass, &particles);
  if (profiler_flag) { profiler.reading_time = get_time() - t0;}

  size_t n_local = particles.n / size;
  size_t offset = rank * n_local;

  // Slide assigned particles down to 0
  memmove(particles.x, particles.x + offset, n_local * sizeof(dtype));
  memmove(particles.y, particles.y + offset, n_local * sizeof(dtype));
  memmove(particles.z, particles.z + offset, n_local * sizeof(dtype));
  memmove(particles.vx, particles.vx + offset, n_local * sizeof(dtype));
  memmove(particles.vy, particles.vy + offset, n_local * sizeof(dtype));
  memmove(particles.vz, particles.vz + offset, n_local * sizeof(dtype));

  particles.n = n_local;

  // Get energy baseline
  if (!quiet)
  {
    if (profiler_flag) { t0 = get_time();}
    // Global energy computation
    global_energy = checked_global_energy(&particles, g, eps, rank, size, &kinetic0, &potential0);
    if (profiler_flag) { profiler.total_energy_time = get_time() - t0;}
  }
  // Print header
  if (!quiet && rank == 0)
    {
      printf ("# Direct N-body DKD\n");
      printf ("# arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
              particles.n, nsteps, (double) dt, (double) eps,
              (double) g, (double) mass);
      const char * kernel_name = retrieve_kernel_name(kernel);
      printf ("Acceleration Kernel: %s\n", kernel_name);
      printf ("# Step \t Time \t\t\t Kinetic \t\t\t Potential \t\t\t Total \t\t\t Relative Energy Drift\n");
      printf ("%zu \t %.17g \t %.17g \t %.17g \t %.17g \t %.17g\n",
              (size_t) 0u, 0.0, (double) kinetic0, (double) potential0,
              (double) global_energy, 0.0);
    }


  // Integration
  double max_rel_drift = 0.0;
  for (size_t step = 1u; step <= nsteps; ++step)
    {
      if (profiler_flag) { t0 = get_time();}
      leapfrog_dkd_step_mpi (&particles, g, eps, dt, &profiler, profiler_flag, step-1);
      if (profiler_flag) { profiler.total_step_time[step-1] = get_time() - t0;}

      // Get diagnostics, once in a while
      if (!quiet && (((step % energy_every) == 0u) || (step == nsteps)))
      {
        dtype         kinetic;
        dtype         potential;

        // Calculate true energy
        dtype current_global = checked_global_energy(&particles, g, eps, rank, size, &kinetic, &potential);

        const double  denom  = fmax (fabs ((double) global_energy), (double) DTYPE_MIN_NORMAL);
        const double  rel    = fabs ((double) (current_global - global_energy)) / denom;

        if (rel > max_rel_drift)
          max_rel_drift = rel;

        if (!quiet && rank == 0)
          printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
                  step, (double) step * (double) dt, (double) kinetic,
                  (double) potential, (double) current_global, rel);
      }
    }

  // Write final file
  if (output_path != NULL && rank == 0)
  {
    if (profiler_flag) { t0 = get_time();}
    particles_write_binary (output_path, &particles);
    if (profiler_flag) { profiler.writing_time = get_time() - t0;}

    printf ("# final: N=%zu steps=%zu arithmetic_dtype=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
          particles.n, nsteps, DTYPE_NAME, max_rel_drift, (double) energy_tol,
          (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

    if (max_rel_drift > (double) energy_tol)
      fprintf (stderr,
               "warning: relative energy drift %.6e exceeds tolerance %.6e; "
               "try smaller --dt, larger --eps, or better initial conditions\n",
               max_rel_drift, (double) energy_tol);

    // Print profiler statistics if requested
    if (profiler_flag) print_statistics (&profiler);
    if (profiler_flag && profiler_path != NULL)
    {
      // Save -- flags as config
      config_t config = (struct config_s) {
        .nsteps = nsteps,
        .energy_every = energy_every,
        .dt = dt,
        .eps = eps,
        .g = g,
        .mass = mass,
        .energy_tol = energy_tol,
        .kernel_name = retrieve_kernel_name(kernel),
        .kinetic0 = kinetic0,
        .potential0 = potential0,
        .max_relative_error = max_rel_drift
      };
      save_statistics(profiler_path, &config, &profiler);
    }

    if (profiler_flag)
    {
      if (papi) profiler_papi_free(&profiler);
      profiler_free(&profiler);
    }
  }

  // Don't leave garbage behind you
  particles_free (&particles);

  MPI_Finalize();
  return EXIT_SUCCESS;
}

// SERIAL/OMP only ENGINE
#else

static void retrieve_kernel (const char *kernel_choice, kernel_t *kernel)
{
  if (strcmp(kernel_choice, "ba") == 0)
    *kernel = compute_accelerations_baseline;
  else if (strcmp(kernel_choice, "n") == 0)
    *kernel = compute_accelerations_naive;
  else if (strcmp(kernel_choice, "t") == 0)
    *kernel = compute_accelerations_third_law;
  else if (strcmp(kernel_choice, "r") == 0)
    *kernel = compute_accelerations_rsqrt;
  else if (strcmp(kernel_choice, "b") == 0)
    *kernel = compute_accelerations_blocks;
  else if (strcmp(kernel_choice, "rt") == 0)
    *kernel = compute_accelerations_rsqrt_third_law;
  else if (strcmp(kernel_choice, "bt") == 0)
    *kernel = compute_accelerations_blocks_third_law;
  else if (strcmp(kernel_choice, "br") == 0)
    *kernel = compute_accelerations_blocks_rsqrt;
  else if (strcmp(kernel_choice, "brt") == 0)
    *kernel = compute_accelerations_blocks_rsqrt_third_law;
  else if (strcmp(kernel_choice, "obr") == 0)
    *kernel = compute_accelerations_omp_br;
  else if (strcmp(kernel_choice, "ort") == 0)
    *kernel = compute_accelerations_omp_rt;
  else if (strcmp(kernel_choice, "obrt") == 0)
    *kernel = compute_accelerations_omp_brt;
  else if (strcmp(kernel_choice, "ortr") == 0)
    *kernel = compute_accelerations_omp_rt_red;
  else if (strcmp(kernel_choice, "obrtr") == 0)
    *kernel = compute_accelerations_omp_brt_red;
  else
    die ("unknown kernel choice: %s", kernel_choice);
}

static const char *retrieve_kernel_name(const kernel_t kernel)
{
  if (kernel == compute_accelerations_baseline)
    return "ba";
  else if (kernel == compute_accelerations_naive)
    return "n";
  else if (kernel == compute_accelerations_third_law)
    return "t";
  else if (kernel == compute_accelerations_rsqrt)
    return "r";
  else if (kernel == compute_accelerations_blocks)
    return "b";
  else if (kernel == compute_accelerations_rsqrt_third_law)
    return "rt";
  else if (kernel == compute_accelerations_blocks_third_law)
    return "bt";
  else if (kernel == compute_accelerations_blocks_rsqrt)
    return "br";
  else if (kernel == compute_accelerations_blocks_rsqrt_third_law)
    return "brt";
  else if (kernel == compute_accelerations_omp_br)
    return "obr";
  else if (kernel == compute_accelerations_omp_rt)
    return "ort";
  else if (kernel == compute_accelerations_omp_brt)
    return "obrt";
  else if (kernel == compute_accelerations_omp_rt_red)
    return "ortr";
  else if (kernel == compute_accelerations_omp_brt_red)
    return "obrtr";
  else
    die ("unknown kernel function pointer");
  return "unknown";
}

int main (int argc, char **argv)
{
  const char  *input_path    = NULL;
  const char  *output_path   = NULL;
  size_t       nsteps        = 10u;
  size_t       energy_every  = 1u;
  dtype        dt            = (dtype) 1.0e-3;
  dtype        eps           = (dtype) 1.0e-2;
  dtype        g             = (dtype) 1.0;
  dtype        mass          = (dtype) 1.0;
  dtype        energy_tol    = (dtype) 1.0e-3;
  bool         quiet         = false;
  bool         papi          = false;
  particles_t  particles;
  dtype        kinetic0;
  dtype        potential0;
  dtype        energy0;

  // Profiling Stuff
  size_t       profiler_flag  = 0u;
  const char  *profiler_path  = NULL;
  double       t0             = 0.0;
  profiler_t   profiler;

  // Kernel Choice
  kernel_t kernel = compute_accelerations_baseline;

  // Parse CLI
  for (int argi = 1; argi < argc; ++argi)
    {
      const char *value;

      if ((value = option_value (&argi, argc, argv, "--input")) != NULL)
        input_path = value;
      else if ((value = option_value (&argi, argc, argv, "--output")) != NULL)
        output_path = value;
      else if ((value = option_value (&argi, argc, argv, "--nsteps")) != NULL)
        nsteps = parse_size (value, "--nsteps");
      else if ((value = option_value (&argi, argc, argv, "--energy-every")) != NULL)
        energy_every = parse_size (value, "--energy-every");
      else if ((value = option_value (&argi, argc, argv, "--dt")) != NULL)
        dt = parse_dtype (value, "--dt");
      else if ((value = option_value (&argi, argc, argv, "--eps")) != NULL)
        eps = parse_dtype (value, "--eps");
      else if ((value = option_value (&argi, argc, argv, "--G")) != NULL)
        g = parse_dtype (value, "--G");
      else if ((value = option_value (&argi, argc, argv, "--mass")) != NULL)
        mass = parse_dtype (value, "--mass");
      else if ((value = option_value (&argi, argc, argv, "--energy-tol")) != NULL)
        energy_tol = parse_dtype (value, "--energy-tol");
      else if ((value = option_value (&argi, argc, argv, "--profiler")) != NULL)
        profiler_flag = parse_size (value, "--profiler");
      else if ((value = option_value (&argi, argc, argv, "--profiler-path")) != NULL)
        profiler_path = value;
      else if ((value = option_value (&argi, argc, argv, "--kernel")) != NULL)
        retrieve_kernel(value, &kernel);
      else if ((value = option_value (&argi, argc, argv, "--papi")) != NULL)
      {
        papi = parse_size (value, "--papi");
        if (papi != 0u && papi != 1u)
          die ("--papi must be 0 or 1");
      }
      else if (strcmp (argv[argi], "--quiet") == 0)
        quiet = true;
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

  if (input_path == NULL)
    {
      print_usage (argv[0]);
      die ("missing required --input FILE");
    }
  if (!(dt > (dtype) 0.0))          die ("--dt must be positive");
  if (!(eps >= (dtype) 0.0))        die ("--eps must be non-negative");
  if (!(g > (dtype) 0.0))           die ("--G must be positive");
  if (!(mass > (dtype) 0.0))        die ("--mass must be positive");
  if (energy_every == 0u)           die ("--energy-every must be positive");
  if (!(energy_tol > (dtype) 0.0))  die ("--energy-tol must be positive");

  // Instantiate profiler if requested
  if (profiler_flag)
  {
    profiler_allocate (&profiler, nsteps);
    if (papi) profiler_papi_init(&profiler);
  }

  // Allocate particles' container to an empty state
  particles_init_empty (&particles);

  // Read particles from input file
  if (profiler_flag) { t0 = get_time();}
  particles_read_binary (input_path, mass, &particles);
  if (profiler_flag) { profiler.reading_time = get_time() - t0;}

  // Get energy baseline
  if (profiler_flag) { t0 = get_time();}
  energy0 = total_energy (&particles, g, eps, &kinetic0, &potential0);
  if (profiler_flag) { profiler.total_energy_time = get_time() - t0;}

  // Print header
  if (!quiet)
    {
      printf ("# Direct N-body DKD\n");
      printf ("# arithmetic_dtype=%s binary_storage=float32 format=%s\n",
              DTYPE_NAME, NBODY_BINARY_VERSION_TEXT);
      printf ("# N=%zu nsteps=%zu dt=%.17g eps=%.17g G=%.17g mass=%.17g\n",
              particles.n, nsteps, (double) dt, (double) eps,
              (double) g, (double) mass);
      const char * kernel_name = retrieve_kernel_name(kernel);
      printf ("Acceleration Kernel: %s\n", kernel_name);
      printf ("# Step \t Time \t\t\t Kinetic \t\t\t Potential \t\t\t Total \t\t\t Relative Energy Drift\n");
      printf ("%zu \t %.17g \t %.17g \t %.17g \t %.17g \t %.17g\n",
              (size_t) 0u, 0.0, (double) kinetic0, (double) potential0,
              (double) energy0, 0.0);
    }


  // Integration
  double max_rel_drift = 0.0;
  for (size_t step = 1u; step <= nsteps; ++step)
    {
      if (profiler_flag) { t0 = get_time();}
      leapfrog_dkd_step (&particles, g, eps, dt, &profiler, profiler_flag, step-1, kernel);
      if (profiler_flag) { profiler.total_step_time[step-1] = get_time() - t0;}

      // Get diagnostics, once in a while
      if (((step % energy_every) == 0u) || (step == nsteps))
        {
          dtype         kinetic;
          dtype         potential;
          const dtype   energy = total_energy (&particles, g, eps, &kinetic, &potential);
          const double  denom  = fmax (fabs ((double) energy0), (double) DTYPE_MIN_NORMAL);
          const double  rel    = fabs ((double) (energy - energy0)) / denom;

          if (rel > max_rel_drift)
            max_rel_drift = rel;
          if (!quiet)
            printf ("%zu %.17g %.17g %.17g %.17g %.17g\n",
                    step, (double) step * (double) dt, (double) kinetic,
                    (double) potential, (double) energy, rel);
        }
    }

  // Write final file
  if (output_path != NULL)
  {
    if (profiler_flag) { t0 = get_time();}
    particles_write_binary (output_path, &particles);
    if (profiler_flag) { profiler.writing_time = get_time() - t0;}
  }

  // Say good-bye
  printf ("# final: N=%zu steps=%zu arithmetic_dtype=%s max_relative_energy_drift=%.17g tolerance=%.17g status=%s\n",
          particles.n, nsteps, DTYPE_NAME, max_rel_drift, (double) energy_tol,
          (max_rel_drift <= (double) energy_tol) ? "OK" : "WARNING");

  if (max_rel_drift > (double) energy_tol)
    fprintf (stderr,
             "warning: relative energy drift %.6e exceeds tolerance %.6e; "
             "try smaller --dt, larger --eps, or better initial conditions\n",
             max_rel_drift, (double) energy_tol);


  // Print profiler statistics if requested
  if (profiler_flag) print_statistics (&profiler);
  if (profiler_flag && profiler_path != NULL)
  {
    // Save -- flags as config
    config_t config = (struct config_s) {
      .nsteps = nsteps,
      .energy_every = energy_every,
      .dt = dt,
      .eps = eps,
      .g = g,
      .mass = mass,
      .energy_tol = energy_tol,
      .kernel_name = retrieve_kernel_name(kernel),
      .kinetic0 = kinetic0,
      .potential0 = potential0,
      .max_relative_error = max_rel_drift
    };
    save_statistics(profiler_path, &config, &profiler);
  }


  if (profiler_flag)
  {
    if (papi) profiler_papi_free(&profiler);
    profiler_free(&profiler);
  }

  // Don't leave garbage behind you
  particles_free (&particles);

  return EXIT_SUCCESS;
}
#endif
