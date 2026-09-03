/*
 * Contains implementations for the serial time profiling and more
*/

#include "./headers/profiling.h"
#include "./headers/common.h"
#include "./headers/utils.h"

/* PROFILING */

void profiler_allocate (profiler_t *profiler, const size_t n_steps)
{
    const size_t bytes = n_steps * sizeof(double);

    // One-Time Measurements
    profiler->reading_time              = 0.0;
    profiler->writing_time              = 0.0;
    profiler->total_energy_time         = 0.0;

    // Per-Step Measurements
    profiler->n_steps                   = n_steps;
    profiler->force_time                = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->first_drift_time          = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->kick_time                 = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->second_drift_time         = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->total_step_time           = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);

#ifdef USE_PAPI
    profiler->papi_eventset             = PAPI_NULL;
    profiler->papi_cycles                = checked_aligned_alloc(n_steps * sizeof(long long), NBODY_ALIGNMENT);
    profiler->papi_instructions          = checked_aligned_alloc(n_steps * sizeof(long long), NBODY_ALIGNMENT);
    profiler->papi_l1_dcm                = checked_aligned_alloc(n_steps * sizeof(long long), NBODY_ALIGNMENT);
    profiler->papi_l2_dcm                = checked_aligned_alloc(n_steps * sizeof(long long), NBODY_ALIGNMENT);
    profiler->papi_vec_dp                = checked_aligned_alloc(n_steps * sizeof(long long), NBODY_ALIGNMENT);
#endif
}


void profiler_free (const profiler_t *profiler)
{
    free(profiler->force_time);
    free(profiler->first_drift_time);
    free(profiler->kick_time);
    free(profiler->second_drift_time);
    free(profiler->total_step_time);
#ifdef USE_PAPI
    free(profiler->papi_cycles);
    free(profiler->papi_instructions);
    free(profiler->papi_l1_dcm);
    free(profiler->papi_l2_dcm);
    free(profiler->papi_vec_dp);
#endif
}



/* SAVING */

static void print_single_statistics (const char *label, const double *times, const size_t n_steps)
{
    if (n_steps == 0) return;

    // Sort times
    double *sorted = malloc (n_steps * sizeof(double));
    if (!sorted) die ("Memory allocation failed in 'print_statistics'");
    for (size_t i = 0; i < n_steps; i++) sorted[i] = times[i];
    qsort(sorted, n_steps, sizeof(double), compare_doubles);

    // Compute Median
    double median = (n_steps % 2 != 0) ? sorted[n_steps / 2] : (sorted[(n_steps - 1) / 2] + sorted[n_steps / 2]) / 2.0;

    // Compute Trim Mean
    size_t trim_count = n_steps / 10;       // 10% trim, maybe useless with 5 tries
    double sum = 0.0;
    size_t valid_elements = n_steps - 2 * trim_count;
    for (size_t i = trim_count; i < n_steps - trim_count; i++) sum += sorted[i];
    double trimmed_mean = sum / valid_elements;

    // Compute Standard Deviation
    double variance = 0.0;
    for (size_t i = trim_count; i < n_steps - trim_count; i++)
    {
        double dev = sorted[i] - trimmed_mean;
        variance += dev * dev;
    }
    double std = sqrt(variance / valid_elements);

    // Print Statistics
    printf ("%-15s \t\t Median; %.6e s \t\t Trimmed Mean: %.6e s \t\t Standard Deviation: %.6e s\n", label, median, trimmed_mean, std);

    free(sorted);
}


void print_statistics (const profiler_t *profiler)
{
    printf ("\n--- Performance Profiling Report ---\n");

    // Report One-Time Measurements
    printf ("%-15s : %.6e s\n", "File Read", profiler->reading_time);
    printf ("%-15s : %.6e s\n", "File Write", profiler->writing_time);
    printf ("%-15s : %.6e s\n", "Total Run", profiler->total_energy_time);

    printf ("\n--- Step Statistics (%zu steps) ---\n", profiler->n_steps);

    // Report Per-Step Measurements
    print_single_statistics ("Total Step", profiler->total_step_time, profiler->n_steps);
    print_single_statistics ("First Drift", profiler->first_drift_time, profiler->n_steps);
    print_single_statistics ("Compute Force", profiler->force_time, profiler->n_steps);
    print_single_statistics ("Kick", profiler->kick_time, profiler->n_steps);
    print_single_statistics ("Second Drift", profiler->second_drift_time, profiler->n_steps);
}


void save_single_statistics (const char *path, const char *label, const double *times, const size_t n_steps)
{
    FILE *fp = fopen(path, "a");
    if (!fp) die ("Cannot open file '%s' for writing statistics", path);

    fprintf(fp, "%s\n", label);
    for (size_t i = 0; i < n_steps; i++)
    {
        fprintf(fp, "%.6e\n", times[i]);
    }
    fprintf(fp, "\n");

    fclose(fp);
}


void save_config(const char *path, const config_t *config)
{
    FILE *fp = fopen(path, "a");
    if (!fp) die ("Cannot open file '%s' for writing configuration", path);

    fprintf(fp, "--- Configuration ---\n");
    fprintf(fp, "nsteps: %zu\n", config->nsteps);
    fprintf(fp, "dt: %.6e\n", config->dt);
    fprintf(fp, "eps: %.6e\n", config->eps);
    fprintf(fp, "G: %.6e\n", config->g);
    fprintf(fp, "mass: %.6e\n", config->mass);
    fprintf(fp, "energy_every: %zu\n", config->energy_every);
    fprintf(fp, "energy_tol: %.6e\n", config->energy_tol);
    fprintf(fp, "kernel_choice: %s\n", config->kernel_name);
    fprintf(fp, "kinetic0: %.6e\n", config->kinetic0);
    fprintf(fp, "potential0: %.6e\n", config->potential0);
    fclose(fp);
}


void save_statistics (const char *path, const config_t *config, const profiler_t *profiler)
{
    save_single_statistics(path, "File Read", &profiler->reading_time, 1);
    save_single_statistics(path, "File Write", &profiler->writing_time, 1);
    save_single_statistics(path, "Total Run", &profiler->total_energy_time, 1);

    save_single_statistics(path, "Total Step", profiler->total_step_time, profiler->n_steps);
    save_single_statistics(path, "First Drift", profiler->first_drift_time, profiler->n_steps);
    save_single_statistics(path, "Compute Force", profiler->force_time, profiler->n_steps);
    save_single_statistics(path, "Kick", profiler->kick_time, profiler->n_steps);
    save_single_statistics(path, "Second Drift", profiler->second_drift_time, profiler->n_steps);

    save_config(path, config);
#ifdef USE_PAPI
    save_single_papi_statistics(path, "PAPI Cycles", profiler->papi_cycles, profiler->n_steps);
    save_single_papi_statistics(path, "PAPI Instructions", profiler->papi_instructions, profiler->n_steps);
    save_single_papi_statistics(path, "PAPI L1 Misses", profiler->papi_l1_dcm, profiler->n_steps);
    save_single_papi_statistics(path, "PAPI L2 Misses", profiler->papi_l2_dcm, profiler->n_steps);
    save_single_papi_statistics(path, "PAPI Vectorial DP", profiler->papi_vec_dp, profiler->n_steps);
#endif
}

#ifdef USE_PAPI
void save_single_papi_statistics (const char *path, const char *label, const long long *values, const size_t n_steps)
{
    if (values == NULL || n_steps == 0) return;

    FILE *fp = fopen(path, "a");
    if (!fp) die ("Cannot open file '%s' for writing PAPI stuff", path );

    fprintf(fp, "%s\n", label);
    for (size_t i = 0; i < n_steps; i++)
    {
        fprintf(fp, "%lld\n", values[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}
#endif

void profiler_papi_init (profiler_t *profiler)
{
#ifdef USE_PAPI
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT)
        die("PAPI init error: %s", PAPI_strerror(retval));

    profiler->papi_eventset = PAPI_NULL;
    if (PAPI_create_eventset(&profiler->papi_eventset) != PAPI_OK)
        die("PAPI_create_eventset failed");

    // Add events
    PAPI_add_event(profiler->papi_eventset, PAPI_TOT_CYC);
    PAPI_add_event(profiler->papi_eventset, PAPI_TOT_INS);
    PAPI_add_event(profiler->papi_eventset, PAPI_L1_DCM);
    PAPI_add_event(profiler->papi_eventset, PAPI_DP_OPS);
#else
    (void) profiler;
#endif
}

void profiler_papi_start (profiler_t *profiler)
{
#ifdef USE_PAPI
    PAPI_reset(profiler->papi_eventset);
    PAPI_start(profiler->papi_eventset);
#else
    (void) profiler;
#endif
}

void profiler_papi_stop (profiler_t *profiler, const size_t step)
{
#ifdef USE_PAPI
    long long values[PAPI_EVENTS_COUNT] = {0};
    PAPI_stop(profiler->papi_eventset, values);
    profiler->papi_cycles[step]       = values[0];
    profiler->papi_instructions[step] = values[1];
    profiler->papi_l1_dcm[step]       = values[2];
    profiler->papi_vec_dp[step]       = values[3];
#else
    (void) profiler; (void) step;
#endif
}

void profiler_papi_free (profiler_t *profiler)
{
#ifdef USE_PAPI
    PAPI_cleanup_eventset(profiler->papi_eventset);
    PAPI_destroy_eventset(&profiler->papi_eventset);
    PAPI_shutdown();
#else
    (void) profiler;
#endif
}