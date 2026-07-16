/*
 * Contains implementations for the serial time profiling and more
*/

#include "./headers/nbody_profiling.h"
#include "./headers/nbody_common.h"
#include "./headers/utils.h"


/* ==================================================================================== */
/* UTILS */

/*
 * Get time for serial code.
 * Uses CLOCK_MONOTONIC to ensure precision independently of system clock adjustments.
*/
static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_sec * 1.0e-9;
}


/* ==================================================================================== */
/* PROFILING */

static void profiler_allocate (profiler_t *profiler, const size_t n_steps)
{
    const size_t bytes = n_steps * sizeof(double);

    // One-Time Measurements
    profiler->reading_time              = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->writing_time              = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->allocation_time           = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->compute_acceleration_time = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->total_energy_time         = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);

    // Per-Step Measurements
    profiler->n_steps                   = n_steps;
    profiler->force_time                = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->drift_time                = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->kick_time                 = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
    profiler->total_step_time           = checked_aligned_alloc(bytes, NBODY_ALIGNMENT);
}


static void profiler_free (const profiler_t *profiler)
{
    free(profiler->reading_time);
    free(profiler->writing_time);
    free(profiler->allocation_time);
    free(profiler->compute_acceleration_time);
    free(profiler->total_energy_time);

    free(profiler->force_time);
    free(profiler->drift_time);
    free(profiler->kick_time);
    free(profiler->total_step_time);
}



/* ==================================================================================== */
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


static void print_statistics (const profiler_t *profiler)
{
    printf ("\n--- Performance Profiling Report ---\n");

    // Report One-Time Measurements
    printf ("%-15s : %.6e s\n", "File Read", *profiler->reading_time);
    printf ("%-15s : %.6e s\n", "File Write", *profiler->writing_time);
    printf ("%-15s : %.6e s\n", "Allocation", *profiler->allocation_time);
    printf ("%-15s : %.6e s\n", "Compute Acceleration", *profiler->compute_acceleration_time);
    printf ("%-15s : %.6e s\n", "Total Run", *profiler->total_energy_time);

    printf ("\n--- Step Statistics (%zu steps) ---\n", profiler->n_steps);

    // Report Per-Step Measurements
    print_single_statistics ("Total Step", profiler->total_step_time, profiler->n_steps);
    print_single_statistics ("Drift", profiler->drift_time, profiler->n_steps);
    print_single_statistics ("Compute Force", profiler->force_time, profiler->n_steps);
    print_single_statistics ("Kick", profiler->kick_time, profiler->n_steps);
}


static void save_statistics (const char *path, const char *label, const double *times, const size_t n_steps)
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