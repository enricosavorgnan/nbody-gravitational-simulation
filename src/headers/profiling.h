/*
 * Definitions of Wall-Time clocks used to measure time-performances
 * in the n-Body simulation.
 * The times are stored into a struct timespec.
 */

#ifndef PROFILING_H
#define PROFILING_H

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "utils.h"


typedef struct profiler_s
{
    // One-Time Measurements
    double reading_time;
    double writing_time;
    double compute_acceleration_time;
    double total_energy_time;

    // Per-Step Measurements
    size_t n_steps;
    double *force_time;
    double *first_drift_time;
    double *kick_time;
    double *second_drift_time;
    double *total_step_time;

} profiler_t ;


static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1.0e-9;
}

void profiler_allocate (profiler_t *profiler, const size_t n_steps);
void profiler_free (const profiler_t *profiler);
void print_statistics (const profiler_t *profiler);
void save_statistics (const char *path, const profiler_t *profiler);

#endif // PROFILING_H