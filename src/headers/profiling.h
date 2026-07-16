/*
 * Definitions of Wall-Time clocks used to measure time-performances
 * in the n-Body simulation.
 * The times are stored into a struct timespec.
 */

#ifndef NBODY_TIME_H
#define NBODY_TIME_H
#endif //NBODY_TIME_H

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <stdio.h>


typedef struct profiler_s
{
    // One-Time Measurements
    double *reading_time;
    double *writing_time;
    double *allocation_time;
    double *compute_acceleration_time;
    double *total_energy_time;

    // Per-Step Measurements
    size_t n_steps;
    double *force_time;
    double *drift_time;
    double *kick_time;
    double *total_step_time;

} profiler_t ;


static void profiler_allocate (profiler_t *profiler, const size_t n_steps) {};
static void profiler_free (const profiler_t *profiler) {};
static void print_single_statistics (const char *label, const double *times, const size_t n_steps) {};
static void save_statistics (const char *path, const char *label, const double *times, const size_t n_steps) {};
static int compare_doubles (const void *a, const void *b);


