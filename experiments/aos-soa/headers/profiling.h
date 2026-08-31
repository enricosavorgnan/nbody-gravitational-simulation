/*
 * Definitions of Wall-Time clocks used to measure time-performances
 * in the n-Body simulation.
 * The times are stored into a struct timespec.
 */
#define _POSIX_C_SOURCE 199309L

#ifndef PROFILING_H
#define PROFILING_H

#ifdef USE_PAPI
#include <papi.h>
#define PAPI_EVENTS_COUNT 5
#endif

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "utils.h"


typedef struct profiler_s
{
    // One-Time Measurements
    double reading_time;
    double writing_time;
    double total_energy_time;

    // Per-Step Measurements
    size_t n_steps;
    double *force_time;
    double *first_drift_time;
    double *kick_time;
    double *second_drift_time;
    double *total_step_time;

#ifdef USE_PAPI
    int papi_eventset;
    long long *papi_cycles;
    long long *papi_instructions;
    long long *papi_l1_dcm;
    long long *papi_l2_dcm;
    long long *papi_vec_dp;
#endif

} profiler_t ;


typedef struct config_s
{
    size_t nsteps;
    dtype dt;
    dtype eps;
    dtype g;
    dtype mass;
    size_t energy_every;
    dtype energy_tol;
    const char * kernel_name;
    dtype kinetic0;
    dtype potential0;
} config_t;


static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1.0e-9;
}

void profiler_allocate (profiler_t *profiler, const size_t n_steps);
void profiler_free (const profiler_t *profiler);
void print_statistics (const profiler_t *profiler);
void save_single_statistics (const char *path, const char *label, const double *times, const size_t n_steps);
void save_config(const char *path, const config_t *config);
void save_statistics (const char *path, const config_t *config, const profiler_t *profiler);

// PAPI stuff
void profiler_papi_init(profiler_t *profiler);
void profiler_papi_start(profiler_t *profiler);
void profiler_papi_stop(profiler_t *profiler, const size_t step);
void profiler_papi_free(profiler_t *profiler);


#endif // PROFILING_H