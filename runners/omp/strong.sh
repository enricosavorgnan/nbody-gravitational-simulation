#!/bin/bash

# Fixed parameters for Strong Scaling
PARTICLES=100000
STEPS=100

echo "Submitting OMP Strong Scaling Experiments (Fixed N = $PARTICLES)"

# Iterate through powers of 2 for OMP threads
for THREADS in 1 2 4 8 16 32 64 128; do
    echo "Submitting job for $THREADS OMP threads..."

    PROFILER_FILE="./profilings/s_${THREADS}.txt"

    # Submit the sbatch script, requesting 1 task but $THREADS cpus-per-task
    sbatch --ntasks=1 --cpus-per-task=$THREADS \
           --job-name="S_${THREADS}" \
           --export=ALL,N_PARTICLES=$PARTICLES,STEPS=$STEPS,PROFILER_FILE=$PROFILER_FILE \
           ./runners/omp/template.sh
done

echo "All strong scaling jobs submitted."
