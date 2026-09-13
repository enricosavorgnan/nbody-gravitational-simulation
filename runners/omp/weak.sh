#!/bin/bash

# Fixed parameters for Weak Scaling
# N_BASE is the number of particles for a SINGLE thread
N_BASE=100000
STEPS=100

echo "Submitting OMP Weak Scaling Experiments (Base N = $N_BASE for 1 thread)"

# Iterate through powers of 2 for OMP threads
for THREADS in 1 2 4 8 16 32 64; do

    # Calculate N for this run: N = N_BASE * sqrt(THREADS)
    # We use awk to do the math and round to the nearest integer
    CURRENT_N=$(awk -v base=$N_BASE -v threads=$THREADS 'BEGIN { printf "%.0f", base * sqrt(threads) }')

    echo "Submitting job for $THREADS OMP threads (Particles = $CURRENT_N)..."

    PROFILER_FILE="./profilings/w_${THREADS}_${CURRENT_N}.txt"

    # Submit the sbatch script with the dynamically calculated N
    sbatch --ntasks=1 --cpus-per-task=$THREADS \
           --job-name="W_${THREADS}" \
           --export=ALL,N_PARTICLES=$CURRENT_N,STEPS=$STEPS,PROFILER_FILE=$PROFILER_FILE \
           ./runners/omp/template.sh
done

echo "All weak scaling jobs submitted."
