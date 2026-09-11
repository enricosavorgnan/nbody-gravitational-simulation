#!/bin/bash

# Fixed parameters for Weak Scaling
# N_BASE is the number of particles for a SINGLE rank
N_BASE=100000
STEPS=100

echo "Submitting Weak Scaling Experiments (Base N = $N_BASE for 1 rank)"

# Iterate through powers of 2 for MPI ranks
for RANKS in 1 2 4 8 16 32 64 128; do

    # Calculate N for this run: N = N_BASE * sqrt(RANKS)
    # We use awk to do the math and round to the nearest integer
    CURRENT_N=$(awk -v base=$N_BASE -v ranks=$RANKS 'BEGIN { printf "%.0f", base * sqrt(ranks) }')

    echo "Submitting job for $RANKS MPI ranks (Particles = $CURRENT_N)..."

    # Submit the sbatch script with the dynamically calculated N
    sbatch --ntasks=$RANKS \
           --job-name="weak_${RANKS}" \
           --export=ALL,N_PARTICLES=$CURRENT_N,STEPS=$STEPS \
           ./runners/scaling/template.sh
done

echo "All weak scaling jobs submitted."