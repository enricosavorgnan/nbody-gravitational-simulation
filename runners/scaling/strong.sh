#!/bin/bash

# Fixed parameters for Strong Scaling
PARTICLES=100000
STEPS=100

echo "Submitting Strong Scaling Experiments (Fixed N = $PARTICLES)"

# Iterate through powers of 2 for MPI ranks
for RANKS in 1 2 4 8 16 32 64 128; do
    echo "Submitting job for $RANKS MPI ranks..."

    # Submit the sbatch script, requesting $RANKS tasks, and passing our variables
    sbatch --ntasks=$RANKS \
           --job-name="S_${RANKS}" \
           --export=ALL,N_PARTICLES=$PARTICLES,STEPS=$STEPS \
           ./runners/scaling/template.sh
done

echo "All strong scaling jobs submitted."