#!/bin/bash
# Fixed parameters for Container Strong Scaling
PARTICLES=100000
STEPS=100

mkdir -p ./profilings/container

echo "Submitting Container Strong Scaling (Fixed N = $PARTICLES)"

for RANKS in 1 2 4 8 16 32 64; do
    echo "Submitting container job for $RANKS MPI ranks..."
    PROFILER_FILE="./profilings/container/s_${RANKS}.txt"

    sbatch --ntasks=$RANKS \
           --job-name="CS_${RANKS}" \
           --export=ALL,N_PARTICLES=$PARTICLES,STEPS=$STEPS,PROFILER_FILE=$PROFILER_FILE \
           ./runners/container/template_container.sh
done

echo "All container strong scaling jobs submitted."