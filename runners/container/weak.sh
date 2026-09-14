#!/bin/bash
# Fixed parameters for Container Weak Scaling (N = N_BASE * sqrt(RANKS))
N_BASE=100000
STEPS=100

mkdir -p ./profilings/container

echo "Submitting Container Weak Scaling (Base N = $N_BASE)"

for RANKS in 1 2 4 8 16 32 64; do
    CURRENT_N=$(awk -v base=$N_BASE -v ranks=$RANKS 'BEGIN { printf "%.0f", base * sqrt(ranks) }')
    echo "Submitting container job for $RANKS ranks (N = $CURRENT_N)..."
    PROFILER_FILE="./profilings/container/w_${RANKS}_${CURRENT_N}.txt"

    sbatch --ntasks=$RANKS \
           --job-name="CW_${RANKS}" \
           --export=ALL,N_PARTICLES=$CURRENT_N,STEPS=$STEPS,PROFILER_FILE=$PROFILER_FILE \
           ./runners/container/template_container.sh
done

echo "All container weak scaling jobs submitted."