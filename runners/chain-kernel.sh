chain-aos.sh#!/bin/bash

# Configuration
SBATCH_FILE="./runners/sbatch-kernel.sh"
TOTAL_RUNS=5

# Submit the first job
PREV_JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
echo "Submitted initial job $PREV_JOB_ID (Run 1/$TOTAL_RUNS)"

# Chain subsequent jobs to depend on the previous job completing successfully
for ((i=2; i<=TOTAL_RUNS; i++)); do
    PREV_JOB_ID=$(sbatch --parsable --dependency=afterok:"$PREV_JOB_ID" "$SBATCH_FILE")
    echo "Submitted dependent job $PREV_JOB_ID (Run $i/$TOTAL_RUNS)"
done