#!/bin/bash

#SBATCH --job-name=UNROLL2
#SBATCH --output=./log/unroll2_%j.out
#SBATCH --error=./log/unroll2_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/loop-unroll"

make test-auto-unroll USE_PAPI=1
