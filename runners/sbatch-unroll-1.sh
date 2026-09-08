#!/bin/bash

#SBATCH --job-name=UNROLL1
#SBATCH --output=./log/unroll1_%j.out
#SBATCH --error=./log/unroll1_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/loop-unroll"

make test-n-unroll USE_PAPI=1
