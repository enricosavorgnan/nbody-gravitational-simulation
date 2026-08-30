#!/bin/bash

#SBATCH --job-name=NBODY
#SBATCH --output=./log/nbody_%j.out
#SBATCH --error=./log/nbody_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/loop-unroll"

make clean
make clean-folders
make test-n-unroll
make test-auto-unroll
make test-no-unroll