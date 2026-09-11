#!/bin/bash

#SBATCH --job-name=NBODY
#SBATCH --output=./log/nbody_%j.out
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p log
cd "./src"

make test-mpi USE_PAPI=1