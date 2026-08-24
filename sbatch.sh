#!/bin/bash

#SBATCH --job-name=NBODY
#SBATCH --output=log/nbody_%j.out
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p log

make clean
make all PRECISION=double
make test-serial