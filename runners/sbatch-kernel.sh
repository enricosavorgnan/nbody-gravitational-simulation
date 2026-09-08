#!/bin/bash

#SBATCH --job-name=KERNEL
#SBATCH --output=./log/kernel_%j.out
#SBATCH --error=./log/kernel_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/kernel"

make test-kernel USE_PAPI=1