#!/bin/bash

#SBATCH --job-name=MAU
#SBATCH --output=./log/fma_%j.out
#SBATCH --error=./log/fma_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/fma"

make test-fma USE_PAPI=1