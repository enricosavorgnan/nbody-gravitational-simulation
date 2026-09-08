#!/bin/bash

#SBATCH --job-name=RSQRT
#SBATCH --output=./log/rsqrt_%j.out
#SBATCH --error=./log/rsqrt_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/rsqrt-impact"

make clean
make clean-folders
make test-rsqrt USE_PAPI=1