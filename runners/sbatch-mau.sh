#!/bin/bash

#SBATCH --account=enricosavorgnan
#SBATCH --job-name=MAU
#SBATCH --output=./log/mau_%j.out
#SBATCH --error=./log/mau_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/mau"

make test-mau USE_PAPI=1