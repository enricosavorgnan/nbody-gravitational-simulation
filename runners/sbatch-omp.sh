#!/bin/bash

#SBATCH --job-name=OMP
#SBATCH --output=./log/omp_%j.out
#SBATCH --error=./log/omp_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/aos-soa"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=TRUE # TODO: Remove this line when running serious tests!!!

make test-omp USE_PAPI=1