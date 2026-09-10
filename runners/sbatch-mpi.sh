#!/bin/bash

#SBATCH --job-name=MPI
#SBATCH --output=./log/mpi_%j.out
#SBATCH --error=./log/mpi_%j.err
#SBATCH --partition=GENOA
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/mpi"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=TRUE

make test-mpi USE_PAPI=1