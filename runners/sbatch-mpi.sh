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

module load openMPI

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/mpi/"

export OMP_STACKSIZE=64M
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=TRUE


mkdir -p ./bins
mkdir -p ./profilings
make all USE_PAPI=1

srun ./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-mpi.bin

srun ./main --input ./bins/test-mpi.bin \
            --nsteps 100 \
            --dt 1e-4 \
            --eps 0.05 \
            --energy-every 100 \
            --output ./bins/obrtr.out \
             --kernel "obrc" \
             --profiler 1  \
             --profiler-path ./profilings/obrc.txt

srun ./main --input ./bins/test-mpi.bin \
            --nsteps 100 \
            --dt 1e-4 \
            --eps 0.05 \
            --energy-every 100 \
            --output ./bins/obrtr.out \
             --kernel "obrc" \
             --profiler 1  \
             --profiler-path ./profilings/obrc.txt

srun ./main --input ./bins/test-mpi.bin \
            --nsteps 100 \
            --dt 1e-4 \
            --eps 0.05 \
            --energy-every 100 \
            --output ./bins/obrtr.out \
             --kernel "obrc" \
             --profiler 1  \
             --profiler-path ./profilings/obrc.txt

srun ./main --input ./bins/test-mpi.bin \
            --nsteps 100 \
            --dt 1e-4 \
            --eps 0.05 \
            --energy-every 100 \
            --output ./bins/obrtr.out \
             --kernel "obrc" \
             --profiler 1  \
             --profiler-path ./profilings/obrc.txt

srun ./main --input ./bins/test-mpi.bin \
            --nsteps 100 \
            --dt 1e-4 \
            --eps 0.05 \
            --energy-every 100 \
            --output ./bins/obrtr.out \
             --kernel "obrc" \
             --profiler 1  \
             --profiler-path ./profilings/obrc.txt