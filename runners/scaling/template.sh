#!/bin/bash

#SBATCH --account=dssc
#SBATCH --job-name=scale
#SBATCH --output=./log/scale_%x_%j.out
#SBATCH --error=./log/scale_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=GENOA
#SBATCH --exclusive

# Variables passed via --export during sbatch submission
N_PARTICLES=${N_PARTICLES:-100000}
STEPS=${STEPS:-100}
PROFILER_FILE=${PROFILER_FILE:-"./profilings/profiler_${N_PARTICLES}.txt"}
INPUT_FILE="./bins/${N_PARTICLES}.bin"
OUTPUT_FILE="./bins/${N_PARTICLES}.out"

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "MPI Ranks: $SLURM_NTASKS"
echo "Particles: $N_PARTICLES"
echo "=========================================================="

module load openMPI

cd "$SLURM_SUBMIT_DIR"
cd "./experiments/mpi/"

export OMP_STACKSIZE=64M
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=TRUE

mkdir -p ./bins
mkdir -p ./profilings
make all USE_PAPI=1

echo "Generating initial conditions..."
srun ./generate_initial_conditions --model 0 --n $N_PARTICLES --seed 42 --output $INPUT_FILE

echo "Starting simulation..."
/usr/bin/time -a -o $PROFILER_FILE -f "\n--- OS / MPI Launch Time ---\nReal: %e seconds\nUser: %U seconds\nSys: %S seconds" \
mpirun -n $SLURM_NTASKS ./main --input $INPUT_FILE --nsteps $STEPS --dt 1e-4 --eps 0.05 --energy-every 100 --output $OUTPUT_FILE --kernel "obrc" --profiler 1 --profiler-path $PROFILER_FILE

echo "Done."