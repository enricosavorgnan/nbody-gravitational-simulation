#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=c_scale
#SBATCH --output=./log/c_scale_%x_%j.out
#SBATCH --error=./log/c_scale_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=GENOA
#SBATCH --exclusive

N_PARTICLES=${N_PARTICLES:-100000}
STEPS=${STEPS:-100}
PROFILER_FILE=${PROFILER_FILE:-"./profilings/container/s_${SLURM_NTASKS}.txt"}
INPUT_FILE="./bins/input_${N_PARTICLES}.bin"
OUTPUT_FILE="./bins/c_out_${N_PARTICLES}_${SLURM_NTASKS}.out"

module load singularity
module load openMPI

cd "$SLURM_SUBMIT_DIR"
mkdir -p ./bins ./profilings/container ./log

# Generate initial conditions if not already present
if [ ! -f "$INPUT_FILE" ]; then
    echo "Generating initial conditions..."
    singularity exec nbody.sif /app/src/generate_initial_conditions --model 0 --n $N_PARTICLES --seed 42 --output $INPUT_FILE
fi

echo "=========================================================="
echo "Container Job: $SLURM_JOB_ID | Tasks: $SLURM_NTASKS | N: $N_PARTICLES"
echo "=========================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Host-driven MPI execution wrapping the container binary
/usr/bin/time -a -o $PROFILER_FILE -f "\n--- OS / MPI Launch Time ---\nReal: %e seconds\nUser: %U seconds\nSys: %S seconds" \
mpirun -n $SLURM_NTASKS \
       --bind-to core \
       --map-by core \
       singularity exec nbody.sif /app/src/main \
           --input $INPUT_FILE \
           --nsteps $STEPS \
           --dt 1e-4 \
           --eps 0.05 \
           --energy-every 100 \
           --output $OUTPUT_FILE \
           --kernel "obrc" \
           --profiler 1 \
           --profiler-path $PROFILER_FILE \
           --quiet

echo "Done."