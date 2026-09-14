#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=bench_mismatch
#SBATCH --output=./log/mismatch_%j.out
#SBATCH --error=./log/mismatch_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

module load singularity
module load openMPI

cd "$SLURM_SUBMIT_DIR"
mkdir -p ./bins ./profilings/container ./log

N_PARTICLES=100000
STEPS=100
INPUT_FILE="./bins/mismatch_ic_${N_PARTICLES}.bin"
N_RUNS=5

# Generate initial conditions if absent
if [ ! -f "$INPUT_FILE" ]; then
    echo "Generating initial conditions..."
    ./bins/main_native --help > /dev/null 2>&1  # test binary
    ./src/generate_initial_conditions --model 0 --n $N_PARTICLES --seed 42 --output $INPUT_FILE
fi

echo "=========================================================="
echo "Starting 3-Way Target Mismatch Benchmark (Runs: $N_RUNS, N: $N_PARTICLES)"
echo "=========================================================="

# 1. Benchmark Native AVX-512
PROF_NATIVE="./profilings/container/mismatch_native.txt"
rm -f $PROF_NATIVE
echo "=== Running Target 1: Native AVX-512 (-march=native) ==="
for i in $(seq 1 $N_RUNS); do
    echo "  Native AVX-512 run $i/$N_RUNS..."
    /usr/bin/time -a -o $PROF_NATIVE -f "\n--- OS / MPI Launch Time ---\nReal: %e seconds\nUser: %U seconds\nSys: %S seconds" \
    mpirun -np 1 ./bins/main_native --input $INPUT_FILE --nsteps $STEPS --dt 1e-4 --eps 0.05 --energy-every 100 \
                       --output ./bins/dummy.out --kernel "obrc" --profiler 1 --profiler-path $PROF_NATIVE --quiet
done


# 2. Benchmark Native AVX2 (v3)
PROF_V3="./profilings/container/mismatch_v3.txt"
rm -f $PROF_V3
echo "=== Running Target 2: Native AVX2 (-march=x86-64-v3) ==="
for i in $(seq 1 $N_RUNS); do
    echo "  Native AVX2 run $i/$N_RUNS..."
    /usr/bin/time -a -o $PROF_V3 -f "\n--- OS / MPI Launch Time ---\nReal: %e seconds\nUser: %U seconds\nSys: %S seconds" \
    mpirun -np 1 ./bins/main_v3 --input $INPUT_FILE --nsteps $STEPS --dt 1e-4 --eps 0.05 --energy-every 100 \
                   --output ./bins/dummy.out --kernel "obrc" --profiler 1 --profiler-path $PROF_V3 --quiet
done


# 3. Benchmark Container AVX2 (v3 inside Apptainer)
PROF_CONTAINER="./profilings/container/mismatch_container.txt"
rm -f $PROF_CONTAINER
echo "=== Running Target 3: Container AVX2 (nbody.sif) ==="
for i in $(seq 1 $N_RUNS); do
    echo "  Container AVX2 run $i/$N_RUNS..."
    /usr/bin/time -a -o $PROF_CONTAINER -f "\n--- OS / MPI Launch Time ---\nReal: %e seconds\nUser: %U seconds\nSys: %S seconds" \
    mpirun -np 1 singularity exec nbody.sif /app/src/main --input $INPUT_FILE --nsteps $STEPS --dt 1e-4 --eps 0.05 --energy-every 100 \
                                            --output ./bins/dummy.out --kernel "obrc" --profiler 1 --profiler-path $PROF_CONTAINER --quiet
done

rm -f ./bins/dummy.out
echo "Benchmark completed successfully."