#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=bench_launch
#SBATCH --output=./log/launch_%j.out
#SBATCH --error=./log/launch_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load singularity

N_RUNS=15
OUT_FILE="./reports/container/launch_overhead.csv"
mkdir -p ./reports/container ./log

echo "run,type,real_sec" > $OUT_FILE

echo "=== Benchmarking Bare-Metal /bin/true ==="
for i in $(seq 1 $N_RUNS); do
    T=$(/usr/bin/time -f "%e" /bin/true 2>&1)
    echo "$i,native,$T" >> $OUT_FILE
done

echo "=== Benchmarking Singularity exec /bin/true ==="
for i in $(seq 1 $N_RUNS); do
    T=$(/usr/bin/time -f "%e" singularity exec nbody.sif /bin/true 2>&1)
    echo "$i,container,$T" >> $OUT_FILE
done

echo "Done. Results stored in $OUT_FILE"