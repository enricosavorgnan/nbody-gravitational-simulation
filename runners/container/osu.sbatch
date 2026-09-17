#!/bin/bash -l
#SBATCH --job-name=osu_bench
#SBATCH --partition=GENOA
#SBATCH --account=dssc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=./log/osu_%j.out
#SBATCH --error=./log/osu_%j.err

# OSU Micro-Benchmark: Native vs. Container MPI Latency and Bandwidth
# Follows the exact methodology developed for the UniTS HPC Orfeo benchmarks.

set -euo pipefail

SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT"
mkdir -p ./log ./profilings/container/osu ./reports/container

module purge
module load openMPI
module load singularity

SIF="${SIF:-$SUBMIT/nbody.sif}"
if [ ! -f "$SIF" ]; then
    echo "ERROR: Singularity image not found at $SIF" >&2
    exit 1
fi

echo "=== ENVIRONMENT ==="
echo "Node:             $SLURMD_NODENAME"
echo "Job ID:           $SLURM_JOB_ID"
echo "Host mpirun:      $(which mpirun) — $(mpirun --version 2>/dev/null | head -1)"
echo "Container mpirun: $(singularity exec "$SIF" mpirun --version 2>/dev/null | head -1 || true)"
echo ""

# Build / Locate Official Native OSU Binaries
NAT_DIR="$SUBMIT/bins/osu/pt2pt"
if [ ! -x "$NAT_DIR/osu_latency" ] || [ ! -x "$NAT_DIR/osu_bw" ]; then
    echo "Native OSU binaries not found. Building official OSU Micro-Benchmarks natively..."
    bash "$SUBMIT/runners/container/build_osu.sh" "$SUBMIT/bins/osu"
fi

NAT_LATENCY="$NAT_DIR/osu_latency"
NAT_BW="$NAT_DIR/osu_bw"

# Locate / Prepare Official Container OSU Binaries
CONT_LATENCY=""
CONT_BW=""
for path_cand in "/opt/osu/pt2pt" "/opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt"; do
    if singularity exec "$SIF" test -x "$path_cand/osu_latency" 2>/dev/null; then
        CONT_LATENCY="$path_cand/osu_latency"
        CONT_BW="$path_cand/osu_bw"
        echo "Found official OSU inside container at: $path_cand"
        break
    fi
done

# If container image was built without OSU, build it using the container environment
if [ -z "$CONT_LATENCY" ]; then
    echo "OSU not found inside container image. Compiling official OSU 7.4 via container..."
    CONT_BUILD_PREFIX="$SUBMIT/bins/osu_container"
    mkdir -p "$CONT_BUILD_PREFIX"
    CONT_BUILD_PREFIX="$(cd "$CONT_BUILD_PREFIX" && pwd)"
    
    WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/osu_cont_build.XXXXXX")"
    cd "$WORKDIR"
    wget -q "https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz"
    tar xzf "osu-micro-benchmarks-7.4.tar.gz"
    cd "osu-micro-benchmarks-7.4"
    
    singularity exec "$SIF" ./configure CC=mpicc CXX=mpicxx --prefix="$CONT_BUILD_PREFIX/dist" >/dev/null
    singularity exec "$SIF" make -j"$(nproc 2>/dev/null || echo 4)" >/dev/null
    singularity exec "$SIF" make install >/dev/null
    
    mkdir -p "$CONT_BUILD_PREFIX/pt2pt"
    cp "$CONT_BUILD_PREFIX/dist/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency" \
       "$CONT_BUILD_PREFIX/dist/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw" \
       "$CONT_BUILD_PREFIX/pt2pt/"
    
    CONT_LATENCY="$CONT_BUILD_PREFIX/pt2pt/osu_latency"
    CONT_BW="$CONT_BUILD_PREFIX/pt2pt/osu_bw"
    cd "$SUBMIT"
    rm -rf "$WORKDIR"
fi

MPI_FLAGS="--map-by core --bind-to core"
REPS=5
OUTDIR="./profilings/container/osu"

echo "=== STARTING BENCHMARKS ($REPS repetitions) ==="
echo ""

# ---- Native Latency ----
echo "--- 1. Native osu_latency ---"
for rep in $(seq 1 $REPS); do
    echo "  rep ${rep}/${REPS}..."
    mpirun -np 2 ${MPI_FLAGS} "$NAT_LATENCY" > "$OUTDIR/native_latency_rep${rep}.txt" 2>&1
done

# ---- Container Latency ----
echo "--- 2. Container osu_latency ---"
for rep in $(seq 1 $REPS); do
    echo "  rep ${rep}/${REPS}..."
    singularity exec --env PMIX_MCA_psec=none "$SIF" mpirun -np 2 ${MPI_FLAGS} "$CONT_LATENCY" > "$OUTDIR/container_latency_rep${rep}.txt" 2>&1
done

# ---- Native Bandwidth ----
echo "--- 3. Native osu_bw ---"
for rep in $(seq 1 $REPS); do
    echo "  rep ${rep}/${REPS}..."
    mpirun -np 2 ${MPI_FLAGS} "$NAT_BW" > "$OUTDIR/native_bw_rep${rep}.txt" 2>&1
done

# ---- Container Bandwidth ----
echo "--- 4. Container osu_bw ---"
for rep in $(seq 1 $REPS); do
    echo "  rep ${rep}/${REPS}..."
    singularity exec --env PMIX_MCA_psec=none "$SIF" mpirun -np 2 ${MPI_FLAGS} "$CONT_BW" > "$OUTDIR/container_bw_rep${rep}.txt" 2>&1
done

echo ""
echo "=== OSU benchmarks complete. Outputs saved to $OUTDIR/ ==="

# Trigger plotting script
if [ -f "$SUBMIT/analysis/plot_osu.py" ]; then
    echo "Generating plots..."
    python3 "$SUBMIT/analysis/plot_osu.py" || true
fi
