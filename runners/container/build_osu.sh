#!/usr/bin/env bash
# Build the official OSU micro-benchmarks NATIVELY on the cluster.

set -euo pipefail

OSU_VERSION=7.4
PREFIX="${1:-./bins/osu}"
URL="https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz"

command -v mpicc >/dev/null || { echo "mpicc not found: module load openMPI first" >&2; exit 2; }

# Make prefix absolute
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/osu_build.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

echo "== Downloading official OSU Micro-Benchmarks ${OSU_VERSION}..."
wget -q "$URL"
tar xzf "osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
cd "osu-micro-benchmarks-${OSU_VERSION}"

echo "== Configuring and compiling (CC=mpicc)..."
./configure CC=mpicc CXX=mpicxx --prefix="$PREFIX/dist" >/dev/null
make -j"$(nproc 2>/dev/null || echo 4)" >/dev/null
make install >/dev/null

mkdir -p "$PREFIX/pt2pt"
cp "$PREFIX/dist/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency" \
   "$PREFIX/dist/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw" \
   "$PREFIX/pt2pt/"

echo "== Installed official OSU benchmarks at: $PREFIX/pt2pt/{osu_latency,osu_bw}"
"$PREFIX/pt2pt/osu_latency" --help 2>&1 | head -2 || true
