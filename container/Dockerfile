FROM ubuntu:22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    make \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Compile official OSU Micro-Benchmarks 7.4
ARG OSU_VERSION=7.4
RUN wget -q https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz \
 && tar xzf osu-micro-benchmarks-${OSU_VERSION}.tar.gz \
 && cd osu-micro-benchmarks-${OSU_VERSION} \
 && ./configure CC=mpicc CXX=mpicxx --prefix=/opt/osu \
 && make -j"$(nproc)" && make install \
 && mkdir -p /opt/osu/pt2pt \
 && cp /opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency \
       /opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw \
       /opt/osu/pt2pt/ \
 && cd .. && rm -rf osu-micro-benchmarks-${OSU_VERSION}*

ENV PATH=/opt/osu/pt2pt:$PATH

# Define build arguments that the user can override
ARG USE_PAPI=0
ARG USE_MPI=1
ARG PRECISION=double

# Set up the working directory and copy your repository
WORKDIR /app
COPY . /app/

# Build the MPI experiment
RUN cd src && \
    make clean && \
    make clean-folders && \
    make all \
         USE_MPI=${USE_MPI} \
         PRECISION=${PRECISION} \
         CFLAGS="-O3 -march=x86-64-v3 -Wall -Wextra -Wpedantic -fopenmp -ffast-math -flto -D_POSIX_C_SOURCE=199309L -DUSE_MPI=1 -DNBODY_USE_DOUBLE"