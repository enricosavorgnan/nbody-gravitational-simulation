FROM ubuntu:22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

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
         CFLAGS="-O3 -march=x86-64-v3 -Wall -Wextra -Wpedantic -fopenmp -ffast-math -flto -D_POSIX_C_SOURCE=199309L"