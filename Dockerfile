# Use the latest Ubuntu LTS as the base image
FROM ubuntu:24.04

# Install essential build tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    git \
    ssh \
    vim \
    wget \
    libopenmpi-dev \
    openmpi-bin \
    libomp-dev \
    ocl-icd-opencl-dev \
    clinfo

# Optionally, install CUDA toolkit for GPU parallelism (only if you have a GPU)
# RUN apt-get install -y nvidia-cuda-toolkit

# Install profiling tools (e.g., gprof, valgrind, perf)
RUN apt-get install -y binutils valgrind linux-tools-common linux-tools-generic

# Set up a user (optional)
RUN useradd -ms /bin/bash jjmow

# Set the working directory
WORKDIR /home/jjmow

# Copy any local files to the container (if needed)
# COPY . /home/jjmow/

# Set environment variables (optional)
# ENV OMP_NUM_THREADS=4

# Command to run by default
CMD ["/bin/bash"]
