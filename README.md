# ZeRO-2 CUDA/MPI/NCCL
# Zero Redundancy Optimizer (ZeRO) Stage 2 Implementation

A minimalist educational implementation of ZeRO Stage 2 using C++, CUDA, MPI, and NCCL.

## Overview

This project demonstrates the core concepts behind the **ZeRO (Zero Redundancy Optimizer) Stage 2** strategy through a "from-scratch" implementation. It shows how to distribute optimizer states and gradients across multiple GPUs to reduce memory redundancy during distributed training, enabling larger model support.

> **Note:** This is an educational prototype for understanding ZeRO mechanics, not a production-ready library.

## Key Features

* **ZeRO Stage 2 Implementation:**
  * Full parameter replication
  * Partitioned optimizer states (using SGD+Momentum)
  * Gradient partitioning via `ncclReduceScatter`
  * Parameter synchronization via `ncclAllGather`
* **Multi-GPU Execution:**
  * MPI for process initialization and management
  * NCCL for efficient GPU-to-GPU collective communication
* **Performance Optimizations:**
  * Flattened buffers for gradient reduction and parameter synchronization
  * Efficient communication patterns
* **Example Components:**
  * Simple linear model implementation
  * Basic training loop for demonstration

## System Requirements

* **CMake:** v3.18+ (recommended for proper CUDA/NCCL detection)
* **C++ Compiler:** C++17 compatible (GCC 9+)
* **CUDA Toolkit:** Tested with CUDA 12.4
* **MPI Implementation:** Compatible with your system (Open MPI, MPICH, etc.)
* **NVIDIA NCCL Library:** Development package (`libnccl-dev` or `nccl-devel`) must be installed
* **GPU:** Configured for NVIDIA A100s (can be modified in `CMakeLists.txt` for other architectures)

## Build Process

### 1. Environment Setup

Ensure all dependencies are correctly installed and configured.

### 2. Create a Build Script

Create a `build.sh` script with the following content:

```bash
#!/bin/bash

# --- Configure Paths ---
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
# Uncomment if libraries are in non-standard locations:
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/path/to/nccl/lib:$LD_LIBRARY_PATH

# Optional: Set NCCL path if not found automatically
# export NCCL_INSTALL_PATH="/usr"

echo "Using CUDA from: $CUDA_HOME"

# --- Build Steps ---
echo "Cleaning build directory..."
rm -rf build
mkdir build
cd build

echo "Running CMake..."
cmake .. \
    -D CMAKE_CUDA_COMPILER:PATH=$CUDA_HOME/bin/nvcc \
    -D CUDA_TOOLKIT_ROOT_DIR:PATH=$CUDA_HOME
    # Add -D CMAKE_PREFIX_PATH=$NCCL_INSTALL_PATH if needed

# Check CMake exit code
if [ $? -ne 0 ]; then
  echo "CMake configuration failed!"
  exit 1
fi

echo "Running Make..."
make -j

# Check Make exit code
if [ $? -ne 0 ]; then
  echo "Make build failed!"
  exit 1
fi

echo "Build successful!"
cd ..
```

### 3. Build the Project

```bash
chmod +x build.sh
./build.sh
```

## Running the Demo

Execute the compiled program using MPI with the number of processes matching your available GPUs:

```bash
# For 4 GPUs
mpirun -np 4 ./build/zero2_demo
```

## Troubleshooting

If you encounter build issues:

1. Verify CUDA and NCCL paths are correct in your environment
2. Ensure development headers for all dependencies are installed
3. Check compiler compatibility with CUDA version
4. Try explicitly providing paths with CMake's `-D` flags
