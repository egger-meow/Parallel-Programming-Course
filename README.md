
# Parallel Programming Course - Academic Portfolio

[![Final Grade: A+](https://img.shields.io/badge/Final%20Grade-A%2B-success)](https://github.com)
[![Course: 2024 Fall](https://img.shields.io/badge/Course-2024%20Fall-blue)](https://github.com)
[![Language: C/C++](https://img.shields.io/badge/Language-C%2FC%2B%2B-blue.svg)](https://isocpp.org)
[![OpenMP](https://img.shields.io/badge/OpenMP-Parallel%20Computing-green)](https://www.openmp.org)
[![SIMD](https://img.shields.io/badge/SIMD-Vectorization-orange)](https://en.wikipedia.org/wiki/SIMD)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)

## 🎯 Overview

This repository demonstrates comprehensive expertise in parallel programming, high-performance computing, and system optimization through hands-on implementation of parallel algorithms, SIMD vectorization, and performance profiling. All projects showcase deep understanding of computer architecture, memory hierarchies, and parallel computing paradigms essential for modern software engineering.

## 🚀 Core Competencies Demonstrated

### ⚡ High-Performance Computing
- **SIMD Vectorization** with custom intrinsics implementation
- **OpenMP parallelization** for multi-core systems
- **Memory optimization** and cache-friendly algorithms
- **Performance profiling** and bottleneck analysis

### 🔧 System Programming & Optimization
- **Low-level C/C++ programming** with manual memory management
- **Compiler optimization** understanding and utilization
- **Hardware-aware programming** for maximum efficiency
- **Parallel algorithm design** and implementation

### 🐳 Development Environment & Tools
- **Docker containerization** for reproducible builds
- **Makefile automation** for complex build systems
- **Performance measurement** and benchmarking tools
- **Cross-platform development** considerations

## 🛠 Technical Implementation Highlights

### HW0: Monte Carlo Pi Estimation - Foundation
[View Code](HW0/pi.cpp) | **Algorithm:** Monte Carlo Method

```cpp
// Core Implementation Features:
✓ Random number generation and statistical sampling
✓ Geometric probability calculation (circle-in-square)
✓ Large-scale computation handling (938M+ iterations)
✓ Numerical precision and convergence analysis
✓ Serial baseline implementation for comparison
```

**Technical Skills:** Statistical computing, random number generation, numerical methods, algorithm complexity analysis

---

### HW1 Part 1: SIMD Vectorization Mastery
[View Code](HW1/part1/) | **Focus:** Custom SIMD Implementation

```cpp
// Advanced Vectorization Features:
✓ Custom PPintrin library development
✓ SIMD instruction set utilization (SSE/AVX)
✓ Vector operations implementation from scratch
✓ Performance comparison: scalar vs vectorized code
✓ Memory alignment and data structure optimization
```

**Key Files:**
- `PPintrin.h/cpp`: Custom SIMD intrinsics library
- `vectorOP.cpp`: Optimized vector operations
- `main.cpp`: Performance benchmarking suite

**Technical Skills:** SIMD programming, instruction-level parallelism, performance optimization, assembly understanding

---

### HW1 Part 2: Performance Profiling & Optimization
[View Code](HW1/part2/) | **Focus:** System Performance Analysis

```c
// Performance Engineering Features:
✓ High-resolution timing with fasttime.h
✓ Cache performance optimization
✓ Memory access pattern analysis
✓ Compiler optimization flag comparison
✓ Performance bottleneck identification
```

**Technical Skills:** Performance profiling, cache optimization, timing measurement, system-level programming

---

### HW2: Parallel Monte Carlo Implementation
[View Code](HW2/part1/) | **Focus:** Multi-threading & Synchronization

```cpp
// Parallel Programming Features:
✓ OpenMP parallelization strategies
✓ Thread-safe random number generation
✓ Load balancing across multiple cores
✓ Parallel reduction operations
✓ Scalability analysis and performance metrics
```

**Technical Skills:** OpenMP, thread synchronization, parallel algorithms, load balancing, scalability analysis

---

### HW3: Conjugate Gradient Method - Advanced Numerical Computing
[View Code](HW3/part1/) | **Focus:** Scientific Computing Parallelization

```c
// Advanced Numerical Computing:
✓ Sparse matrix operations optimization
✓ Iterative solver implementation (Conjugate Gradient)
✓ Memory-efficient data structures
✓ Numerical stability and convergence criteria
✓ High-performance linear algebra operations
```

**Key Components:**
- `cg_impl.c`: Core conjugate gradient implementation
- `cg.c`: Main solver interface
- `grade.c`: Performance validation and testing

**Technical Skills:** Numerical linear algebra, sparse matrix computation, iterative methods, scientific computing

## 🔧 Technologies & Development Environment

| Category | Technologies |
|----------|-------------|
| **Programming Languages** | C, C++, Assembly (inline) |
| **Parallel Computing** | OpenMP, SIMD (SSE/AVX), Pthreads |
| **Development Tools** | GCC, Make, Docker, Performance Profilers |
| **System Programming** | Linux, Memory Management, Cache Optimization |
| **Mathematical Libraries** | Custom Linear Algebra, Numerical Methods |

## 📋 Key Learning Outcomes

### Computer Architecture Mastery
- **Instruction-Level Parallelism:** SIMD programming and vectorization techniques
- **Memory Hierarchy:** Cache optimization and memory access pattern analysis
- **Multi-Core Systems:** Thread management and synchronization strategies
- **Performance Measurement:** Profiling tools and benchmarking methodologies

### Advanced Programming Skills
- **Low-Level Optimization:** Assembly-level understanding and compiler optimization
- **System Programming:** Direct hardware interaction and resource management
- **Parallel Algorithm Design:** Scalable solutions for computational problems
- **Software Engineering:** Modular design, testing, and performance validation

### Mathematical & Numerical Computing
- **Monte Carlo Methods:** Statistical sampling and convergence analysis
- **Linear Algebra:** Sparse matrix operations and iterative solvers
- **Numerical Stability:** Precision handling and error analysis
- **Algorithm Complexity:** Time and space complexity optimization

## 🐳 Containerized Development Environment

```bash
# Build optimized development environment
docker build -t pp2024fall .

# Run with volume mounting for development
docker run --name pp -v /path/to/course:/home/workspace -it pp2024fall
```

**DevOps Skills:** Docker containerization, reproducible builds, cross-platform development, environment isolation

## 🎯 Professional Value Proposition

This coursework demonstrates:

1. **Systems Programming Excellence** - Deep understanding of computer architecture and low-level optimization
2. **Parallel Computing Expertise** - Practical experience with multi-threading, SIMD, and distributed computing
3. **Performance Engineering** - Ability to profile, analyze, and optimize computational bottlenecks
4. **Scientific Computing** - Implementation of advanced numerical algorithms for real-world applications
5. **Software Engineering Maturity** - Clean code, testing, containerization, and documentation practices

## 🚀 Industry Applications

The skills demonstrated directly apply to:

- **High-Performance Computing (HPC):** Scientific simulation, financial modeling, machine learning acceleration
- **Game Development:** Real-time graphics, physics engines, performance-critical systems
- **Systems Software:** Operating systems, database engines, compiler optimization
- **Embedded Systems:** IoT devices, automotive software, real-time control systems
- **Cloud Computing:** Distributed systems, containerization, scalable microservices
- **Machine Learning Infrastructure:** GPU acceleration, distributed training, model optimization

## 🔬 Advanced Technical Concepts Mastered

### Parallel Programming Paradigms
- **Shared Memory Programming:** OpenMP, thread synchronization, race condition handling
- **SIMD Programming:** Vector operations, data alignment, instruction throughput optimization
- **Performance Modeling:** Amdahl's Law, scalability analysis, parallel efficiency metrics

### Computer Systems Optimization
- **Cache-Aware Programming:** Data locality optimization, cache-friendly algorithms
- **Memory Management:** Manual allocation, alignment requirements, NUMA considerations
- **Compiler Optimization:** Understanding of compiler transformations and optimization flags

---

*This repository showcases advanced parallel programming skills, system-level optimization expertise, and high-performance computing knowledge essential for modern software engineering roles in performance-critical domains.*