# ⚡ SIMD-Optimized AXPY Kernel

High-performance AXPY operation (`Y = α·X + Y`) using AVX2 + FMA intrinsics with aggressive loop unrolling, targeting Intel Xeon processors.

## Overview

This project implements an optimized BLAS Level-1 AXPY kernel for double-precision floating-point arrays. The implementation leverages:

- **Fused Multiply-Add (FMA)** — combines `α·X[i] + Y[i]` into a single instruction (`_mm256_fmadd_pd`) for better throughput and precision
- **32-Element Loop Unrolling** — processes 256 bytes per outer iteration, reducing loop overhead and maximizing instruction-level parallelism
- **AVX2 256-bit Vectors** — processes 4 doubles per SIMD instruction
- **Hierarchical Cleanup Loops** — 4-wide SIMD cleanup followed by scalar tail

## Target Architecture

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Xeon Bronze 3204 (6 cores @ 1.9 GHz) |
| **Memory** | 16 GB DDR4 2666 MHz RDIMM ECC |
| **SIMD** | AVX2 (256-bit), FMA3 |

## Build & Run

```bash
# One-step build and test
./runner_script.sh

# Manual build
mkdir -p build && cd build
cmake .. && make -j
./bin/tester 1048576 2.5 42   # <vector_length> <alpha> <seed>
```

### Prerequisites

- GCC/G++ with C++17 support
- CMake ≥ 3.11
- CPU with AVX2 + FMA support

## Project Structure

```
├── CMakeLists.txt          # Root build configuration
├── include/
│   └── studentlib.h        # Public API declaration
├── src/
│   ├── main.cpp            # Optimized AXPY kernel
│   └── CMakeLists.txt      # Library build config (with -O3, -mavx2, -mfma flags)
├── tester/
│   ├── tester.cpp          # Harness: generates inputs, verifies output (ε < 1e-9)
│   └── CMakeLists.txt      # Tester build config
├── runner_script.sh        # Convenience build + run script
└── report.md               # Detailed optimization report
```

## Key Optimizations

### 1. FMA Instructions
```cpp
_mm256_storeu_pd(&Y[i+j], _mm256_fmadd_pd(alpha_vec, x, y));
```
FMA fuses multiply + add into one operation — better precision (single rounding) and higher throughput than separate `mul` + `add`.

### 2. 32-Element Unrolled Loop
```cpp
for (; i + 32 <= n; i += 32) {       // 32 doubles = 256 bytes per iteration
    for (int j = 0; j < 32; j += 4)  // 8 FMA ops per outer iteration
```
Amortizes loop control overhead and enables out-of-order execution to overlap independent FMA operations.

### 3. Alpha Broadcast
```cpp
__m256d alpha_vec = _mm256_set1_pd(alpha);  // Broadcast once, reuse everywhere
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Bytes per element | 24 (Read X + Read Y + Write Y) |
| FLOPs per element | 2 (1 multiply + 1 add, fused) |
| Arithmetic intensity | 0.083 FLOPs/byte |
| **Bottleneck** | **Memory bandwidth** |

## Design Decisions

- **No OpenMP**: Thread creation overhead outweighs benefit for this memory-bound, I/O-inclusive workload
- **Unaligned loads**: `std::vector` doesn't guarantee 32-byte alignment; penalty is negligible on modern CPUs
- **No software prefetching**: Hardware prefetcher handles sequential access patterns efficiently

## License

Academic project — IIIT Hyderabad, Software Systems for Programmers (SPP), Spring 2026.
