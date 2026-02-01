# AXPY - Optimization Report

## Target Architecture
- **CPU**: Intel Xeon Bronze 3204 (Dual Socket, 6 cores total @ 1.9GHz)
- **Memory**: 16GB DDR4 2666MHz RDIMM ECC
- **SIMD Support**: AVX2 (256-bit vectors), FMA3

## Operation
```
Y[i] = alpha * X[i] + Y[i]  for i = 0 to n-1
```

## Implementation Summary

### Data Type
- **double** (8 bytes per element)
- **__m256d** vectors (4 doubles per vector)

### Key Optimizations Applied

#### 1. AVX2 + FMA Instructions
```cpp
__m256d alpha_vec = _mm256_set1_pd(alpha);
// ...
_mm256_storeu_pd(&Y[i + j], _mm256_fmadd_pd(alpha_vec, x, y));
```
- FMA (`_mm256_fmadd_pd`) combines multiply and add in single instruction
- Better precision and throughput than separate mul + add
- Alpha broadcast done once outside the loop

#### 2. Loop Unrolling (8x inner loop within 32-element blocks)
```cpp
for (; i + 32 <= n; i += 32) {
    for (int j = 0; j < 32; j += 4) {
        __m256d x = _mm256_loadu_pd(&X[i + j]);
        __m256d y = _mm256_loadu_pd(&Y[i + j]);
        _mm256_storeu_pd(&Y[i + j], _mm256_fmadd_pd(alpha_vec, x, y));
    }
}
```
- Processes 32 doubles (256 bytes) per outer iteration
- Inner loop with 8 iterations of 4 doubles each
- Reduces loop overhead and improves instruction-level parallelism

#### 3. Efficient Memory Access
- Unaligned loads/stores (`_mm256_loadu_pd`, `_mm256_storeu_pd`)
- Works safely with `std::vector` which doesn't guarantee 32-byte alignment
- Modern CPUs have minimal penalty for unaligned access when data happens to be aligned

#### 4. Pre-allocated Vector with Reserve
```cpp
std::vector<double> data;
data.reserve(n);
data.resize(n);
```
- Avoids reallocation during read
- Single contiguous memory block

#### 5. Scalar Cleanup Loop
```cpp
for (; i < n; ++i) {
    Y[i] = alpha * X[i] + Y[i];
}
```
- Handles remaining elements (0-3) that don't fill a vector

### File I/O
- Standard C++ `std::ifstream`/`std::ofstream` with binary mode
- Bulk read/write of entire vector at once

## Performance Characteristics

### Memory Bandwidth
| Operation | Bytes per element |
|-----------|-------------------|
| Read X | 8 |
| Read Y | 8 |
| Write Y | 8 |
| **Total** | **24** |

### Compute Intensity
- 2 FLOPs per element (1 multiply + 1 add, fused)
- Arithmetic intensity: 2/24 = 0.083 FLOPs/byte
- **Memory-bound operation** - performance limited by memory bandwidth

## Design Decisions

### Why No Prefetching
- Tested software prefetching - minimal benefit on this CPU
- Hardware prefetcher handles sequential access patterns well

### Why No OpenMP
- AXPY timing includes file I/O which dominates
- Thread creation overhead significant for memory-bound operations
- Single-threaded SIMD often faster for this workload size

### Why Unaligned Loads
- `std::vector` doesn't guarantee 32-byte alignment
- Aligned loads would require custom allocation
- Performance difference minimal on modern CPUs
