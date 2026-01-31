# AXPY SIMD Optimization Report

## Target Architecture
- **CPU**: Intel Xeon Bronze 3204 (Dual Processors)
- **Memory**: 16GB DDR4 2666MHz RDIMM ECC
- **SIMD Support**: AVX2 (256-bit vectors), FMA3

## Operation
```
Y[i] = α × X[i] + Y[i]  for i = 0 to n-1
```

AXPY is a BLAS Level 1 operation (Basic Linear Algebra Subprogram).

## Optimization Techniques Applied

### 1. Alpha Broadcast (Hoisted Outside Loop)
```cpp
// ONCE before the loop - not inside!
__m256d alpha_vec = _mm256_set1_pd(alpha);
```

**Why this matters**:
- `_mm256_set1_pd` broadcasts scalar to all vector lanes
- Doing this once avoids repeated broadcast overhead
- The alpha vector is reused for all iterations

### 2. FMA for Fused Operation
```cpp
// Y = alpha * X + Y in single instruction
y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
```

**Benefits of FMA**:
- Single instruction instead of multiply then add
- Better numerical precision (single rounding)
- Higher throughput on FMA-capable CPUs
- Perfect fit for AXPY operation

### 3. Loop Unrolling (4x Factor)
```cpp
// Process 16 elements per iteration
for (; i + BLOCK_SIZE <= n; i += BLOCK_SIZE) {
    // Load 4 vectors from X
    __m256d x0 = _mm256_loadu_pd(&X[i]);
    __m256d x1 = _mm256_loadu_pd(&X[i + 4]);
    __m256d x2 = _mm256_loadu_pd(&X[i + 8]);
    __m256d x3 = _mm256_loadu_pd(&X[i + 12]);
    
    // Load 4 vectors from Y
    __m256d y0 = _mm256_loadu_pd(&Y[i]);
    __m256d y1 = _mm256_loadu_pd(&Y[i + 4]);
    __m256d y2 = _mm256_loadu_pd(&Y[i + 8]);
    __m256d y3 = _mm256_loadu_pd(&Y[i + 12]);
    
    // 4 independent FMA operations
    y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
    y1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
    y2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
    y3 = _mm256_fmadd_pd(alpha_vec, x3, y3);
    
    // Store 4 vectors back to Y
    _mm256_storeu_pd(&Y[i], y0);
    _mm256_storeu_pd(&Y[i + 4], y1);
    _mm256_storeu_pd(&Y[i + 8], y2);
    _mm256_storeu_pd(&Y[i + 12], y3);
}
```

**Benefits**:
- Amortizes loop overhead
- Enables instruction-level parallelism
- All 4 FMAs are independent (no dependency chain)
- Better scheduling of loads and stores

### 4. In-Place Update Pattern
```cpp
// Read Y, modify, write back Y
__m256d y = _mm256_loadu_pd(&Y[i]);
y = _mm256_fmadd_pd(alpha_vec, x, y);
_mm256_storeu_pd(&Y[i], y);
```

**Benefits**:
- Y array stays in cache (read-modify-write)
- Better cache utilization than separate output array
- Reduces memory traffic

### 5. Software Prefetching
```cpp
_mm_prefetch(reinterpret_cast<const char*>(&X[i + 32]), _MM_HINT_T0);
_mm_prefetch(reinterpret_cast<const char*>(&Y[i + 32]), _MM_HINT_T0);
```

**Strategy**:
- Prefetch X (read-only)
- Prefetch Y (read-write)
- Distance: 32 elements = 256 bytes ahead

## Performance Analysis

### Memory Access Pattern
| Array | Access Type | Bytes |
|-------|-------------|-------|
| X | Read | 8n |
| Y | Read | 8n |
| Y | Write | 8n |
| **Total** | | **24n** |

### Arithmetic Intensity
- Operations: 2n FLOPs (n multiplies + n adds, or n FMAs)
- Memory: 24n bytes
- Intensity = 2n / 24n = 0.083 FLOP/byte
- **Conclusion**: Heavily memory-bound

### Bottleneck Analysis
AXPY is a classic example of a **memory-bound** operation:
- Very low arithmetic intensity
- Performance limited by memory bandwidth, not compute
- Optimizations focus on maximizing memory throughput

## Code Structure

```cpp
void axpy_simd_fma(double alpha, const double* X, double* Y, size_t n) {
    // Broadcast alpha once
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    // Main unrolled SIMD loop
    for (i = 0; i + 16 <= n; i += 16) {
        // Prefetch
        // Load 4 vectors from X and Y
        // 4 FMA operations
        // Store 4 vectors to Y
    }
    
    // Secondary SIMD loop (4 elements)
    for (; i + 4 <= n; i += 4) { ... }
    
    // Scalar cleanup (0-3 elements)
    for (; i < n; ++i) {
        Y[i] = alpha * X[i] + Y[i];
    }
}
```

## Comparison: Naive vs Optimized

### Naive Implementation
```cpp
for (int i = 0; i < n; ++i) {
    y[i] = alpha * x[i] + y[i];
}
```
- Scalar operations
- No vectorization
- Compiler may auto-vectorize but suboptimally

### Optimized Implementation
- Explicit AVX2 intrinsics
- FMA fusion
- 4x unrolling
- Prefetching
- Controlled memory access pattern

## Expected Speedup
- **Baseline (scalar)**: 1x
- **AVX2 vectorization**: ~4x
- **With FMA + unrolling**: ~4-6x
- **Actual**: Limited by memory bandwidth

For memory-bound operations like AXPY, the speedup saturates once memory bandwidth is saturated.

## Compiler Flags
- `-ffast-math`: Enables FMA generation
- `-march=native`: Enables AVX2/FMA
- `-funroll-loops`: Additional unrolling

## Future Optimizations

### 1. OpenMP Parallelization
```cpp
#pragma omp parallel for
for (size_t i = 0; i < n; i += BLOCK_SIZE) {
    // SIMD kernel
}
```
Would utilize both Xeon Bronze 3204 processors.

### 2. Non-Temporal Stores
```cpp
_mm256_stream_pd(&Y[i], y0);
```
If Y won't be read again soon, streaming stores bypass cache.

### 3. Cache Blocking
For arrays larger than LLC, process in cache-sized blocks.

### 4. NUMA Awareness
Ensure data is allocated on the same NUMA node as the processing core.
