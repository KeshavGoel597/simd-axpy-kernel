/**
 * AXPY (Y = alpha * X + Y) - Highly Optimized SIMD Implementation with FMA
 * 
 * Optimizations Applied:
 * 1. AVX2 256-bit vectors processing 4 doubles per instruction
 * 2. Alpha broadcasted to vector register OUTSIDE the loop
 * 3. FMA (Fused Multiply-Add): y = alpha * x + y in single instruction
 * 4. 4x loop unrolling (16 elements per iteration) to hide latency
 * 5. Software prefetching for upcoming data
 * 6. In-place update of Y array for cache efficiency
 * 
 * AXPY is a classic BLAS Level 1 operation and is memory-bound.
 * The key is to maximize memory bandwidth utilization through:
 * - Prefetching
 * - Large unroll factors
 * - Minimizing loop overhead
 * 
 * Target: Intel Xeon Bronze 3204 with AVX2 + FMA support
 */

#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <studentlib.h>
#include <immintrin.h>  // AVX2 + FMA intrinsics

namespace solution {
namespace {

std::vector<double> read_vec(const std::string &path, int n) {
    std::vector<double> data(n);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(data.data()), sizeof(double) * n);
    return data;
}

std::string write_vec(const std::vector<double> &data) {
    const std::string out_path = (std::filesystem::temp_directory_path() / "axpy_out.dat").string();
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(data.data()), sizeof(double) * data.size());
    return out_path;
}

/**
 * SIMD AXPY with FMA and 4x Unrolling
 * 
 * Operation: Y[i] = alpha * X[i] + Y[i]
 * 
 * Strategy:
 * - Broadcast alpha once before the loop
 * - Use FMA: Y = alpha * X + Y (single instruction, better precision)
 * - Process 16 elements per iteration (4 vectors × 4 doubles)
 * - Update Y in-place to maximize cache utilization
 */
void axpy_simd_fma(double alpha, 
                   const double* __restrict__ X, 
                   double* __restrict__ Y, 
                   size_t n) {
    const size_t AVX_WIDTH = 4;      // 4 doubles per __m256d
    const size_t UNROLL_FACTOR = 4;  // Process 4 vectors per iteration
    const size_t BLOCK_SIZE = AVX_WIDTH * UNROLL_FACTOR;  // 16 elements per iteration
    
    // Broadcast alpha to all lanes of the vector register
    // This is done ONCE outside the loop for efficiency
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    size_t i = 0;
    
    // Main SIMD loop: 4x unrolled with FMA
    // Processing 16 doubles per iteration
    for (; i + BLOCK_SIZE <= n; i += BLOCK_SIZE) {
        // Prefetch data for next iteration
        // Prefetch X (read-only) and Y (read-write)
        _mm_prefetch(reinterpret_cast<const char*>(&X[i + 32]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&Y[i + 32]), _MM_HINT_T0);
        
        // Load 4 vectors from X
        __m256d x0 = _mm256_loadu_pd(&X[i]);
        __m256d x1 = _mm256_loadu_pd(&X[i + 4]);
        __m256d x2 = _mm256_loadu_pd(&X[i + 8]);
        __m256d x3 = _mm256_loadu_pd(&X[i + 12]);
        
        // Load 4 vectors from Y (current values)
        __m256d y0 = _mm256_loadu_pd(&Y[i]);
        __m256d y1 = _mm256_loadu_pd(&Y[i + 4]);
        __m256d y2 = _mm256_loadu_pd(&Y[i + 8]);
        __m256d y3 = _mm256_loadu_pd(&Y[i + 12]);
        
        // FMA: Y = alpha * X + Y
        // Each FMA is independent, enabling out-of-order execution
        y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
        y1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
        y2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
        y3 = _mm256_fmadd_pd(alpha_vec, x3, y3);
        
        // Store updated Y vectors back
        _mm256_storeu_pd(&Y[i], y0);
        _mm256_storeu_pd(&Y[i + 4], y1);
        _mm256_storeu_pd(&Y[i + 8], y2);
        _mm256_storeu_pd(&Y[i + 12], y3);
    }
    
    // Secondary SIMD loop: handle remaining vectors (4 elements at a time)
    for (; i + AVX_WIDTH <= n; i += AVX_WIDTH) {
        __m256d x = _mm256_loadu_pd(&X[i]);
        __m256d y = _mm256_loadu_pd(&Y[i]);
        y = _mm256_fmadd_pd(alpha_vec, x, y);
        _mm256_storeu_pd(&Y[i], y);
    }
    
    // Scalar cleanup loop: handle remaining elements (0-3 elements)
    for (; i < n; ++i) {
        Y[i] = alpha * X[i] + Y[i];
    }
}

} // namespace

std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n) {
    auto x = read_vec(x_path, n);
    auto y = read_vec(y_path, n);

    // Execute optimized SIMD AXPY with FMA
    // Convert alpha from float to double for computation
    axpy_simd_fma(static_cast<double>(alpha), x.data(), y.data(), static_cast<size_t>(n));

    return write_vec(y);
}
} // namespace solution
