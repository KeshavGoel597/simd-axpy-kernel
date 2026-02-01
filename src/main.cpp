/**
 * AXPY (Y = alpha * X + Y) - v1 Structure with Prefetching
 * 
 * Push 7: Restore original v1 structure that achieved 25th place
 * - Separate helper function for SIMD loop
 * - Software prefetching for next iteration
 * - 4x unrolling with FMA
 */

#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <studentlib.h>
#include <immintrin.h>

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
 * SIMD AXPY with FMA and 4x Unrolling + Prefetching
 */
void axpy_simd_fma(double alpha, 
                   const double* __restrict__ X, 
                   double* __restrict__ Y, 
                   size_t n) {
    const __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    size_t i = 0;
    
    // Main SIMD loop: 4x unrolled (16 doubles per iteration)
    for (; i + 16 <= n; i += 16) {
        // Prefetch data for next iteration (32 doubles ahead = 256 bytes)
        _mm_prefetch(reinterpret_cast<const char*>(&X[i + 32]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&Y[i + 32]), _MM_HINT_T0);
        
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
        
        // FMA: Y = alpha * X + Y
        y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
        y1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
        y2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
        y3 = _mm256_fmadd_pd(alpha_vec, x3, y3);
        
        // Store back to Y
        _mm256_storeu_pd(&Y[i], y0);
        _mm256_storeu_pd(&Y[i + 4], y1);
        _mm256_storeu_pd(&Y[i + 8], y2);
        _mm256_storeu_pd(&Y[i + 12], y3);
    }
    
    // Handle remaining 4-element chunks
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(&X[i]);
        __m256d y = _mm256_loadu_pd(&Y[i]);
        y = _mm256_fmadd_pd(alpha_vec, x, y);
        _mm256_storeu_pd(&Y[i], y);
    }
    
    // Scalar cleanup
    for (; i < n; ++i) {
        Y[i] = alpha * X[i] + Y[i];
    }
}

} // namespace

std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n) {
    auto x = read_vec(x_path, n);
    auto y = read_vec(y_path, n);

    axpy_simd_fma(static_cast<double>(alpha), x.data(), y.data(), static_cast<size_t>(n));

    return write_vec(y);
}
} // namespace solution
