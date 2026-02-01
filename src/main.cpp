/**
 * AXPY (Y = alpha * X + Y) - Maximally Optimized AVX2 Implementation
 * 
 * Constraints:
 * - Data type: double (8 bytes)
 * - Vector: __m256d (4 doubles per vector)
 * - Unrolling: 4x (16 doubles per iteration, ~9 registers used)
 * - Memory: Unaligned loads/stores (std::vector not guaranteed aligned)
 * - FMA: _mm256_fmadd_pd
 * - No prefetching (hardware prefetcher handles linear access)
 * 
 * Target: Intel Xeon Bronze 3204 (AVX2 + FMA3)
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

} // namespace

std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n) {
    std::vector<double> x = read_vec(x_path, n);
    std::vector<double> y = read_vec(y_path, n);
    
    // Cast alpha to double and broadcast to all 4 lanes
    const __m256d alpha_vec = _mm256_set1_pd(static_cast<double>(alpha));
    
    const double* X = x.data();
    double* Y = y.data();
    const size_t N = static_cast<size_t>(n);
    
    size_t i = 0;
    
    // Main SIMD loop: 4x unrolled (16 doubles per iteration)
    // Register usage: 4 for X, 4 for Y, 1 for alpha = 9 registers (fits in 16 YMM)
    for (; i + 16 <= N; i += 16) {
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
        
        // Store 4 vectors back to Y
        _mm256_storeu_pd(&Y[i], y0);
        _mm256_storeu_pd(&Y[i + 4], y1);
        _mm256_storeu_pd(&Y[i + 8], y2);
        _mm256_storeu_pd(&Y[i + 12], y3);
    }
    
    // Tail: handle remaining elements (0-15) with scalar loop
    const double a = static_cast<double>(alpha);
    for (; i < N; ++i) {
        Y[i] = a * X[i] + Y[i];
    }

    return write_vec(y);
}
} // namespace solution
