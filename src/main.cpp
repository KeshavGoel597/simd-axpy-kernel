/**
 * AXPY (Y = alpha * X + Y) - Optimized SIMD Implementation
 * 
 * Key insight: AXPY is memory-bound. Focus on:
 * 1. Minimize memory allocations
 * 2. Simple vectorization without OpenMP overhead
 * 3. Let compiler optimize with -ffast-math -march=native
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
    auto x = read_vec(x_path, n);
    auto y = read_vec(y_path, n);
    
    const double a = static_cast<double>(alpha);
    const size_t N = static_cast<size_t>(n);
    
    // Broadcast alpha to vector register once
    const __m256d alpha_vec = _mm256_set1_pd(a);
    
    const double* X = x.data();
    double* Y = y.data();
    
    size_t i = 0;
    
    // Main SIMD loop: 4x unrolled (16 elements per iteration)
    // Using unaligned loads since std::vector may not be 32-byte aligned
    for (; i + 16 <= N; i += 16) {
        __m256d x0 = _mm256_loadu_pd(&X[i]);
        __m256d x1 = _mm256_loadu_pd(&X[i + 4]);
        __m256d x2 = _mm256_loadu_pd(&X[i + 8]);
        __m256d x3 = _mm256_loadu_pd(&X[i + 12]);
        
        __m256d y0 = _mm256_loadu_pd(&Y[i]);
        __m256d y1 = _mm256_loadu_pd(&Y[i + 4]);
        __m256d y2 = _mm256_loadu_pd(&Y[i + 8]);
        __m256d y3 = _mm256_loadu_pd(&Y[i + 12]);
        
        y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
        y1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
        y2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
        y3 = _mm256_fmadd_pd(alpha_vec, x3, y3);
        
        _mm256_storeu_pd(&Y[i], y0);
        _mm256_storeu_pd(&Y[i + 4], y1);
        _mm256_storeu_pd(&Y[i + 8], y2);
        _mm256_storeu_pd(&Y[i + 12], y3);
    }
    
    // Handle remaining 4-element chunks
    for (; i + 4 <= N; i += 4) {
        __m256d xv = _mm256_loadu_pd(&X[i]);
        __m256d yv = _mm256_loadu_pd(&Y[i]);
        yv = _mm256_fmadd_pd(alpha_vec, xv, yv);
        _mm256_storeu_pd(&Y[i], yv);
    }
    
    // Scalar cleanup
    for (; i < N; ++i) {
        Y[i] = a * X[i] + Y[i];
    }

    return write_vec(y);
}
} // namespace solution
