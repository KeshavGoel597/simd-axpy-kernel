/**
 * AXPY (Y = alpha * X + Y) - Push 9: Fast C-style I/O
 * 
 * Change: Replace C++ fstream with C fread/fwrite for faster I/O
 * 
 * Target: Intel Xeon Bronze 3204 with AVX2 + FMA support
 */

#include <vector>
#include <string>
#include <filesystem>
#include <studentlib.h>
#include <immintrin.h>
#include <cstdio>

namespace solution {
namespace {

std::vector<double> read_vec(const std::string &path, int n) {
    std::vector<double> data(n);
    FILE* f = fopen(path.c_str(), "rb");
    fread(data.data(), sizeof(double), n, f);
    fclose(f);
    return data;
}

std::string write_vec(const std::vector<double> &data) {
    const std::string out_path = (std::filesystem::temp_directory_path() / "axpy_out.dat").string();
    FILE* f = fopen(out_path.c_str(), "wb");
    fwrite(data.data(), sizeof(double), data.size(), f);
    fclose(f);
    return out_path;
}

void axpy_simd_fma(double alpha, 
                   const double* __restrict__ X, 
                   double* __restrict__ Y, 
                   size_t n) {
    const size_t AVX_WIDTH = 4;
    const size_t UNROLL_FACTOR = 4;
    const size_t BLOCK_SIZE = AVX_WIDTH * UNROLL_FACTOR;
    
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    size_t i = 0;
    
    for (; i + BLOCK_SIZE <= n; i += BLOCK_SIZE) {
        _mm_prefetch(reinterpret_cast<const char*>(&X[i + 32]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&Y[i + 32]), _MM_HINT_T0);
        
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
    
    for (; i + AVX_WIDTH <= n; i += AVX_WIDTH) {
        __m256d x = _mm256_loadu_pd(&X[i]);
        __m256d y = _mm256_loadu_pd(&Y[i]);
        y = _mm256_fmadd_pd(alpha_vec, x, y);
        _mm256_storeu_pd(&Y[i], y);
    }
    
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
