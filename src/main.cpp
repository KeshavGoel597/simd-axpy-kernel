/**
 * AXPY (Y = alpha * X + Y) - Push 15: Fully inlined SIMD
 * 
 * Change: No helper function, SIMD code directly in compute()
 * 
 * Target: Intel Xeon Bronze 3204 with AVX2 + FMA support
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
    const double* __restrict__ X = x.data();
    double* __restrict__ Y = y.data();
    const size_t N = static_cast<size_t>(n);
    
    __m256d alpha_vec = _mm256_set1_pd(a);
    size_t i = 0;
    
    for (; i + 16 <= N; i += 16) {
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
    
    for (; i + 4 <= N; i += 4) {
        __m256d xv = _mm256_loadu_pd(&X[i]);
        __m256d yv = _mm256_loadu_pd(&Y[i]);
        yv = _mm256_fmadd_pd(alpha_vec, xv, yv);
        _mm256_storeu_pd(&Y[i], yv);
    }
    
    for (; i < N; ++i) {
        Y[i] = a * X[i] + Y[i];
    }

    return write_vec(y);
}
} // namespace solution
