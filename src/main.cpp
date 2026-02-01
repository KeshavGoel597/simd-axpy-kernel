#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <immintrin.h>
#include <studentlib.h>

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

void axpy_simd_fma(double alpha, const double* __restrict__ X, double* __restrict__ Y, size_t n) {
    const size_t AVX_WIDTH = 4;
    const size_t UNROLL_FACTOR = 8; // Increased from 4 to 8
    const size_t BLOCK_SIZE = AVX_WIDTH * UNROLL_FACTOR;

    __m256d alpha_vec = _mm256_set1_pd(alpha);
    size_t i = 0;

    // Main loop: 8x unrolled (32 elements per iteration)
    for (; i + BLOCK_SIZE <= n; i += BLOCK_SIZE) {
        // Prefetch further ahead
        _mm_prefetch(reinterpret_cast<const char*>(&X[i + 64]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&Y[i + 64]), _MM_HINT_T0);

        // Process 8 vectors (32 doubles)
        __m256d x0 = _mm256_loadu_pd(&X[i]);
        __m256d x1 = _mm256_loadu_pd(&X[i + 4]);
        __m256d x2 = _mm256_loadu_pd(&X[i + 8]);
        __m256d x3 = _mm256_loadu_pd(&X[i + 12]);
        __m256d x4 = _mm256_loadu_pd(&X[i + 16]);
        __m256d x5 = _mm256_loadu_pd(&X[i + 20]);
        __m256d x6 = _mm256_loadu_pd(&X[i + 24]);
        __m256d x7 = _mm256_loadu_pd(&X[i + 28]);

        __m256d y0 = _mm256_loadu_pd(&Y[i]);
        __m256d y1 = _mm256_loadu_pd(&Y[i + 4]);
        __m256d y2 = _mm256_loadu_pd(&Y[i + 8]);
        __m256d y3 = _mm256_loadu_pd(&Y[i + 12]);
        __m256d y4 = _mm256_loadu_pd(&Y[i + 16]);
        __m256d y5 = _mm256_loadu_pd(&Y[i + 20]);
        __m256d y6 = _mm256_loadu_pd(&Y[i + 24]);
        __m256d y7 = _mm256_loadu_pd(&Y[i + 28]);

        // FMA
        y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
        y1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
        y2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
        y3 = _mm256_fmadd_pd(alpha_vec, x3, y3);
        y4 = _mm256_fmadd_pd(alpha_vec, x4, y4);
        y5 = _mm256_fmadd_pd(alpha_vec, x5, y5);
        y6 = _mm256_fmadd_pd(alpha_vec, x6, y6);
        y7 = _mm256_fmadd_pd(alpha_vec, x7, y7);

        // Store
        _mm256_storeu_pd(&Y[i], y0);
        _mm256_storeu_pd(&Y[i + 4], y1);
        _mm256_storeu_pd(&Y[i + 8], y2);
        _mm256_storeu_pd(&Y[i + 12], y3);
        _mm256_storeu_pd(&Y[i + 16], y4);
        _mm256_storeu_pd(&Y[i + 20], y5);
        _mm256_storeu_pd(&Y[i + 24], y6);
        _mm256_storeu_pd(&Y[i + 28], y7);
    }

    // Secondary loop: 4 elements at a time
    for (; i + AVX_WIDTH <= n; i += AVX_WIDTH) {
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
}