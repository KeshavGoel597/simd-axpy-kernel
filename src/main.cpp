/**
 * AXPY - PARALLEL I/O VERSION
 * Y = alpha * X + Y
 * 
 * Strategy: Read X and Y files simultaneously!
 */

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <filesystem>
#include <immintrin.h>
#include <omp.h>
#include <studentlib.h>

namespace solution {

std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n) {
    const size_t sz = n * sizeof(double);
    const double alpha_d = static_cast<double>(alpha);

    // Allocate aligned buffers
    double* X = (double*)aligned_alloc(64, sz);
    double* Y = (double*)aligned_alloc(64, sz);

    // PARALLEL I/O - Read X and Y simultaneously!
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            int fx = open(x_path.c_str(), O_RDONLY);
            read(fx, X, sz);
            close(fx);
        }

        #pragma omp section
        {
            int fy = open(y_path.c_str(), O_RDONLY);
            read(fy, Y, sz);
            close(fy);
        }
    }

    // SIMD Compute with FMA - broadcast alpha
    __m256d alpha_vec = _mm256_set1_pd(alpha_d);

    // Main loop: 8x unrolling (32 elements per iteration)
    size_t i = 0;
    for (; i + 32 <= (size_t)n; i += 32) {
        // Process 8 vectors (32 doubles)
        __m256d x0 = _mm256_load_pd(&X[i]);
        __m256d x1 = _mm256_load_pd(&X[i+4]);
        __m256d x2 = _mm256_load_pd(&X[i+8]);
        __m256d x3 = _mm256_load_pd(&X[i+12]);
        __m256d x4 = _mm256_load_pd(&X[i+16]);
        __m256d x5 = _mm256_load_pd(&X[i+20]);
        __m256d x6 = _mm256_load_pd(&X[i+24]);
        __m256d x7 = _mm256_load_pd(&X[i+28]);

        __m256d y0 = _mm256_load_pd(&Y[i]);
        __m256d y1 = _mm256_load_pd(&Y[i+4]);
        __m256d y2 = _mm256_load_pd(&Y[i+8]);
        __m256d y3 = _mm256_load_pd(&Y[i+12]);
        __m256d y4 = _mm256_load_pd(&Y[i+16]);
        __m256d y5 = _mm256_load_pd(&Y[i+20]);
        __m256d y6 = _mm256_load_pd(&Y[i+24]);
        __m256d y7 = _mm256_load_pd(&Y[i+28]);

        // FMA: Y = alpha * X + Y
        y0 = _mm256_fmadd_pd(alpha_vec, x0, y0);
        y1 = _mm256_fmadd_pd(alpha_vec, x1, y1);
        y2 = _mm256_fmadd_pd(alpha_vec, x2, y2);
        y3 = _mm256_fmadd_pd(alpha_vec, x3, y3);
        y4 = _mm256_fmadd_pd(alpha_vec, x4, y4);
        y5 = _mm256_fmadd_pd(alpha_vec, x5, y5);
        y6 = _mm256_fmadd_pd(alpha_vec, x6, y6);
        y7 = _mm256_fmadd_pd(alpha_vec, x7, y7);

        // Store back
        _mm256_store_pd(&Y[i], y0);
        _mm256_store_pd(&Y[i+4], y1);
        _mm256_store_pd(&Y[i+8], y2);
        _mm256_store_pd(&Y[i+12], y3);
        _mm256_store_pd(&Y[i+16], y4);
        _mm256_store_pd(&Y[i+20], y5);
        _mm256_store_pd(&Y[i+24], y6);
        _mm256_store_pd(&Y[i+28], y7);
    }

    // Cleanup: 4 elements at a time
    for (; i + 4 <= (size_t)n; i += 4) {
        __m256d x = _mm256_load_pd(&X[i]);
        __m256d y = _mm256_load_pd(&Y[i]);
        y = _mm256_fmadd_pd(alpha_vec, x, y);
        _mm256_store_pd(&Y[i], y);
    }

    // Scalar cleanup
    for (; i < (size_t)n; ++i) {
        Y[i] = alpha_d * X[i] + Y[i];
    }

    // Write output
    std::string out = (std::filesystem::temp_directory_path() / "axpy_out.dat").string();
    int fo = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    write(fo, Y, sz);
    close(fo);

    free(X);
    free(Y);
    return out;
}

} // namespace solution
