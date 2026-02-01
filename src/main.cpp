/**
 * AXPY with mmap - Zero-copy I/O
 * Y = alpha * X + Y
 */

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <filesystem>
#include <immintrin.h>
#include <studentlib.h>

namespace solution {

std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n) {
    const size_t sz = n * sizeof(double);
    const double alpha_d = static_cast<double>(alpha);

    // Open and mmap X (read-only)
    int fx = open(x_path.c_str(), O_RDONLY);
    const double* X = (const double*)mmap(nullptr, sz, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fx, 0);
    close(fx);

    // Allocate Y buffer
    double* Y = (double*)aligned_alloc(64, sz);

    // Read Y file
    int fy = open(y_path.c_str(), O_RDONLY);
    read(fy, Y, sz);
    close(fy);

    // SIMD Compute
    __m256d alpha_vec = _mm256_set1_pd(alpha_d);

    size_t i = 0;
    for (; i + 32 <= (size_t)n; i += 32) {
        // Unroll 8x (32 elements)
        for (int j = 0; j < 32; j += 4) {
            __m256d x = _mm256_loadu_pd(&X[i+j]);
            __m256d y = _mm256_load_pd(&Y[i+j]);
            _mm256_store_pd(&Y[i+j], _mm256_fmadd_pd(alpha_vec, x, y));
        }
    }

    // Cleanup
    for (; i + 4 <= (size_t)n; i += 4) {
        __m256d x = _mm256_loadu_pd(&X[i]);
        __m256d y = _mm256_load_pd(&Y[i]);
        _mm256_store_pd(&Y[i], _mm256_fmadd_pd(alpha_vec, x, y));
    }

    for (; i < (size_t)n; ++i) {
        Y[i] = alpha_d * X[i] + Y[i];
    }

    munmap((void*)X, sz);

    // Write output
    std::string out = (std::filesystem::temp_directory_path() / "axpy_out.dat").string();
    int fo = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    write(fo, Y, sz);
    close(fo);

    free(Y);
    return out;
}

} // namespace solution
