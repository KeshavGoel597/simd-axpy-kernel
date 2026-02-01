/**
 * AXPY ULTIMATE - All optimizations combined
 * Y = alpha * X + Y
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

    double* X = (double*)aligned_alloc(64, sz);
    double* Y = (double*)aligned_alloc(64, sz);

    // PARALLEL I/O
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            int fx = open(x_path.c_str(), O_RDONLY);
            posix_fadvise(fx, 0, sz, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
            size_t total = 0;
            while (total < sz) {
                ssize_t r = read(fx, (char*)X + total, sz - total);
                if (r <= 0) break;
                total += r;
            }
            close(fx);
        }

        #pragma omp section
        {
            int fy = open(y_path.c_str(), O_RDONLY);
            posix_fadvise(fy, 0, sz, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
            size_t total = 0;
            while (total < sz) {
                ssize_t r = read(fy, (char*)Y + total, sz - total);
                if (r <= 0) break;
                total += r;
            }
            close(fy);
        }
    }

    // PARALLEL COMPUTE with 4 threads
    __m256d alpha_vec = _mm256_set1_pd(alpha_d);

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = (n / nthreads / 32) * 32;
        size_t start = tid * chunk;
        size_t end = (tid == nthreads - 1) ? n : start + chunk;

        // Process 32 elements at a time
        size_t i = start;
        for (; i + 32 <= end; i += 32) {
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

            _mm256_store_pd(&Y[i], _mm256_fmadd_pd(alpha_vec, x0, y0));
            _mm256_store_pd(&Y[i+4], _mm256_fmadd_pd(alpha_vec, x1, y1));
            _mm256_store_pd(&Y[i+8], _mm256_fmadd_pd(alpha_vec, x2, y2));
            _mm256_store_pd(&Y[i+12], _mm256_fmadd_pd(alpha_vec, x3, y3));
            _mm256_store_pd(&Y[i+16], _mm256_fmadd_pd(alpha_vec, x4, y4));
            _mm256_store_pd(&Y[i+20], _mm256_fmadd_pd(alpha_vec, x5, y5));
            _mm256_store_pd(&Y[i+24], _mm256_fmadd_pd(alpha_vec, x6, y6));
            _mm256_store_pd(&Y[i+28], _mm256_fmadd_pd(alpha_vec, x7, y7));
        }

        // Cleanup 4 at a time
        for (; i + 4 <= end; i += 4) {
            __m256d x = _mm256_load_pd(&X[i]);
            __m256d y = _mm256_load_pd(&Y[i]);
            _mm256_store_pd(&Y[i], _mm256_fmadd_pd(alpha_vec, x, y));
        }

        // Scalar cleanup
        for (; i < end; ++i) {
            Y[i] = alpha_d * X[i] + Y[i];
        }
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
