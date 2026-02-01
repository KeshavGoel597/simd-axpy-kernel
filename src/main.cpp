/**
 * AXPY (Y = alpha * X + Y) - Push 17: mmap for faster file I/O
 * 
 * Change: Use mmap instead of fstream for reading input files
 * 
 * Target: Intel Xeon Bronze 3204 with AVX2 + FMA support
 */

#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <filesystem>
#include <studentlib.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace solution {
namespace {

// mmap-based read for faster I/O
std::vector<double> read_vec_mmap(const std::string &path, int n) {
    int fd = open(path.c_str(), O_RDONLY);
    size_t size = n * sizeof(double);
    
    void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    std::vector<double> data(n);
    std::memcpy(data.data(), mapped, size);
    
    munmap(mapped, size);
    close(fd);
    
    return data;
}

std::string write_vec(const std::vector<double> &data) {
    const std::string out_path = (std::filesystem::temp_directory_path() / "axpy_out.dat").string();
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(data.data()), sizeof(double) * data.size());
    return out_path;
}

void axpy_simd_fma(double alpha, 
                   const double* __restrict__ X, 
                   double* __restrict__ Y, 
                   size_t n) {
    const size_t BLOCK_SIZE = 16;
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
    
    for (; i + 4 <= n; i += 4) {
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
    auto x = read_vec_mmap(x_path, n);
    auto y = read_vec_mmap(y_path, n);

    axpy_simd_fma(static_cast<double>(alpha), x.data(), y.data(), static_cast<size_t>(n));

    return write_vec(y);
}
} // namespace solution
