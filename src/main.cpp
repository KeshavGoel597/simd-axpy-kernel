#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <immintrin.h>
#include <studentlib.h>
namespace solution {
namespace {
std::vector<double> read_vec(const std::string &path, int n) {
    std::vector<double> data;
    data.reserve(n); // Pre-allocate
    data.resize(n);
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
    __m256d alpha_vec = _mm256_set1_pd(alpha);
   size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        for (int j = 0; j < 32; j += 4) {
            __m256d x = _mm256_loadu_pd(&X[i + j]);
            __m256d y = _mm256_loadu_pd(&Y[i + j]);
            _mm256_storeu_pd(&Y[i + j], _mm256_fmadd_pd(alpha_vec, x, y));
        }
    }
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(&X[i]);
        __m256d y = _mm256_loadu_pd(&Y[i]);
        _mm256_storeu_pd(&Y[i], _mm256_fmadd_pd(alpha_vec, x, y));
    }
    for (; i < n; ++i) {
        Y[i] = alpha * X[i] + Y[i];
    }}}
    std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n) {
    auto x = read_vec(x_path, n);
    auto y = read_vec(y_path, n);

    axpy_simd_fma(static_cast<double>(alpha), x.data(), y.data(), static_cast<size_t>(n));

    return write_vec(y);
}

}