#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <studentlib.h>

namespace {
std::vector<double> make_vec(int n, std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    std::vector<double> data(n);
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
    return data;
}

std::string write_vec(const std::string &path, const std::vector<double> &v) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(v.data()), sizeof(double) * v.size());
    return path;
}

std::vector<double> read_vec(const std::string &path, int n) {
    std::vector<double> data(n);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(data.data()), sizeof(double) * n);
    return data;
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./tester <n> <alpha> [seed]\n";
        return 1;
    }
    const int n = std::atoi(argv[1]);
    const float alpha = std::atof(argv[2]);
    const int seed = (argc > 3) ? std::atoi(argv[3]) : std::random_device{}();
    std::mt19937 rng(seed);

    const std::string x_path = (std::filesystem::temp_directory_path() / ("axpy_x_" + std::to_string(n) + ".dat")).string();
    const std::string y_path = (std::filesystem::temp_directory_path() / ("axpy_y_" + std::to_string(n) + ".dat")).string();

    auto x = make_vec(n, rng);
    auto y = make_vec(n, rng);
    write_vec(x_path, x);
    write_vec(y_path, y);

    auto start = std::chrono::high_resolution_clock::now();
    const std::string out_path = solution::compute(x_path, y_path, alpha, n);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    auto student = read_vec(out_path, n);
    for (int i = 0; i < n; ++i) {
        double ref = static_cast<double>(alpha) * x[i] + y[i];
        double diff = std::abs(student[i] - ref);
        if (diff > 1e-9) {
            std::cerr << "Mismatch at " << i << " got " << student[i] << " expected " << ref << " diff " << diff << "\n";
            std::cout << -1 << std::endl;
            return 0;
        }
    }

    std::filesystem::remove(out_path);
    std::cout << "OK in " << duration.count() << " ns" << std::endl;
    return 0;
}
