#pragma once
#include <string>

namespace solution {
    // Computes y = alpha * x + y for length n. Returns path to output vector file.
    std::string compute(const std::string &x_path, const std::string &y_path, float alpha, int n);
}
