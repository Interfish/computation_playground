#include <assert.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

#include "gemm.hpp"

int main(int argc, char *argv[]) {
    int m, k, n;
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);

    float* a = new float[m * k];
    int8_t* aq = new int8_t[m * k];
    float* b = new float[k * n];
    int8_t* bq = new int8_t[k * n];
    float* c = new float[m * n];
    float* cq = new float[m * n];

    std::mt19937 gen;
    std::uniform_real_distribution<float> u(-1, 1);
    for(int i = 0; i < m * k; i++)
      a[i] = u(gen);

    for(int i = 0; i < k * n; i++)
      b[i] = u(gen);

    std::vector<float> a_scales(m);
    std::vector<int> a_zeros(m);
    std::vector<float> b_scales(n);
    std::vector<int> b_zeros(n);

    for(int i = 0; i < m; i++) {
      a_scales[i] = 1.0f / 128.0f;
      a_zeros[i] = 0;
    }

    for(int i = 0; i < n; i++) {
      b_scales[i] = 1.0f / 128.0f;
      b_zeros[i] = 0;
    }

    auto c_start = std::chrono::high_resolution_clock::now();
    gemm_vanilla_transposed_b(a, b, c, m, n, k);
    auto c_end = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(2) << "gemm vanilla time used: "
              << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;


    float2int8_byrow(b, bq, k, b_scales, b_zeros);
    c_start = std::chrono::high_resolution_clock::now();
    float2int8_byrow(a, aq, k, a_scales, a_zeros);
    gemm_vanilla_transposed_b_quantized(aq, bq, cq, m, n, k, a_scales, b_scales, a_zeros, b_zeros);
    c_end = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(2) << "gemm vanilla quantized time used: "
              << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;


    float max_diff = 0;
    float max_diff_c = 0;
    float max_diff_cq = 0;
    float max_relative_diff = 0;
    float max_relative_diff_c = 0;
    float max_relative_diff_cq = 0;
    for(int i = 0 ; i < m * n; i++){
      float diff = std::abs(c[i] - cq[i]);
      float relative_diff = std::abs((c[i] - cq[i]) / c[i]);
      // std::cout << c[i] << ' ' << cq[i] << ' ' << diff << ' ' << c[i] - cq[i] << std::endl;
      if(diff > max_diff) {
        max_diff = diff;
        max_diff_c = c[i];
        max_diff_cq = cq[i];
      }
      if(relative_diff > max_relative_diff) {
        max_relative_diff = diff;
        max_relative_diff_c = c[i];
        max_relative_diff_cq = cq[i];
      }
    }
    std::cout << "Max diff: " << max_diff << " c: " << max_diff_c << " cq: " << max_diff_cq << std::endl;
    std::cout << "Max relative diff: " << max_relative_diff << " c: " << max_relative_diff_c << " cq: " << max_relative_diff_cq << std::endl;
}