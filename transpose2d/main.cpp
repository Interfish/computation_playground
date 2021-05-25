#include <iomanip>
#include <cassert>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

#include "common.h"
#include "transpose2d.h"

int main(int argc, char *argv[]) {
  int min_m, max_m, min_n, max_n, step, repeat;
  min_m = atoi(argv[1]);
  max_m = atoi(argv[2]);
  min_n = atoi(argv[3]);
  max_n = atoi(argv[4]);
  step = atoi(argv[5]);
  repeat = atoi(argv[6]);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for (int m = min_m; m <= max_m; m += step) {
    for (int n = min_n; n <= max_n; n += step) {
        float* in = new float[m * n];
        float* out = new float[n * m];
        for (int i = 0 ; i < m * n ; i++) {
          in[i] = float(i);
        }
        float* in_d, *out_d;
        int size = m * n * sizeof(float);

        CUDA_CHECK(cudaMalloc((void**)&in_d, size));
        CUDA_CHECK(cudaMalloc((void**)&out_d, size));

        CUDA_CHECK(cudaMemcpyAsync((void*)in_d, (void*)in, size, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto start = std::chrono::steady_clock::now();
        for(int i = 0 ; i < repeat; i++) {
          //computation_playground::transpose2d_naive(in_d, out_d, m, n, stream);
          computation_playground::transpose2d_tile(in_d, out_d, m, n, 8, 16, 1, 1, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto cost = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count() / float(repeat) / 1000.0f;
        std::cout << "m: " << m << ", n: " << n << ", time: " << cost << " us" << std::endl;

        CUDA_CHECK(cudaMemcpyAsync((void*)out, (void*)out_d, size, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < n; i++) {
          for(int j = 0; j < m; j++) {
            // std::cout << out[j * n + i] << " " << std::endl;
            assert(out[j * n + i] == float(i * m + j));
          }
        }

        CUDA_CHECK(cudaFree(in_d));
        CUDA_CHECK(cudaFree(out_d));

        delete[] in;
        delete[] out;
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}