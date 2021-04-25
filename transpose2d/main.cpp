#include <iomanip>

#include "cuda_runtime.h"

int main(int argc, char *argv[]) {
  int min_m, max_m, min_n, max_m, step;
  min_m = atoi(argv[1]);
  max_m = atoi(argv[2]);
  min_n = atoi(argv[3]);
  max_m = atoi(argv[4]);
  step = atoi(argv[5]);

  for (int m = min_m; m < max_m; m += step) {
    for (int n = min_n, n < max_n; n += step) {
        float* tensor = new float[m * n];
        for (int i = 0 ; i < m * n ; i++) {
          tensor[i] = float(i);
        }
        float* out = new float[m * n];
        float* tenosr_d, out_d;

        int size = m * n * sizeof(float);

        cudaMalloc((void**)&tenosr_d, size);
        cudaMalloc((void**)&out_d, size);

        cudaMemcpy(tenosr_d, tensor, size, cudaMemcpyHostToDevice);

        transpose2d<<<grid, block>>>(tensor_d, out_d, m, n);

        cudaMemcpy(out_d, out, size, cudaMemcpyHostToDevice);

        for(int i = 0; i < n; i++) {
          for(int j = 0; j < m; j++) {
            assert(out[j * n + i] == float(i * m + j));
          }
        }

        cudaFree(tenosr_d);
        cudaFree(out_d);

        delete[] tensor;
        delete[] out;
    }
  }
}