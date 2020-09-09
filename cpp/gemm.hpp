#include <vector>

using std::vector;

void gemm_vanilla_transposed_b(float* a, float* transposed_b, float* c, int m, int n, int k) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      for(int l = 0; l < k; l++) {
        c[n * i + j] += a[i * k + l] * transposed_b[j * k + l];
      }
    }
  }
}

void float2int8_byrow(float* src, int8_t* dst, int row_size, vector<float> &scales, vector<int> &zeros) {
  for(int i = 0; i < scales.size();i++) {
    for(int j = 0; j < row_size; j++) {
      dst[i * row_size + j] = static_cast<int8_t>(src[i * row_size + j] / scales[i] + zeros[i]);
    }
  }
}

void int82float_byrow(int8_t* src, float* dst, int row_size, vector<float> &scales, vector<int> &zeros) {
  for(int i = 0; i < scales.size();i++) {
    for(int j = 0; j < row_size; j++) {
      dst[i * row_size + j] = static_cast<float>((int(src[i * row_size + j]) + zeros[i]) * scales[i]);
    }
  }
}

void gemm_vanilla_transposed_b_quantized(int8_t* a, int8_t* transposed_b, float* c,
                                         int m, int n, int k,
                                         vector<float> &scales_a, vector<float> &scales_b,
                                         vector<int> &zeros_a, vector<int> &zeros_b) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      float scale = scales_a[i] * scales_b[j];
      int zeros_sum = k * zeros_a[i] * zeros_b[j];
      int ab_sum = 0;
      int a_sum = 0;
      int b_sum = 0;
      for(int l = 0; l < k; l++) {
        ab_sum += a[i * k + l] * transposed_b[j * k + l];
        a_sum += a[i * k + l];
        b_sum += transposed_b[j * k + l];
      }
      c[n * i + j] = float(ab_sum + a_sum + b_sum + zeros_sum) * scale;
    }
  }
}