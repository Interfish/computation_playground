#include <immintrin.h>
#include <iostream>

void conv1d(float* input, float* kernelData, float* output,
            int inChannels, int outChannels, int kernelSize,
            int width) {
  for(int ic = 0; ic + 4 <= inChannels; ic += 4) {
    float* inWStart0 = input + ic * width;
    float* inWStart1 = inWStart0 + 1 * width;
    float* inWStart2 = inWStart0 + 2 * width;
    float* inWStart3 = inWStart0 + 3 * width;

    for(int oc = 0; oc < outChannels; oc++) {
      float* kStart0 = kernelData + ic * kernelSize * outChannels + oc * kernelSize;
      float* kStart1 = kStart0 + 1 * kernelSize * outChannels;
      float* kStart2 = kStart0 + 2 * kernelSize * outChannels;
      float* kStart3 = kStart0 + 3 * kernelSize * outChannels;

      for(int w = 0; w < width - kernelSize + 1; w++) {
        int outIndex = oc * (width - kernelSize + 1) + w;
        int innerW = 0;
        float sum = 0.0;
        for(; innerW + 4 <= kernelSize; innerW += 4) {
          auto iwOffside = w + innerW;
          auto i0w4 = _mm_loadu_ps(inWStart0 + iwOffside);
          auto i1w4 = _mm_loadu_ps(inWStart1 + iwOffside);
          auto i2w4 = _mm_loadu_ps(inWStart2 + iwOffside);
          auto i3w4 = _mm_loadu_ps(inWStart3 + iwOffside);

          auto k04 = _mm_loadu_ps(kStart0 + innerW);
          auto k14 = _mm_loadu_ps(kStart1 + innerW);
          auto k24 = _mm_loadu_ps(kStart2 + innerW);
          auto k34 = _mm_loadu_ps(kStart3 + innerW);

          auto result = _mm_mul_ps(i0w4, k04);
          result = _mm_add_ps(result, _mm_mul_ps(i1w4, k14));
          result = _mm_add_ps(result, _mm_mul_ps(i2w4, k24));
          result = _mm_add_ps(result, _mm_mul_ps(i3w4, k34));
          // reduce sum
          result = _mm_hadd_ps(result, result);
          result = _mm_hadd_ps(result, result);
          // accumulate
          sum += _mm_cvtss_f32(result);
          // std::cout << _mm_cvtss_f32(result) << std::endl;
        }
        for(;innerW < kernelSize; innerW++) {
          sum += *(inWStart0 + innerW) + *(inWStart1 + innerW)
                 + *(inWStart2 + innerW) + *(inWStart3 + innerW);
        }
        output[outIndex] += sum;
      }
    }
  }
  // todo, when inChannels is not multiple of 4
}