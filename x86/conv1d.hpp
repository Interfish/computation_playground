#include <immintrin.h>

void conv1d(float* input, float* kernelData, float* output,
            int inChannels, int outChannels, int kernelSize,
            int width) {
  for(int ic = 0; ic < inChannels; ic += 4) {
    float* inWStart0 = input + ic * width;
    float* inWStart1 = inWStart0 + 1 * width;
    float* inWStart2 = inWStart0 + 2 * width;
    float* inWStart3 = inWStart0 + 3 * width;

    for(int oc = 0; oc < outChannels; oc++) {
      float* kStart0 = kernelData + ic * 4 * kernelSize * outChannels + oc * kernelSize;
      float* kStart1 = kStart0 + 1 * kernelSize * outChannels;
      float* kStart2 = kStart0 + 2 * kernelSize * outChannels;
      float* kStart3 = kStart0 + 3 * kernelSize * outChannels;

      for(int w = 0; w < width - kernelSize + 1; w++) {
        int outIndex = oc * (width - kernelSize + 1) + w;
        output[outIndex] = 0.0f;
        int innnerW = 0;
        for(; innnerW < kernelSize; innnerW += 4) {
          auto iwOffside = w + innnerW;
          auto i0w4 = _mm_loadu_ps(inWStart0 + iwOffside);
          auto i1w4 = _mm_loadu_ps(inWStart1 + iwOffside);
          auto i2w4 = _mm_loadu_ps(inWStart2 + iwOffside);
          auto i3w4 = _mm_loadu_ps(inWStart3 + iwOffside);

          auto k04 = _mm_loadu_ps(kStart0 + innnerW);
          auto k14 = _mm_loadu_ps(kStart1 + innnerW);
          auto k24 = _mm_loadu_ps(kStart2 + innnerW);
          auto k34 = _mm_loadu_ps(kStart3 + innnerW);

          auto result = _mm_mul_ps(i0w4, k04);
          result = _mm_mul_ps(result, _mm_mul_ps(i1w4, k14));
          result = _mm_mul_ps(result, _mm_mul_ps(i2w4, k24));
          result = _mm_mul_ps(result, _mm_mul_ps(i3w4, k34));
          // reduce sum
          result = _mm_hadd_ps(result, result);
          result = _mm_hadd_ps(result, result);
          // accumulate
          output[outIndex] += _mm_cvtss_f32(result);
        }
        for(;innnerW < kernelSize; innnerW++) {
          output[outIndex] += *(inWStart0 + innnerW) + *(inWStart1 + innnerW)
                            + *(inWStart2 + innnerW) + *(inWStart3 + innnerW);
        }
      }
    }
  }
  // todo, when inChannels is not multiple of 4
}