void im2col_nhwc(float* o, float* ok,  float* i, float* k, int iw, int ih, int kw, int kh, int channels, int batchSize, int kernelNums) {
  // Suppose input and kernel is NHWC
  // transform input
  int kernelSize = kw * kh;
  int ow = kw * kh * channels;
  int hConvs = ih - kh + 1;
  int wConvs = iw - kw + 1;
  for (int n = 0; n < batchSize; n++) {
    for (int hStart = 0; hStart < hConvs; hStart++) {
      for (int wStart = 0; wStart < wConvs; wStart++) {
        for (int y = 0; y < kh; y++) {
          for (int x = 0; x < kw; x++) {
            int iStart = n * iw * ih * channels + (hStart + y) * iw * channels + (wStart + x) * channels;
            int oStart = n * hConvs * wConvs * kernelSize * channels + hStart * wStart * kernelSize * channels;
            for (int c = 0; c < channels; c++) {
              o[oStart + c * kernelSize] = i[iStart + c];
            }
          }
        }
      }
    }
  }
}

void im2col_nchw(float* o, float* ok,  float* i, float* k, int iw, int ih, int kw, int kh, int channels, int batchSize, int kernelNums) {
  // Suppose input and kernel is NCHW
  // transform input
  int kernelSize = kw * kh;
  int ow = kw * kh * channels;
  int hConvs = ih - kh + 1;
  int wConvs = iw - kw + 1;
  for (int n = 0; n < batchSize; n++) {
    for (int c = 0; c < channels; c++) {
      int imageStart = n * channels * iw * ih + c * iw * ih;
      for (int hStart = 0; hStart < hConvs; hStart++) {
        for (int wStart = 0; wStart < wConvs; wStart++) {
          for (int y = 0; y < kh; y++) {
            for (int x = 0; x < kw; x++) {
              int iStart = imageStart + (hStart + y) * iw + wStart + x;
              int oStart = n * hConvs * wConvs * kernelSize * channels + hStart * wStart * kernelSize * channels + c * kernelSize + y * kw + x;
              o[oStart] = i[iStart];
            }
          }
        }
      }
    }
  }
}

void im2col_fluent_read(float* o, float* ok,  float* i, float* k, int iw, int ih, int kw, int kh, int channels, int batchSize, int kernelNums) {
  // Suppose input and kernel is NCHW
  // transform input
  for (int n = 0; n < batchSize; n++) {
    for (int c = 0; c < channels; c++) {
      for(int h = 0; h < ih; h++) {
        for (int w = 0; w < iw; w++) {

        }
      }
    }
  }
}