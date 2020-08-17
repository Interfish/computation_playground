#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <assert.h>

#include "../cpp/im2col.h"

void setArray(float* array, int size, float value) {
    for (int i = 0 ; i < size; i++)
        array[i] = value;
}

int main(int argc, char *argv[]) {
    int iw, ih, kw, kh, channels, batchSize, kernelNums;
    iw = atoi(argv[1]);
    ih = atoi(argv[2]);
    kw = atoi(argv[3]);
    kh = atoi(argv[4]);
    channels = atoi(argv[5]);
    batchSize = atoi(argv[6]);
    kernelNums = atoi(argv[7]);

    int inputSize = iw * ih * channels * batchSize;
    int kernelSize = kw * kh * kernelNums;
    int i2cMatrixSize = kw * kh * channels * (iw - kw + 1) * (ih - kh + 1) * batchSize;
    float *i = new float[inputSize];
    float *k = new float[kernelSize];
    float *i2cMatrix = new float[i2cMatrixSize];
    //float *i2cMatrix = (float*)malloc(sizeof(float) * i2cMatrixSize);

    setArray(i, inputSize, 1.0);
    setArray(k, kernelSize, 1.0);
    setArray(i2cMatrix, i2cMatrixSize, 0.0);

    // auto c_start = std::chrono::high_resolution_clock::now();
    // im2col_nhwc(i2cMatrix, i, k, iw, ih, kw, kh, channels, batchSize, kernelNums);
    // auto c_end = std::chrono::high_resolution_clock::now();
    // std::cout << std::fixed << std::setprecision(2) << "im2col_nhwc time used: "
    //           << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;

    // auto c_start = std::chrono::high_resolution_clock::now();
    // im2col_nchw(i2cMatrix, i, k, iw, ih, kw, kh, channels, batchSize, kernelNums);
    // auto c_end = std::chrono::high_resolution_clock::now();
    // std::cout << std::fixed << std::setprecision(2) << "im2col_nchw time used: "
    //           << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;

    auto c_start = std::chrono::high_resolution_clock::now();
    im2col_fluent_read(i2cMatrix, i, k, iw, ih, kw, kh, channels, batchSize, kernelNums);
    auto c_end = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(2) << "im2col_fluent_read time used: "
              << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;

    // for (int j = 0; j < i2cMatrixSize; j++) {
    //     // std::cout << j << std::endl;
    //     assert(i2cMatrix[j] == 1.0);
    // }

    delete[] i;
    delete[] k;
    delete[] i2cMatrix;
}