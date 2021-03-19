#include <random>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"

__global__ void bankConflictTest(float* fake_result) {
    __shared__ float sm[128];
    int i = threadIdx.x / 4;
    fake_result[i] = sm[i];
}

int main(int argc, char *argv[]) {
    float *fake_result_d;
    cudaMalloc((void**)&fake_result_d, sizeof(float) * 128);
    dim3 block(128);
    dim3 grid(1);
    bankConflictTest<<<grid, block>>>(fake_result_d);
    cudaFree(fake_result_d);
}