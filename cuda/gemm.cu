#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <ctime>
#include "cuda_runtime.h"

#include "common.cuh"
#include "gemmk.cuh"

int main(int argc, char *argv[]) {
    int m, k, n, kernelSize;
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);
    kernelSize = atoi(argv[4]);

    float* a = new float[m * k];
    float* b = new float[k * n];
    float* c = new float[m * n];
    float *ad, *bd, *cd;

    for (int i=0; i < m*k; i++) {
        a[i] = 1.0;
    }
    for (int i=0;i<k*n;i++) {
        b[i] = 1.0;
    }
    gpuErrchk(cudaMalloc((void**)&ad, m * k * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&bd, k * n * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&cd, m * n * sizeof(float)));

    gpuErrchk(cudaMemcpy(ad, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bd, b, n * k * sizeof(float), cudaMemcpyHostToDevice));


    dim3 block(kernelSize, kernelSize);
    dim3 grid(ceil(float(n) / kernelSize), ceil(float(m) / kernelSize));


    cudaEvent_t start, stop;
    float elapsedTime;
    // gemmFast1
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    gemmFast1<<<grid, block, 2 * kernelSize * kernelSize * sizeof(float)>>>(ad, bd, cd, m, k, n);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << std::fixed << std::setprecision(2) << "gemmFast kernel time used: "
              << elapsedTime << "ms" << std::endl;
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c, cd, m *n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m *n; i++) {
        //printf("%d, %f\n", i, c[i]);
        // assert(c[i] == k);
    }


    // gemmVanilla
    gemmVanilla<<<grid, block>>>(ad, bd, cd, m, k, n);
    // std::cout << std::fixed << std::setprecision(2) << "gemmVanilla kernel time used: "
    //           << double(c_end - c_start) / CLOCKS_PER_SEC << " s" << std::endl;
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c, cd, m *n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m *n; i++)
        assert(c[i] == k);

    // matrixMul_noBankConflict
    // matrixMul_noBankConflict<<<grid, block>>>(cd, ad, bd, k, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c, cd, m *n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m *n; i++)
        assert(c[i] == k);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    delete []a;
    delete []b;
    delete []c;
}