#include <random>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <assert.h>
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


    // gemmFast1
    gemmFast1<<<grid, block, 2 * kernelSize * kernelSize * sizeof(float)>>>(ad, bd, cd, m, k, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c, cd, m *n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m *n; i++) {
        //printf("%d, %f\n", i, c[i]);
        // assert(c[i] == k);
    }


    // gemmVanilla
    gemmVanilla<<<grid, block>>>(ad, bd, cd, m, k, n);
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