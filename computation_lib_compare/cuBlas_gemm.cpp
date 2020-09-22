//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <assert.h>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../cuda/common.cuh"


int main (int argc, char *argv[]){
    int m, k, n, kernelSize;
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);

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

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;


    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;


    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, ad, m, bd, k, &beta, cd, m);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << std::fixed << std::setprecision(2) << "gemmFast kernel time used: "
              << elapsedTime << "ms" << std::endl;

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("gemm failed\n");
    }

    gpuErrchk(cudaMemcpy(c, cd, m *n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m * n; i++) {
        // printf("%f ", c[i]);
        assert(c[i] == k);
    }

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    delete []a;
    delete []b;
    delete []c;
    return EXIT_SUCCESS;
}