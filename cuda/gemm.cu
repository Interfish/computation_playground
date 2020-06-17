#include <random>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"

#include "common.cuh"

__global__ void gemm(float* a, float* b, float* c, int m, int k, int n) {
    // a is a m x k matrix
    // b is a k x n matrix
    extern __shared__ float Tile[];
    float *aTile = Tile;
    float *bTile = Tile + blockDim.x * blockDim.y;

    // int tid = blockDim.x * threadIdx.y + threadIdx.x;
    // printf("%d %d %d\n", blockIdx.y, blockIdx.x, tid);

    int globalXA, globalYA, globalXB, globalYB, globalXC, globalYC;
    float *aPoint, *bPoint, accu = 0.0;
    for (int tileStart = 0; tileStart < k; tileStart += blockDim.x) {
        globalXA = tileStart + threadIdx.x;
        globalYA = blockIdx.y * blockDim.y + threadIdx.y;
        globalXB = blockIdx.x * blockDim.x + threadIdx.x;
        globalYB = tileStart + threadIdx.y;
        aPoint = aTile + threadIdx.y * blockDim.x + threadIdx.x;
        bPoint = bTile + threadIdx.y * blockDim.x + threadIdx.x;
        // printf("%d\n", tileStart);

        // printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

        // printf("%d %d %d %d \n", globalXA, globalYA, globalXB, globalYB);

        if (globalXA < k && globalYA < m) {
            *(aPoint) = *(a + globalYA * k + globalXA);
        } else {
            *(aPoint) = 0.0;
            // printf("%d %d %d %d \n", globalXA, globalYA, globalXB, globalYB);
        }
        if (globalXB < n && globalYB < k) {
            *(bPoint) = *(b + globalYB * n + globalXB);
        } else {
            *(bPoint) = 0.0;
        }
        __syncthreads();
        //printf("%d %d %d %d %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, *(aPoint));

        for (int i = 0; i < blockDim.x; i++) {
            accu += aTile[threadIdx.y * blockDim.x + i] * bTile[i * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    globalXC = blockIdx.x * blockDim.x + threadIdx.x;
    globalYC = blockIdx.y * blockDim.y + threadIdx.y;
    if(globalXC < n && globalYC < m) {
        c[globalYC * n + globalXC] = accu;
    }
}

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
    dim3 grid(ceil(float(m) / kernelSize), ceil(float(n) / kernelSize));

    // std::cout << ceil(float(m) / kernelSize) << ' ' << ceil(float(n) / kernelSize) << std::endl;

    gemm<<<grid, block, 2 * kernelSize * kernelSize * sizeof(float)>>>(ad, bd, cd, m, k, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(c, cd, m *n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m *n; i++) {
        assert(c[i] == k);
    }

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    delete []a;
    delete []b;
    delete []c;
}