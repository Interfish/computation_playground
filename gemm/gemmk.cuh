#include "cuda_runtime.h"

__global__ void gemmFast1(float* a, float* b, float* c, int m, int k, int n) {
    // a is a m x k matrix
    // b is a k x n matrix
    extern __shared__ float Tile[];
    float *aTile = Tile;
    float *bTile = Tile + blockDim.x * blockDim.y;

    // int tid = blockDim.x * threadIdx.y + threadIdx.x;
    // printf("%d %d %d\n", threadIdx.y, threadIdx.x, tid);

    int globalXA, globalYA, globalXB, globalYB, globalXC, globalYC;
    int offside = threadIdx.y * blockDim.x + threadIdx.x;
    float *aPoint = aTile + offside;
    float *bPoint = bTile + offside;
    globalYA = blockIdx.y * blockDim.y + threadIdx.y;
    globalXB = blockIdx.x * blockDim.x + threadIdx.x;
    float accu = 0.0;
    for (int tileStart = 0; tileStart < k; tileStart += blockDim.x) {
        globalXA = tileStart + threadIdx.x;
        globalYB = tileStart + threadIdx.y;
        // printf("%d\n", tileStart);

        // printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

        // printf("%d %d %d %d \n", globalXA, globalYA, globalXB, globalYB);

        if (globalXA < k && globalYA < m) {
            // printf("%d, %d, %d, %d\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);
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

        for (int index = 0; index < blockDim.x; index++) {
            //accu += float(threadIdx.y * blockDim.x + index + index * blockDim.x + threadIdx.x);
            accu += aTile[threadIdx.y * blockDim.x + index] * bTile[index * blockDim.x + threadIdx.x];
        }
        //__syncthreads();
    }
    // printf("%d %d %d %d %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, accu);
    globalXC = blockIdx.x * blockDim.x + threadIdx.x;
    globalYC = blockIdx.y * blockDim.y + threadIdx.y;
    if(globalXC < n && globalYC < m) {
        c[globalYC * n + globalXC] = accu;
    }
}

__global__ void gemmVanilla(float* a, float* b, float* c, int m, int k, int n) {
    int globalY = blockDim.y * blockIdx.y + threadIdx.y;
    int globalX = blockDim.x * blockIdx.x + threadIdx.x;
    float accu = 0.0;
    if (globalY < m && globalX < n) {
        for(int i = 0; i < k; i++)
            accu += a[globalY * k + i] * b[i * n + globalX];
        c[globalY * n + globalX] = accu;
    }
}


#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
#define BLOCK_SIZE 16

__global__ void
matrixMul_noBankConflict( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {


        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
	  Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}
