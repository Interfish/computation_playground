#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <ctime>
#include <cstring>
#include <iomanip>

#include <mkl.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using Eigen::Map;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::MatrixXf;

int main(int argc, char *argv[]) {
    int m, k, n, kernelSize;
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    n = atoi(argv[3]);

    float alpha = 1.0;
    float beta = 0.0;

    //Eigen::setNbThreads(1);
    //mkl_set_num_threads(1);

    float* amkl = (float*)mkl_malloc( m * k * sizeof(float), 64 );
    float* bmkl = (float*)mkl_malloc( k * n * sizeof(float), 64 );
    float* cmkl = (float*)mkl_malloc( m * n * sizeof(float), 64 );

    float* a = new float[m * k];
    float* b = new float[k * n];
    float* c = new float[m * n];

    for (int i=0; i < m*k; i++) {
        a[i] = 1.0;
        amkl[i] = 1.0;
    }
    for (int i=0; i < k * n; i++) {
        b[i] = 1.0;
        bmkl[i] = 1.0;
    }
    auto c_start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, amkl, k, bmkl, n, beta, cmkl, n);
    auto c_end = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(2) << "cblas_sgemm time used: "
              << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;

    for (int i = 0; i < m *n; i++)
        assert(cmkl[i] == k);

    c_start = std::chrono::high_resolution_clock::now();
    MatrixXf A = Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(a, m, k);
    MatrixXf B = Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(b, k, n);
    MatrixXf C = A * B;
    c_end = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(2) << "Eigen time used: "
              << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;
    for (int i = 0; i < m *n; i++)
        assert(C.data()[i] == k);

}