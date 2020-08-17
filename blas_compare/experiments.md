```
threads = 1

./build/gemm.out 128 128 128
cblas_sgemm time used: 3.35 ms
Eigen time used: 0.41 ms

./build/gemm.out 256 256 256
cblas_sgemm time used: 3.00 ms
Eigen time used: 1.95 ms

./build/gemm.out 512 512 512
cblas_sgemm time used: 7.29 ms
Eigen time used: 14.42 ms

./build/gemm.out 1024 1024 1024
cblas_sgemm time used: 20.02 ms
Eigen time used: 101.58 ms

./build/gemm.out 2048 2048 2048
cblas_sgemm time used: 106.54 ms
Eigen time used: 719.00 ms

./build/gemm.out 4096 4096 4096
cblas_sgemm time used: 738.86 ms
Eigen time used: 5322.22 ms

./build/gemm.out 1 143 1000
cblas_sgemm time used: 3.45 ms
Eigen time used: 0.62 ms

./build/gemm.out 1 143 256
cblas_sgemm time used: 3.35 ms
Eigen time used: 0.17 ms
```

```
no threads limitation

./build/gemm.out 128 128 128
cblas_sgemm time used: 3.72 ms
Eigen time used: 0.33 ms

./build/gemm.out 256 256 256
cblas_sgemm time used: 4.47 ms
Eigen time used: 1.14 ms

./build/gemm.out 512 512 512
cblas_sgemm time used: 4.83 ms
Eigen time used: 6.51 ms

./build/gemm.out 1024 1024 1024
cblas_sgemm time used: 8.52 ms
Eigen time used: 41.30 ms

./build/gemm.out 2048 2048 2048
cblas_sgemm time used: 37.21 ms
Eigen time used: 280.45 ms

./build/gemm.out 4096 4096 4096
cblas_sgemm time used: 206.55 ms
Eigen time used: 1760.46 ms

./build/gemm.out 1 143 1000
cblas_sgemm time used: 3.29 ms
Eigen time used: 0.78 ms

./build/gemm.out 1 143 256
cblas_sgemm time used: 3.53 ms
Eigen time used: 0.15 ms

```