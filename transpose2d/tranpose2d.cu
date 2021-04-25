namespace computation_playground {

void transpose2d_naive(float *tensor, float* out, int m, int n) {
  dim3 block(1);
  transpose2d_naive<<<1, block>>>(tensor, out, m, n);
}

__global__ void transpose2d_naive(float *tensor, float* out, int m, int n) { }

}