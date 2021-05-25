#include <cmath>

#include <cuda_runtime.h>

namespace computation_playground {

__global__ void transpose2d_naive_kernel(float* in, float* out, int m, int n) {
  int in_row_offet = blockIdx.x * blockDim.x + threadIdx.x;
  if(in_row_offet < m) {
    int in_global_offset = blockIdx.y * m + in_row_offet;
    int out_global_offset = in_row_offet * n + blockIdx.y;
    *(out + out_global_offset) = *(in + in_global_offset);
  }
}

void transpose2d_naive(float* in, float* out, int m, int n, cudaStream_t stream) {
  int threads_per_block = 32;
  int blocks_per_row = std::ceil(m / float(threads_per_block));
  dim3 grid(blocks_per_row, n);
  transpose2d_naive_kernel<<<grid, threads_per_block, 0, stream>>>(in, out, m, n);
}

__global__ void transpose2d_tile_kernel(float* in, float* out, int m, int n,
                                        int m_direction_iteration, int n_direction_iteration) {
  for(int j = 0; j < n_direction_iteration; j++) {
    for(int i = 0; i < m_direction_iteration; i++) {
      int x = blockIdx.x * blockDim.x * m_direction_iteration + i * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y * n_direction_iteration + j * blockDim.y + threadIdx.y;
      if(x < m && y < n) {
        out[x * n + y] = in[y * m + x];
      }
    }
  }
}

void transpose2d_tile(float* in, float* out, int m, int n, int tile_m_dim, int tile_n_dim,
                      int m_direction_iteration, int n_direction_iteration, cudaStream_t stream) {
  dim3 grid(std::ceil(m / (tile_m_dim * m_direction_iteration)), std::ceil(n / (tile_n_dim * n_direction_iteration)));
  dim3 block(tile_m_dim, tile_n_dim);
  transpose2d_tile_kernel<<<grid, block, 0, stream>>>(in, out, m, n, m_direction_iteration, n_direction_iteration);
}

} // namespace computation_playground