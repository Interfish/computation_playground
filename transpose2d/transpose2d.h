#include <cuda_runtime.h>

namespace computation_playground {

void transpose2d_naive(float* in, float* out, int m, int n, cudaStream_t stream);

void transpose2d_tile(float* in, float* out, int m, int n, int tile_m_dim, int tile_n_dim,
                      int m_direction_iteration, int n_direction_iteration, cudaStream_t stream);

} // namespace computation_playground