#include <iostream>

#include <cuda_runtime.h>

namespace computation_playground {

#define CUDA_CK(ret) \
{                                                                                               \
   if (ret != cudaSuccess) {                                                                    \
      std::cerr << "CUDA Error: " <<  cudaGetErrorString(code) << __FILE__                      \
         << std::to_string(__LINE__) << std::endl;                                              \
   }                                                                                            \
}

}