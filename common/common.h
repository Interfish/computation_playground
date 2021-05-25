#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

namespace computation_playground {

#define CUDA_CHECK(ret) \
{                                                                                               \
   if (ret != cudaSuccess) {                                                                    \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(ret) +          \
                               ", " + __FILE__ + ":" + std::to_string(__LINE__) + "\n");        \
   }                                                                                            \
}

}