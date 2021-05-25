cmake_minimum_required(VERSION 3.1)

# nvcc location
set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")

project(computation_playground LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit)

include_directories(
    common
    transpose2d
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

add_subdirectory(common)
add_subdirectory(transpose2d)