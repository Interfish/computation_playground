cmake_minimum_required(VERSION 3.1)

# nvcc location
set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")

project(computation_playground LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA)
include_directories(
    "${CUDA_INCLUDE_DIRS}"
)
add_subdirectory(common)
add_subdirectory(transpose2d)