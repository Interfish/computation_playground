file(GLOB SRC
    *.cpp
    *.cu
)

add_executable(info ${SRC})
target_link_libraries(info PRIVATE CUDA::cuda_driver CUDA::cudart)