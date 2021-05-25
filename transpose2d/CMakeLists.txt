file(GLOB SRC
    *.cpp
    *.cu
)

add_executable(transpose2d ${SRC})
target_link_libraries(transpose2d PRIVATE CUDA::cuda_driver)