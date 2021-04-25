file(GLOB SRC
    *.cpp
    *.cu
)

add_executable(transpose2d ${SRC})