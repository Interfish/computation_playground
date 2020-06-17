# CUDA
CUDA = nvcc
TARGET = $(patsubst %.cu, build/%_cuda.out, $(wildcard *.cu))
FLAGS = -std=c++11 -O2

all:$(TARGET)

${TARGET}:build/%_cuda.out:%.cu
	        ${CUDA} $< ${FLAGS} -o $@

$(info ${CUDA_TARGET})